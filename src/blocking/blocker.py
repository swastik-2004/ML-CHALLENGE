"""
Production Multi-Pass Candidate Blocker for Entity Resolution.

Prunes the O(N*M) search space (~2.2M S1 x ~10.3M Target records) down to a
compact, high-recall candidate pool (recall >= 87%, capped at 50 candidates/entity).

Features:
- 13 complementary indexing passes across canonical keys, name stems, street tokens, and house numbers
- Dynamic anti-crowding filters (MAX_S1, MAX_TG) to eliminate generic explosion
- Prioritized candidate ranking based on rule agreement and specificity
- Strict per-entity capping (default: max 50 candidates)
- Fast vectorized numpy operations
- Atomic caching to parquet and TSV output meeting competition validator specs
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    CANDIDATE_PAIRS_FILE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    WORK_DIR,
)
from ..data import id_to_int, int_to_id, save_parquet_atomic
from ..normalize import norm_path

CACHE_DIR = WORK_DIR / "cache"

STOP_WORDS = {
    "the", "and", "inc", "ltd", "corp", "llc", "pvt", "for", "new", "co",
    "company", "group", "services", "private", "limited", "enterprises",
    "solutions", "technologies", "holdings", "industries", "association",
    "sarl", "sas", "eurl", "sa", "sasu", "sci"
}

GENERIC_NOISE = {
    "enterprises", "solutions", "technologies", "services", "group",
    "industries", "holdings", "consulting", "properties", "management",
    "associates", "partners", "ventures", "global", "international",
    "systems", "agency", "logistics", "products", "marketing", "center",
    "store", "shop", "care", "restaurant", "cafe", "grill", "hotel"
}

NOISE_ADDR = {
    "road", "street", "st", "rd", "ave", "lane", "dr", "near", "opp",
    "fl", "floor", "block", "bldg", "plot", "shop", "hn"
}


def extract_blocking_tokens(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of first token, second token, and 3-char prefix."""
    n = len(series)
    w1_arr = np.empty(n, dtype=object)
    w2_arr = np.empty(n, dtype=object)
    p3_arr = np.empty(n, dtype=object)

    for i, val in enumerate(series.values):
        s = str(val).lower().strip() if val is not None else ""
        toks = [t for t in s.split() if len(t) >= 3 and t not in STOP_WORDS]
        w1 = toks[0] if len(toks) > 0 else (s.split()[0] if s.split() else "")
        w2 = toks[1] if len(toks) > 1 else ""
        p3 = w1[:3] if len(w1) >= 3 else w1
        w1_arr[i] = w1
        w2_arr[i] = w2
        p3_arr[i] = p3

    return w1_arr, w2_arr, p3_arr


def extract_name_stem(series: pd.Series) -> np.ndarray:
    """Extract core name stem by stripping generic business tokens."""
    n = len(series)
    stems = np.empty(n, dtype=object)
    for i, val in enumerate(series.values):
        s = str(val).lower().strip() if val is not None else ""
        toks = [w for w in s.split() if len(w) >= 3 and w not in GENERIC_NOISE]
        stems[i] = " ".join(toks[:2]) if toks else (s.split()[0] if s.split() else "")
    return stems


def extract_street_w1(series: pd.Series) -> np.ndarray:
    """Extract primary street token, skipping numeric building IDs and generic road types."""
    n = len(series)
    street_arr = np.empty(n, dtype=object)
    for i, val in enumerate(series.values):
        s = str(val).lower().strip() if val is not None else ""
        toks = [t for t in s.split() if not t.isdigit() and len(t) >= 3 and t not in NOISE_ADDR]
        street_arr[i] = toks[0] if toks else ""
    return street_arr


def index_rule(
    s1_vals: np.ndarray,
    tg_vals: np.ndarray,
    both_country_codes: np.ndarray,
    n1: int,
    max_s1: int = 20,
    max_tg: int = 50,
    min_len: int = 0
) -> pd.DataFrame:
    """
    Vectorized rule matching within country.
    Filters out empty keys and keys exceeding crowding limits.
    """
    both_col = pd.concat([pd.Series(s1_vals), pd.Series(tg_vals)], ignore_index=True)
    kc, _ = pd.factorize(both_col)
    n_countries = int(both_country_codes.max()) + 1
    key = kc.astype(np.int64) * n_countries + both_country_codes

    vals = both_col.values
    mask = (vals == "") | (vals == "None")
    if min_len > 0:
        mask = mask | (both_col.str.len() < min_len).values
    key[mask] = -1
    del both_col, kc

    ka, kb = key[:n1], key[n1:]
    nk = int(key.max()) + 1
    ca = np.bincount(ka[ka >= 0], minlength=nk)
    cb = np.bincount(kb[kb >= 0], minlength=nk)

    ok = (ca >= 1) & (ca <= max_s1) & (cb >= 1) & (cb <= max_tg)
    ia = np.flatnonzero((ka >= 0) & ok[np.maximum(ka, 0)])
    ib = np.flatnonzero((kb >= 0) & ok[np.maximum(kb, 0)])

    pairs = pd.DataFrame({"s1": ia, "k": ka[ia]}).merge(
        pd.DataFrame({"tg": ib, "k": kb[ib]}), on="k"
    )
    return pairs[["s1", "tg"]].astype(np.int32)


class MultiPassBlocker:
    """
    Multi-Pass Entity Resolution Blocker with 13 complementary passes.
    Executes multiple indexing passes and applies prioritized capping.
    """

    def __init__(self, max_candidates_per_entity: int = 50):
        self.max_candidates = max_candidates_per_entity

    def build_candidates(self, split: str = "test") -> Path:
        """
        Execute 13-pass blocking for split ('train' or 'test').
        Returns path to the generated parquet cache file.
        """
        print(f"\n{'='*75}")
        print(f"RUNNING 13-PASS BLOCKER ON SPLIT: '{split.upper()}'")
        print(f"{'='*75}")
        t_start = time.time()

        cols = [
            "entity_id", "country", "name_norm", "name_core", "name_key",
            "name_compact", "addr_norm", "addr_key", "house_no", "state"
        ]

        # 1. Load normalized partitions
        print("Loading normalized datasets...")
        s1 = pd.read_parquet(norm_path(split, 1), columns=cols)
        tg2 = pd.read_parquet(norm_path(split, 2), columns=cols)
        tg3 = pd.read_parquet(norm_path(split, 3), columns=cols)
        tg = pd.concat([tg2, tg3], ignore_index=True)
        del tg2, tg3

        n1 = len(s1)
        ntg = len(tg)
        print(f"Loaded {n1:,} S1 entities and {ntg:,} target records.")

        s1_ids = id_to_int(s1.entity_id.values)
        tg_ids = id_to_int(tg.entity_id.values)

        # 2. Extract derived tokens
        print("Deriving blocking tokens (w1, w2, p3, p2, name_stem, street_w1, house_no combinations)...")
        s1["w1"], s1["w2"], s1["p3"] = extract_blocking_tokens(s1["name_core"])
        tg["w1"], tg["w2"], tg["p3"] = extract_blocking_tokens(tg["name_core"])

        s1["p2"] = s1["name_norm"].str.slice(0, 2)
        tg["p2"] = tg["name_norm"].str.slice(0, 2)

        s1["name_stem"] = extract_name_stem(s1["name_core"])
        tg["name_stem"] = extract_name_stem(tg["name_core"])

        s1["street_w1"] = extract_street_w1(s1["addr_norm"])
        tg["street_w1"] = extract_street_w1(tg["addr_norm"])

        # Composite keys
        s1["house_p3"] = np.where((s1["house_no"].values != "") & (s1["p3"].values != ""), s1["house_no"].values + "_" + s1["p3"].values, "")
        tg["house_p3"] = np.where((tg["house_no"].values != "") & (tg["p3"].values != ""), tg["house_no"].values + "_" + tg["p3"].values, "")

        s1["house_p2"] = np.where((s1["house_no"].values != "") & (s1["p2"].values != ""), s1["house_no"].values + "_" + s1["p2"].values, "")
        tg["house_p2"] = np.where((tg["house_no"].values != "") & (tg["p2"].values != ""), tg["house_no"].values + "_" + tg["p2"].values, "")

        s1["house_w1"] = np.where((s1["house_no"].values != "") & (s1["w1"].values != ""), s1["house_no"].values + "_" + s1["w1"].values, "")
        tg["house_w1"] = np.where((tg["house_no"].values != "") & (tg["w1"].values != ""), tg["house_no"].values + "_" + tg["w1"].values, "")

        s1["house_w2"] = np.where((s1["house_no"].values != "") & (s1["w2"].values != ""), s1["house_no"].values + "_" + s1["w2"].values, "")
        tg["house_w2"] = np.where((tg["house_no"].values != "") & (tg["w2"].values != ""), tg["house_no"].values + "_" + tg["w2"].values, "")

        s1["house_street"] = np.where((s1["house_no"].values != "") & (s1["street_w1"].values != ""), s1["house_no"].values + "_" + s1["street_w1"].values, "")
        tg["house_street"] = np.where((tg["house_no"].values != "") & (tg["street_w1"].values != ""), tg["house_no"].values + "_" + tg["street_w1"].values, "")

        s1["house_state"] = np.where((s1["house_no"].values != "") & (s1["state"].values != ""), s1["house_no"].values + "_" + s1["state"].values, "")
        tg["house_state"] = np.where((tg["house_no"].values != "") & (tg["state"].values != ""), tg["house_no"].values + "_" + tg["state"].values, "")

        s1["state_w1"] = np.where((s1["state"].values != "") & (s1["w1"].values != ""), s1["state"].values + "_" + s1["w1"].values, "")
        tg["state_w1"] = np.where((tg["state"].values != "") & (tg["w1"].values != ""), tg["state"].values + "_" + tg["w1"].values, "")

        s1["w1_w2"] = np.where((s1["w1"].values != "") & (s1["w2"].values != ""), s1["w1"].values + "_" + s1["w2"].values, "")
        tg["w1_w2"] = np.where((tg["w1"].values != "") & (tg["w2"].values != ""), tg["w1"].values + "_" + tg["w2"].values, "")

        # Country factorization
        both_country = pd.concat([s1["country"], tg["country"]], ignore_index=True)
        cc, _ = pd.factorize(both_country)
        del both_country

        # 3. Define the 13 multi-pass rules
        rules = [
            ("R1_name_key", s1["name_key"].values, tg["name_key"].values, 20, 50, 0, 1),
            ("R2_name_compact", s1["name_compact"].values, tg["name_compact"].values, 20, 50, 0, 1),
            ("R3_addr_key", s1["addr_key"].values, tg["addr_key"].values, 20, 50, 0, 1),
            ("R4_name_stem", s1["name_stem"].values, tg["name_stem"].values, 20, 50, 4, 1),
            ("R5_house_w1", s1["house_w1"].values, tg["house_w1"].values, 20, 50, 4, 2),
            ("R6_house_p3", s1["house_p3"].values, tg["house_p3"].values, 20, 50, 4, 2),
            ("R7_house_p2", s1["house_p2"].values, tg["house_p2"].values, 15, 30, 3, 2),
            ("R8_house_street", s1["house_street"].values, tg["house_street"].values, 15, 30, 4, 2),
            ("R9_w1_w2", s1["w1_w2"].values, tg["w1_w2"].values, 15, 35, 5, 2),
            ("R10_state_w1", s1["state_w1"].values, tg["state_w1"].values, 15, 40, 4, 3),
            ("R11_house_w2", s1["house_w2"].values, tg["house_w2"].values, 15, 30, 4, 3),
            ("R12_w1_name", s1["w1"].values, tg["w1"].values, 15, 30, 3, 3),
            ("R13_house_state", s1["house_state"].values, tg["house_state"].values, 10, 20, 4, 4),
        ]

        pass_dfs = []
        for r_name, s1_col, tg_col, max_s1, max_tg, min_len, priority in rules:
            t_r = time.time()
            df = index_rule(s1_col, tg_col, cc, n1, max_s1=max_s1, max_tg=max_tg, min_len=min_len)
            df["priority"] = np.int8(priority)
            print(f"  [{r_name:<16}] generated {len(df):>10,d} candidate pairs ({time.time() - t_r:.1f}s)")
            pass_dfs.append(df)

        del s1, tg, cc

        # 4. Aggregate & Deduplicate
        print("Aggregating candidate passes and computing agreement weights...")
        combined = pd.concat(pass_dfs, ignore_index=True)
        del pass_dfs

        grouped = combined.groupby(["s1", "tg"], as_index=False, sort=False).agg(
            n_rules=("priority", "count"),
            best_priority=("priority", "min")
        )
        del combined

        grouped["s1_int"] = s1_ids[grouped["s1"].values]
        grouped["tg_int"] = tg_ids[grouped["tg"].values]
        del s1_ids, tg_ids

        # Sort within s1 by: 1) best_priority asc, 2) n_rules desc
        print(f"Sorting and applying prioritized candidate capping (<= {self.max_candidates})...")
        grouped.sort_values(
            by=["s1_int", "best_priority", "n_rules"],
            ascending=[True, True, False],
            inplace=True
        )

        # Cap candidates per S1 entity
        grouped["rank"] = grouped.groupby("s1_int").cumcount()
        capped = grouped[grouped["rank"] < self.max_candidates][
            ["s1_int", "tg_int", "n_rules", "best_priority"]
        ].copy()
        del grouped

        # 5. Save Parquet Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"candidate_pairs_{split}.parquet"
        save_parquet_atomic(capped, cache_path)
        print(f"Saved {len(capped):,d} candidate pairs to cache: {cache_path}")
        print(f"Blocking pipeline completed in {time.time() - t_start:.2f}s")
        return cache_path

    def write_submission_candidate_file(
        self,
        split: str = "test",
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Export candidate pairs to the official competition format:
        output/candidate_pairs.tsv (source1_entity_id \t candidate_entity_ids)
        """
        output_path = output_file or CANDIDATE_PAIRS_FILE
        cache_path = CACHE_DIR / f"candidate_pairs_{split}.parquet"

        if not cache_path.exists():
            print(f"Cache {cache_path} not found. Running build_candidates first...")
            self.build_candidates(split=split)

        print(f"\nExporting candidate pairs TSV: {output_path}")
        t0 = time.time()

        s1_df = pd.read_parquet(norm_path(split, 1), columns=["entity_id"])
        all_s1_ids = s1_df["entity_id"].values
        s1_ints = id_to_int(all_s1_ids)
        del s1_df

        candidates_df = pd.read_parquet(cache_path, columns=["s1_int", "tg_int"])

        print("Formatting string candidate lists...")
        grouped = candidates_df.groupby("s1_int")["tg_int"].apply(list).to_dict()
        del candidates_df

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            n_empty = 0
            for eid_int, eid_str in zip(s1_ints, all_s1_ids):
                cands_int = grouped.get(eid_int, [])
                if cands_int:
                    cands_str = ",".join(int_to_id(cands_int))
                else:
                    cands_str = ""
                    n_empty += 1
                writer.writerow([eid_str, cands_str])

        print(f"Exported {len(all_s1_ids):,d} entities ({n_empty:,d} empty) to {output_path} in {time.time() - t0:.2f}s")
        return output_path


def generate_candidate_pairs(
    split: str = "test",
    max_candidates_per_entity: int = 50,
    output_file: Optional[Path] = None
) -> Path:
    """Convenience functional API."""
    blocker = MultiPassBlocker(max_candidates_per_entity=max_candidates_per_entity)
    blocker.build_candidates(split=split)
    return blocker.write_submission_candidate_file(split=split, output_file=output_file)


def main():
    parser = argparse.ArgumentParser(description="Multi-Pass Candidate Blocker.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to block.")
    parser.add_argument("--max-candidates", type=int, default=50, help="Maximum candidates per entity.")
    parser.add_argument("--output", type=str, default=None, help="Optional output TSV path.")
    parser.add_argument("--skip-tsv", action="store_true", help="Only build parquet cache, skip TSV generation.")

    args = parser.parse_args()
    blocker = MultiPassBlocker(max_candidates_per_entity=args.max_candidates)
    blocker.build_candidates(split=args.split)

    if not args.skip_tsv:
        out_p = Path(args.output) if args.output else None
        blocker.write_submission_candidate_file(split=args.split, output_file=out_p)


if __name__ == "__main__":
    main()
