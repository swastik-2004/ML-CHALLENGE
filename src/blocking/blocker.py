"""
Production Multi-Pass Candidate Blocker for Entity Resolution.

Prunes the O(N*M) search space (~2.2M S1 x ~10.3M target records in train) to a compact,
high-recall candidate pool. Measured on the 100k dev S1s (python -m src.blocking.check_recall):
94.3% of true pairs kept at cap 20 (18.7 candidates/S1), perfect-classifier macro F0.5 ceiling
0.98 (was 84.1% / 0.925 with the previous 13-rule blocker).

- 15 same-country key rules (RULES): exact name/address keys, name tokens, house-number combos,
  rare address tokens, name prefix + state
- crowded keys are narrowed by extra discriminators (state, rare address token, soundex, prefix)
  instead of being dropped
- candidates are ranked by name/address similarity, key specificity and rule agreement before the
  per-S1 cap; the cache stores the rank so smaller caps are simple filters
- atomic parquet cache + competition-format candidate_pairs.tsv
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    CANDIDATE_PAIRS_FILE,
    DEV_IDS_FILE,
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


def _map_unique(values, fn) -> np.ndarray:
    """Apply a python fn once per distinct value (names/addresses repeat a lot)."""
    codes, uniq = pd.factorize(pd.Series(values, dtype=object).fillna(""))
    mapped = np.array([fn(u) for u in uniq], dtype=object)
    return mapped[codes] if len(mapped) else np.array([], dtype=object)


_SOUNDEX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
            "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(word: str) -> str:
    """American Soundex (4 chars). Catches transliteration variants: 'shiva' ~ 'siv' ~ 'shiv'."""
    w = "".join(ch for ch in str(word).lower() if "a" <= ch <= "z")
    if not w:
        return ""
    out, prev = w[0], _SOUNDEX.get(w[0], "")
    for ch in w[1:]:
        d = _SOUNDEX.get(ch, "")
        if d and d != prev:
            out += d
        if ch not in "hw":
            prev = d
    return (out + "000")[:4]


def rare_address_tokens(addr: np.ndarray, min_len: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """(rarest token, 'two rarest tokens') of each address, by document frequency over `addr`.

    Only tokens that occur in >= 2 addresses qualify (a token seen once can never match anything),
    and numbers / generic street words are skipped. Distinctive place tokens ('glitterati',
    'makhmalabad', 'creekedge') survive name transliteration noise, so they catch pairs whose
    names share no key at all."""
    def toks(a):
        return {t for t in a.split() if len(t) >= min_len and not t.isdigit() and t not in NOISE_ADDR}

    codes, uniq = pd.factorize(pd.Series(addr, dtype=object).fillna(""))
    counts = np.bincount(codes, minlength=len(uniq))           # how many records share each address
    df = Counter()
    for u, c in zip(uniq, counts):
        for t in toks(u):
            df[t] += int(c)

    def pick(u):
        ts = sorted((t for t in toks(u) if df[t] >= 2), key=lambda t: (df[t], t))
        return (ts[0] if ts else "", " ".join(sorted(ts[:2])) if len(ts) >= 2 else "")

    picked = [pick(u) for u in uniq]
    r1 = np.array([a for a, _ in picked], dtype=object)[codes]
    r2 = np.array([b for _, b in picked], dtype=object)[codes]
    return r1, r2


def _combine(*arrays) -> np.ndarray:
    """'a|b|c' per row, '' when any part is empty."""
    out = np.asarray(arrays[0], dtype=object)
    ok = out != ""
    for a in arrays[1:]:
        a = np.asarray(a, dtype=object)
        ok &= a != ""
        out = out + "|" + a
    return np.where(ok, out, "").astype(object)


def _match_keys(key: np.ndarray, n1: int, max_s1: int, max_tg: int):
    """Join S1 rows (key[:n1]) with target rows (key[n1:]) on equal key >= 0, keeping only keys with
    1..max_s1 S1 rows and 1..max_tg target rows. Returns (s1, tg, blk, crowded_key_mask)."""
    ka, kb = key[:n1], key[n1:]
    nk = int(key.max()) + 1 if len(key) and key.max() >= 0 else 0
    if nk == 0:
        e = np.array([], dtype=np.int32)
        return e, e, e, np.zeros(0, bool)
    ca = np.bincount(ka[ka >= 0], minlength=nk)
    cb = np.bincount(kb[kb >= 0], minlength=nk)
    ok = (ca >= 1) & (ca <= max_s1) & (cb >= 1) & (cb <= max_tg)
    crowded = (ca >= 1) & (cb >= 1) & ~ok

    ia = np.flatnonzero((ka >= 0) & ok[np.maximum(ka, 0)])
    ib = np.flatnonzero((kb >= 0) & ok[np.maximum(kb, 0)])
    ka_valid, kb_valid = ka[ia], kb[ib]
    order_b = np.argsort(kb_valid, kind="stable")
    kb_sorted, ib_sorted = kb_valid[order_b], ib[order_b]
    b_starts = np.searchsorted(kb_sorted, ka_valid, side="left")
    counts = np.searchsorted(kb_sorted, ka_valid, side="right") - b_starts
    m = counts > 0
    ia_v, cnt_v, st_v = ia[m], counts[m], b_starts[m]
    s1_res = np.repeat(ia_v.astype(np.int32), cnt_v)
    offsets = np.repeat(st_v, cnt_v) + (np.arange(len(s1_res)) - np.repeat(np.cumsum(cnt_v) - cnt_v, cnt_v))
    tg_res = ib_sorted[offsets].astype(np.int32)
    blk = np.repeat(cb[ka_valid[m]].astype(np.int32), cnt_v)   # targets sharing the key: specificity
    return s1_res, tg_res, blk, crowded


def index_rule(
    s1_vals: np.ndarray,
    tg_vals: np.ndarray,
    both_country_codes: np.ndarray,
    n1: int,
    max_s1: int = 20,
    max_tg: int = 50,
    min_len: int = 0,
    narrow: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    Vectorized rule matching within country.
    Filters out empty keys and keys exceeding crowding limits.

    narrow: optional discriminators [(s1_disc, tg_disc), ...]. Instead of dropping a crowded key
    (too many S1 or target rows share it), its rows are re-keyed on (key, discriminator) and matched
    again under the same limits; keys still crowded move on to the next discriminator. Rows with an
    empty discriminator drop out. Example: first word 'sharma' is far too common on its own, but
    'sharma' + rare address token 'makhmalabad' is specific.

    Returns columns s1, tg (row positions) and blk (number of target rows sharing the matched key;
    small = specific evidence, used to rank candidates before the cap).
    """
    both_col = pd.concat([pd.Series(s1_vals, dtype=object), pd.Series(tg_vals, dtype=object)],
                         ignore_index=True).fillna("")
    kc, _ = pd.factorize(both_col)
    n_countries = int(both_country_codes.max()) + 1
    key = kc.astype(np.int64) * n_countries + both_country_codes

    vals = both_col.values
    mask = (vals == "") | (vals == "None")
    if min_len > 0:
        mask = mask | (both_col.str.len() < min_len).values
    key[mask] = -1
    del both_col, kc

    s1_parts, tg_parts, blk_parts = [], [], []
    s, t, b, crowded = _match_keys(key, n1, max_s1, max_tg)
    s1_parts.append(s); tg_parts.append(t); blk_parts.append(b)

    for disc_s1, disc_tg in (narrow or []):
        rows = (key >= 0) & crowded[np.maximum(key, 0)] if len(crowded) else np.zeros(len(key), bool)
        if not rows.any():
            break
        disc = pd.concat([pd.Series(disc_s1, dtype=object), pd.Series(disc_tg, dtype=object)],
                         ignore_index=True).fillna("").values
        dc, _ = pd.factorize(disc)
        rows &= disc != ""
        combo = np.where(rows, key * (int(dc.max()) + 2) + dc, -1)
        new_key, _ = pd.factorize(combo)
        new_key = new_key.astype(np.int64)
        new_key[~rows] = -1
        key = new_key
        s, t, b, crowded = _match_keys(key, n1, max_s1, max_tg)
        s1_parts.append(s); tg_parts.append(t); blk_parts.append(b)

    return pd.DataFrame({"s1": np.concatenate(s1_parts), "tg": np.concatenate(tg_parts),
                         "blk": np.concatenate(blk_parts)})


# ----------------------------------------------------------------------------------------------
# Rule table. (name, key column, max_s1, max_tg, min_len, priority, narrow-by columns)
# priority: 1 = strongest evidence. narrow: discriminator columns tried in order for crowded keys.
# Key columns are derived in MultiPassBlocker._derive_keys.
# ----------------------------------------------------------------------------------------------
RULES = [
    ("R1_name_key",       "name_key",     20, 50, 0, 1, ["state", "ra1"]),
    ("R2_name_compact",   "name_compact", 20, 50, 0, 1, ["state", "ra1"]),
    ("R3_addr_key",       "addr_key",     20, 50, 0, 1, []),
    ("R4_name_stem",      "name_stem",    20, 50, 4, 1, ["state", "ra1"]),
    ("R5_house_w1",       "house_w1",     20, 50, 4, 2, []),
    ("R6_house_p3",       "house_p3",     20, 50, 4, 2, []),
    ("R7_house_p2",       "house_p2",     15, 30, 3, 2, []),
    ("R8_house_street",   "house_street", 15, 30, 4, 2, []),
    ("R9_w1_w2",          "w1_w2",        15, 35, 5, 2, ["state", "ra1"]),
    ("R11_house_w2",      "house_w2",     15, 30, 4, 3, []),
    ("R12_w1_name",       "w1",           15, 30, 3, 3, ["state", "ra1"]),
    ("R13_house_state",   "house_state",  10, 20, 4, 4, []),
    # R10 (state + first word) and R16 (soundex + state) were measured on the dev set and removed:
    # 24 and 11 true pairs no other rule finds, for ~0.7M extra candidates each.
    # new rules for pairs no key above can match (transliterated names, typos)
    ("R14_rare_addr2",    "ra2",          20, 50, 0, 2, ["sx"]),
    ("R15_rare_addr",     "ra1",          20, 50, 0, 3, ["sx", "p4"]),
    ("R17_p4_state",      "p4_state",     20, 50, 0, 4, ["ra1"]),
]


def rank_score(g: pd.DataFrame) -> np.ndarray:
    """Higher = keep first when the cap cuts an S1's candidate list:
    name + address similarity (0-100 each) + key specificity + rule agreement.

    Tuned on the uncapped dev pairs (src.blocking.check_recall / precap_dev_train.parquet): at cap 20
    this keeps 94.3% of true pairs vs 91.3% for the old order (rule priority, then n_rules), with
    the same number of candidates. Strict rule priority is deliberately not used: a crowded
    exact-name match is weaker evidence than a unique rare-address match with a similar name."""
    spec = 1.0 / np.log2(2.0 + g["min_blk"].values)          # 1 for a unique key, -> 0 when crowded
    return (g["name_sim"].values.astype(np.float64) + g["addr_sim"].values
            + 250.0 * spec + 15.0 * g["n_rules"].values)


SIM_CHUNK = 2_000_000   # pairs per rapidfuzz cpdist call


def precap_dev_path(split: str) -> Path:
    return CACHE_DIR / f"precap_dev_{split}.parquet"


class MultiPassBlocker:
    """
    Multi-pass entity resolution blocker.

    1. Every rule in RULES joins S1 and target records on an equal key within the same country.
       Crowded keys are narrowed by extra discriminators instead of being dropped.
    2. Pairs from all rules are merged: n_rules (how many rules agree), best_priority, rules_mask
       (which rules), min_blk (size of the most specific matching block).
    3. Candidates of S1s with more than `sim_min` pairs (default 20, the smallest cap we use: below
       it every candidate is kept whatever the order) get a quick name + address similarity
       (rapidfuzz token_set_ratio), and every S1's candidates are ranked by rank_score (similarity,
       key specificity, rule agreement), then capped at `max_candidates`.
    4. The cache keeps the rank, so smaller caps are `rank < cap` filters; on the train split the
       uncapped pairs of the dev S1s are also saved for src.blocking.check_recall.
    """

    def __init__(self, max_candidates_per_entity: int = 20, sim_min: int = 20):
        self.max_candidates = max_candidates_per_entity
        self.sim_min = sim_min

    @staticmethod
    def _derive_keys(d: pd.DataFrame) -> pd.DataFrame:
        d["w1"], d["w2"], d["p3"] = extract_blocking_tokens(d["name_core"])
        d["p2"] = d["name_norm"].str.slice(0, 2)
        d["name_stem"] = extract_name_stem(d["name_core"])
        d["street_w1"] = extract_street_w1(d["addr_norm"])
        d["sx"] = _map_unique(d["w1"].values, soundex)
        d["p4"] = d["name_core"].str.replace(" ", "", regex=False).str.slice(0, 4).values
        h, st = d["house_no"].values, d["state"].values
        d["house_p3"] = _combine(h, d["p3"].values)
        d["house_p2"] = _combine(h, d["p2"].values)
        d["house_w1"] = _combine(h, d["w1"].values)
        d["house_w2"] = _combine(h, d["w2"].values)
        d["house_street"] = _combine(h, d["street_w1"].values)
        d["house_state"] = _combine(h, st)
        d["w1_w2"] = _combine(d["w1"].values, d["w2"].values)
        d["p4_state"] = _combine(d["p4"].values, st)
        return d

    def build_candidates(self, split: str = "test") -> Path:
        """
        Execute multi-pass blocking for split ('train' or 'test').
        Returns path to the generated parquet cache file.
        """
        import gc
        from rapidfuzz import fuzz
        from rapidfuzz.process import cpdist

        print(f"\n{'='*75}")
        print(f"RUNNING {len(RULES)}-PASS BLOCKER ON SPLIT: '{split.upper()}' (cap {self.max_candidates})")
        print(f"{'='*75}")
        t_start = time.time()

        cols = ["entity_id", "country", "name_norm", "name_core", "name_key",
                "name_compact", "addr_norm", "addr_key", "house_no", "state"]

        # 1. Load normalized partitions (S1 rows first, then S2 + S3)
        print("Loading normalized datasets...")
        s1 = pd.read_parquet(norm_path(split, 1), columns=cols)
        tg = pd.concat([pd.read_parquet(norm_path(split, s), columns=cols) for s in (2, 3)],
                       ignore_index=True)
        n1, ntg = len(s1), len(tg)
        print(f"Loaded {n1:,} S1 entities and {ntg:,} target records.")
        s1_ids = id_to_int(s1.entity_id.values)
        tg_ids = id_to_int(tg.entity_id.values)
        d = pd.concat([s1, tg], ignore_index=True).fillna("")
        del s1, tg
        d.drop(columns=["entity_id"], inplace=True)
        gc.collect()

        # 2. Derived keys (computed once on S1 + targets together)
        t_k = time.time()
        print("Deriving blocking keys...")
        d = self._derive_keys(d)
        d["ra1"], d["ra2"] = rare_address_tokens(d["addr_norm"].values)
        cc, _ = pd.factorize(d["country"])
        print(f"  keys ready ({time.time() - t_k:.0f}s)")

        # 3. Run the rules
        pass_s1, pass_tg, pass_bit, pass_prio, pass_blk = [], [], [], [], []
        for bit, (r_name, col, max_s1, max_tg, min_len, priority, narrow_cols) in enumerate(RULES):
            t_r = time.time()
            v = d[col].values
            narrow = [(d[c].values[:n1], d[c].values[n1:]) for c in narrow_cols]
            df = index_rule(v[:n1], v[n1:], cc, n1, max_s1=max_s1, max_tg=max_tg, min_len=min_len,
                            narrow=narrow)
            pass_s1.append(df.s1.values); pass_tg.append(df.tg.values); pass_blk.append(df.blk.values)
            pass_bit.append(np.full(len(df), 1 << bit, dtype=np.int32))
            pass_prio.append(np.full(len(df), priority, dtype=np.int8))
            print(f"  [{r_name:<18}] {len(df):>11,d} pairs ({time.time() - t_r:.1f}s)")
            del df
        names = d["name_core"].values
        addrs = d["addr_norm"].values
        del d
        gc.collect()

        # 4. Merge passes: one row per (s1, tg)
        print("Merging passes...")
        all_s1 = np.concatenate(pass_s1); all_tg = np.concatenate(pass_tg)
        all_bit = np.concatenate(pass_bit); all_prio = np.concatenate(pass_prio)
        all_blk = np.concatenate(pass_blk)
        del pass_s1, pass_tg, pass_bit, pass_prio, pass_blk
        pair_keys = (all_s1.astype(np.uint64) << np.uint64(32)) | all_tg.astype(np.uint64)
        del all_s1, all_tg
        order = np.argsort(pair_keys, kind="stable")
        sk = pair_keys[order]
        del pair_keys
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]]) if len(sk) else np.array([], np.int64)
        uk = sk[starts]
        del sk
        rules_mask = np.bitwise_or.reduceat(all_bit[order], starts) if len(starts) else np.array([], np.int32)
        best_prio = np.minimum.reduceat(all_prio[order], starts) if len(starts) else np.array([], np.int8)
        min_blk = np.minimum.reduceat(all_blk[order], starts) if len(starts) else np.array([], np.int32)
        del all_bit, all_prio, all_blk, order
        gc.collect()
        s1_pos = (uk >> np.uint64(32)).astype(np.int32)
        tg_pos = (uk & np.uint64(0xFFFFFFFF)).astype(np.int32)
        del uk
        g = pd.DataFrame({
            "s1_int": s1_ids[s1_pos], "tg_int": tg_ids[tg_pos],
            "n_rules": sum(((rules_mask >> b) & 1) for b in range(len(RULES))).astype(np.int16),
            "best_priority": best_prio.astype(np.int8), "rules_mask": rules_mask.astype(np.int32),
            "min_blk": min_blk.astype(np.int32),
        })
        print(f"  {len(g):,} unique pairs, {len(g) / max(n1, 1):.1f} per S1 before the cap")

        # 5. Quick similarity for S1s that have more candidates than sim_min (ranking only matters there)
        t_s = time.time()
        per_s1 = np.bincount(s1_pos, minlength=n1)
        need = per_s1[s1_pos] > self.sim_min
        g["name_sim"] = np.zeros(len(g), np.uint8)
        g["addr_sim"] = np.zeros(len(g), np.uint8)
        if need.any():
            # chunked: cpdist converts every string of a call up front, so one call over tens of
            # millions of pairs runs out of memory
            idx = np.flatnonzero(need)
            name_sim = np.zeros(len(g), np.uint8)
            addr_sim = np.zeros(len(g), np.uint8)
            for lo in range(0, len(idx), SIM_CHUNK):
                sl = idx[lo:lo + SIM_CHUNK]
                qi, ci = s1_pos[sl], n1 + tg_pos[sl]
                name_sim[sl] = cpdist(names[qi], names[ci], scorer=fuzz.token_set_ratio,
                                      workers=-1, dtype=np.uint8)
                addr_sim[sl] = cpdist(addrs[qi], addrs[ci], scorer=fuzz.token_set_ratio,
                                      workers=-1, dtype=np.uint8)
            g["name_sim"], g["addr_sim"] = name_sim, addr_sim
            del idx, name_sim, addr_sim
        print(f"  similarity for {int(need.sum()):,} pairs of S1s with > {self.sim_min} candidates "
              f"({time.time() - t_s:.0f}s)")
        del names, addrs, s1_pos, tg_pos, per_s1, need
        gc.collect()

        # 6. Rank within S1 and keep the diagnostics for the dev S1s (train only)
        g["score"] = rank_score(g)
        g.sort_values(["s1_int", "score"], ascending=[True, False], inplace=True, kind="stable")
        g["rank"] = g.groupby("s1_int", sort=False).cumcount().astype(np.int16)
        if split == "train" and DEV_IDS_FILE.exists():
            dev = set(id_to_int(pd.read_csv(DEV_IDS_FILE, dtype=str)["s1_id"].values))
            save_parquet_atomic(g[g.s1_int.isin(dev)].reset_index(drop=True), precap_dev_path(split))
            print(f"  saved uncapped dev pairs to {precap_dev_path(split)}")

        # 7. Cap and save
        capped = g[g["rank"] < self.max_candidates][
            ["s1_int", "tg_int", "n_rules", "best_priority", "rank"]].reset_index(drop=True)
        del g
        gc.collect()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"candidate_pairs_{split}.parquet"
        save_parquet_atomic(capped, cache_path)
        print(f"Saved {len(capped):,d} candidate pairs ({len(capped) / max(n1, 1):.1f} per S1) to {cache_path}")
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
        from collections import defaultdict
        grouped = defaultdict(list)
        for s1_i, tg_i in zip(candidates_df["s1_int"].values, candidates_df["tg_int"].values):
            grouped[s1_i].append(tg_i)
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
    parser.add_argument("--max-candidates", type=int, default=20, help="Maximum candidates per entity.")
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
