"""
Benchmark candidate blocking strategies on the 100k Dev set.
Evaluates recall ceiling, candidate pool sizing, and crowding limits.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..config import WORK_DIR, DEV_IDS_FILE, OUTPUT_DIR
from ..data import id_to_int
from ..normalize import norm_path

STOP_WORDS = {
    "the", "and", "inc", "ltd", "corp", "llc", "pvt", "for", "new", "co",
    "company", "group", "services", "private", "limited", "enterprises",
    "solutions", "technologies", "holdings", "industries", "association"
}


def extract_blocking_tokens(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of first token, second token, and 3-char prefix."""
    w1_arr = np.empty(len(series), dtype=object)
    w2_arr = np.empty(len(series), dtype=object)
    p3_arr = np.empty(len(series), dtype=object)

    for i, name in enumerate(series.values):
        toks = [t for t in str(name).split() if len(t) >= 3 and t not in STOP_WORDS]
        w1 = toks[0] if len(toks) > 0 else (str(name).split()[0] if str(name).split() else "")
        w2 = toks[1] if len(toks) > 1 else ""
        p3 = w1[:3] if len(w1) >= 3 else w1
        w1_arr[i] = w1
        w2_arr[i] = w2
        p3_arr[i] = p3

    return w1_arr, w2_arr, p3_arr


def run_rule_indexing(
    s1_vals: np.ndarray,
    tg_vals: np.ndarray,
    both_country_codes: np.ndarray,
    n1: int,
    max_s1: int = 20,
    max_tg: int = 50,
    min_len: int = 0
) -> pd.DataFrame:
    """Factorized vectorized candidate pair indexing."""
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

    pairs = pd.DataFrame({"s1": ia, "k": ka[ia]}).merge(pd.DataFrame({"tg": ib, "k": kb[ib]}), on="k")
    return pairs[["s1", "tg"]].astype(np.int32)


def benchmark_blocking():
    print(f"{'='*75}")
    print("MULTI-PASS CANDIDATE BLOCKER BENCHMARK (100k Dev Set)")
    print(f"{'='*75}")

    t_start = time.time()
    # 1. Load dev IDs and Ground Truth
    dev_s1_ids = pd.read_csv(DEV_IDS_FILE)["s1_id"].values
    dev_s1_int_set = set(id_to_int(dev_s1_ids))

    truth_path = WORK_DIR / "cache" / "truth_train.parquet"
    truth = pd.read_parquet(truth_path)
    dev_truth = truth[truth["s1_int"].isin(dev_s1_int_set)].copy()
    n_dev_truth = len(dev_truth)
    print(f"Loaded {len(dev_s1_ids):,} Dev entities with {n_dev_truth:,} true matching pairs.")

    # 2. Load Normalized Data
    cols = ["entity_id", "country", "name_norm", "name_core", "name_key", "name_compact", "addr_norm", "addr_key", "house_no", "state"]
    print("Loading normalized datasets...")
    s1 = pd.read_parquet(norm_path("train", 1), columns=cols)
    tg2 = pd.read_parquet(norm_path("train", 2), columns=cols)
    tg3 = pd.read_parquet(norm_path("train", 3), columns=cols)
    tg = pd.concat([tg2, tg3], ignore_index=True)
    del tg2, tg3

    n1 = len(s1)
    s1_ids = id_to_int(s1.entity_id.values)
    tg_ids = id_to_int(tg.entity_id.values)

    dev_mask_s1 = np.isin(s1_ids, list(dev_s1_int_set))
    dev_indices_set = set(np.flatnonzero(dev_mask_s1))

    # 3. Derive Blocking Tokens
    print("Extracting derived blocking tokens...")
    t0 = time.time()
    s1["w1"], s1["w2"], s1["p3"] = extract_blocking_tokens(s1["name_core"])
    tg["w1"], tg["w2"], tg["p3"] = extract_blocking_tokens(tg["name_core"])

    s1["house_p3"] = np.where((s1["house_no"].values != "") & (s1["p3"].values != ""), s1["house_no"].values + "_" + s1["p3"].values, "")
    tg["house_p3"] = np.where((tg["house_no"].values != "") & (tg["p3"].values != ""), tg["house_no"].values + "_" + tg["p3"].values, "")

    s1["state_w1"] = np.where((s1["state"].values != "") & (s1["w1"].values != ""), s1["state"].values + "_" + s1["w1"].values, "")
    tg["state_w1"] = np.where((tg["state"].values != "") & (tg["w1"].values != ""), tg["state"].values + "_" + tg["w1"].values, "")
    print(f"Token extraction completed in {time.time() - t0:.2f}s")

    both_country = pd.concat([s1["country"], tg["country"]], ignore_index=True)
    cc, _ = pd.factorize(both_country)
    del both_country

    # 4. Multi-Pass Rules Configuration
    # (name, s1_col, tg_col, max_s1, max_tg, min_len, priority)
    rules = [
        ("R1_name_key", s1["name_key"].values, tg["name_key"].values, 20, 50, 0, 1),
        ("R2_name_compact", s1["name_compact"].values, tg["name_compact"].values, 20, 50, 0, 1),
        ("R3_addr_key", s1["addr_key"].values, tg["addr_key"].values, 20, 50, 0, 1),
        ("R4_w1_name", s1["w1"].values, tg["w1"].values, 15, 30, 3, 2),
        ("R5_house_p3", s1["house_p3"].values, tg["house_p3"].values, 20, 50, 4, 2),
        ("R6_state_w1", s1["state_w1"].values, tg["state_w1"].values, 15, 40, 4, 3),
        ("R7_w2_name", s1["w2"].values, tg["w2"].values, 10, 25, 3, 3),
    ]

    all_dev_pairs = []
    cumulative_pairs_set = set()
    rule_metrics = []

    print(f"\n{'Rule Name':<20} | {'Total Pairs':<12} | {'Dev Pairs':<10} | {'Rule Recall':<12} | {'Cumulative Recall'}")
    print("-" * 75)

    for rule_name, s1_c, tg_c, max_s1, max_tg, min_len, priority in rules:
        t_rule = time.time()
        p = run_rule_indexing(s1_c, tg_c, cc, n1, max_s1=max_s1, max_tg=max_tg, min_len=min_len)
        p["priority"] = np.int8(priority)

        dev_p = p[p["s1"].isin(dev_indices_set)].copy()
        dev_p["s1_int"] = s1_ids[dev_p["s1"].values]
        dev_p["tg_int"] = tg_ids[dev_p["tg"].values]
        dev_p["rule"] = rule_name

        rule_truth_m = dev_truth.merge(dev_p[["s1_int", "tg_int"]].drop_duplicates(), on=["s1_int", "tg_int"])
        rule_recall = len(rule_truth_m) / n_dev_truth

        new_pairs = set(zip(dev_p["s1_int"].values, dev_p["tg_int"].values))
        cumulative_pairs_set.update(new_pairs)

        cum_df = pd.DataFrame(list(cumulative_pairs_set), columns=["s1_int", "tg_int"])
        cum_m = dev_truth.merge(cum_df, on=["s1_int", "tg_int"])
        cum_recall = len(cum_m) / n_dev_truth

        print(f"{rule_name:<20} | {len(p):<12,d} | {len(dev_p):<10,d} | {rule_recall*100:>10.2f}% | {cum_recall*100:>15.2f}%")
        rule_metrics.append({
            "rule": rule_name,
            "total_pairs": len(p),
            "dev_pairs": len(dev_p),
            "rule_recall": round(rule_recall * 100, 2),
            "cumulative_recall": round(cum_recall * 100, 2),
            "elapsed_seconds": round(time.time() - t_rule, 2)
        })
        all_dev_pairs.append(dev_p)

    # 5. Candidate Distribution Analysis
    print(f"\n{'='*75}")
    print("CANDIDATE POOL SIZING & DISTRIBUTION ON 100K DEV")
    print(f"{'='*75}")
    dev_combined = pd.concat(all_dev_pairs, ignore_index=True)
    counts = dev_combined.groupby("s1_int")["tg_int"].nunique()

    stats = {
        "total_dev_entities": len(dev_s1_ids),
        "entities_with_candidates": int(len(counts)),
        "coverage_pct": round(len(counts) / len(dev_s1_ids) * 100, 2),
        "mean_candidates_per_entity": round(float(counts.mean()), 2),
        "median_candidates": int(counts.median()),
        "p75_candidates": int(counts.quantile(0.75)),
        "p90_candidates": int(counts.quantile(0.90)),
        "p95_candidates": int(counts.quantile(0.95)),
        "p99_candidates": int(counts.quantile(0.99)),
        "max_candidates": int(counts.max()),
        "total_benchmark_time_seconds": round(time.time() - t_start, 2)
    }

    for k, v in stats.items():
        print(f"  {k:<35}: {v}")

    # Save findings JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "blocking_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rule_metrics": rule_metrics, "pool_statistics": stats}, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")


if __name__ == "__main__":
    benchmark_blocking()
