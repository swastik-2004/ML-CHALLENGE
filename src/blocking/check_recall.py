"""
Check candidate recall and candidate pool size distribution on the 100k dev set.
Usage:
    python -m src.blocking.check_recall [--rebuild] [--split train] [--max-candidates 50]
"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

from ..config import WORK_DIR, DEV_IDS_FILE, OUTPUT_DIR
from ..data import id_to_int
from .blocker import MultiPassBlocker


def check_recall(rebuild: bool = False, max_candidates: int = 50):
    print(f"{'='*75}")
    print("CANDIDATE BLOCKER RECALL & POOL SIZE CHECK (100k Dev Set)")
    print(f"{'='*75}")

    t_start = time.time()
    cache_path = WORK_DIR / "cache" / "candidate_pairs_train.parquet"

    # Rebuild if requested or missing
    if rebuild or not cache_path.exists():
        print(f"Building candidate cache with max_candidates={max_candidates}...")
        blocker = MultiPassBlocker(max_candidates_per_entity=max_candidates)
        blocker.build_candidates(split="train")

    # 1. Load Dev IDs & Ground Truth
    print("Loading dev IDs and ground truth...")
    dev_s1_ids = pd.read_csv(DEV_IDS_FILE)["s1_id"].values
    dev_s1_int_set = set(id_to_int(dev_s1_ids))

    truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
    dev_truth = truth[truth["s1_int"].isin(dev_s1_int_set)].copy()
    n_dev_truth = len(dev_truth)
    print(f"Total Dev S1 Entities  : {len(dev_s1_ids):,}")
    print(f"Total Dev True Pairs   : {n_dev_truth:,}")

    # 2. Load Candidate Pairs
    print(f"Loading candidate pairs from cache: {cache_path}")
    cand_df = pd.read_parquet(cache_path, columns=["s1_int", "tg_int", "n_rules", "best_priority"])
    dev_cands = cand_df[cand_df["s1_int"].isin(dev_s1_int_set)].copy()
    n_cand_pairs = len(dev_cands)
    print(f"Total Dev Candidate Pairs: {n_cand_pairs:,}")

    # 3. Calculate Recall
    matched = dev_truth.merge(dev_cands[["s1_int", "tg_int"]].drop_duplicates(), on=["s1_int", "tg_int"])
    n_matched = len(matched)
    recall = n_matched / n_dev_truth * 100.0

    # 4. Candidate Pool Sizing Stats
    counts = dev_cands.groupby("s1_int")["tg_int"].count()
    entities_with_cands = len(counts)
    coverage = entities_with_cands / len(dev_s1_ids) * 100.0
    mean_cand = n_cand_pairs / len(dev_s1_ids)

    median_cand = int(counts.median()) if len(counts) > 0 else 0
    p75_cand = int(counts.quantile(0.75)) if len(counts) > 0 else 0
    p90_cand = int(counts.quantile(0.90)) if len(counts) > 0 else 0
    p95_cand = int(counts.quantile(0.95)) if len(counts) > 0 else 0
    p99_cand = int(counts.quantile(0.99)) if len(counts) > 0 else 0
    max_cand = int(counts.max()) if len(counts) > 0 else 0

    print(f"\n{'-'*75}")
    print(f"RESULTS SUMMARY:")
    print(f"{'-'*75}")
    print(f"  Candidate Recall Ceiling     : {recall:.2f}% ({n_matched:,} / {n_dev_truth:,} true pairs)")
    print(f"  Missed True Pairs            : {n_dev_truth - n_matched:,} ({100.0 - recall:.2f}%)")
    print(f"  Entities with >= 1 Candidate : {entities_with_cands:,} / {len(dev_s1_ids):,} ({coverage:.2f}%)")
    print(f"  Empty Singletons Retained    : {len(dev_s1_ids) - entities_with_cands:,}")
    print(f"  Mean Candidates per S1 Entity: {mean_cand:.2f}")
    print(f"  Median Candidates per S1     : {median_cand}")
    print(f"  P75 Candidates per S1        : {p75_cand}")
    print(f"  P90 Candidates per S1        : {p90_cand}")
    print(f"  P95 Candidates per S1        : {p95_cand}")
    print(f"  P99 Candidates per S1        : {p99_cand}")
    print(f"  Max Candidates (Hard Cap)    : {max_cand}")
    print(f"  Evaluation Time              : {time.time() - t_start:.2f}s")
    print(f"{'-'*75}")

    results = {
        "dev_entities": len(dev_s1_ids),
        "dev_truth_pairs": n_dev_truth,
        "matched_truth_pairs": n_matched,
        "candidate_recall_pct": round(recall, 2),
        "mean_candidates_per_entity": round(mean_cand, 2),
        "median_candidates": median_cand,
        "p75_candidates": p75_cand,
        "p90_candidates": p90_cand,
        "p95_candidates": p95_cand,
        "p99_candidates": p99_cand,
        "max_candidates": max_cand,
        "coverage_pct": round(coverage, 2)
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "blocking_recall_check.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check candidate recall and pool sizing.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild training candidate cache before checking.")
    parser.add_argument("--max-candidates", type=int, default=50, help="Maximum candidates per S1 entity.")
    args = parser.parse_args()
    check_recall(rebuild=args.rebuild, max_candidates=args.max_candidates)
