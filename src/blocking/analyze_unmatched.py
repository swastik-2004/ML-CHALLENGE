"""
Analyze unmatched ground-truth pairs to discover why baseline rules missed them
and guide candidate blocking design.
"""
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from ..config import WORK_DIR, DEV_IDS_FILE
from ..data import id_to_int
from ..normalize import norm_path


def analyze_unmatched_pairs(sample_size: int = 25):
    print("Loading dev IDs and ground truth...")
    dev_s1_ids = pd.read_csv(DEV_IDS_FILE)["s1_id"].values
    dev_s1_int_set = set(id_to_int(dev_s1_ids))

    truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
    dev_truth = truth[truth["s1_int"].isin(dev_s1_int_set)].copy()
    print(f"Total true matching pairs in dev set: {len(dev_truth)}")

    pairs_cache = WORK_DIR / "cache" / "pairs_train.parquet"
    if not pairs_cache.exists():
        print(f"Error: {pairs_cache} not found. Run baseline first.")
        return

    pairs = pd.read_parquet(pairs_cache)
    dev_pairs = pairs[pairs["s1_int"].isin(dev_s1_int_set)]

    matched = dev_truth.merge(dev_pairs[["s1_int", "tg_int"]], on=["s1_int", "tg_int"])
    print(f"Baseline matches in dev: {len(matched)} / {len(dev_truth)} ({len(matched)/len(dev_truth)*100:.2f}%)")

    unmatched = dev_truth.merge(dev_pairs[["s1_int", "tg_int"]], on=["s1_int", "tg_int"], how="left", indicator=True)
    unmatched = unmatched[unmatched["_merge"] == "left_only"][["s1_int", "tg_int"]]
    print(f"Unmatched true pairs in dev: {len(unmatched)}")

    cols = ["entity_id", "country", "name_norm", "name_core", "name_key", "name_compact", "addr_norm", "addr_key", "house_no", "state"]
    print("Loading normalized S1 and Target data...")
    s1 = pd.read_parquet(norm_path("train", 1), columns=cols)
    s1["s1_int"] = id_to_int(s1.entity_id.values)
    s1_dev = s1[s1["s1_int"].isin(dev_s1_int_set)].set_index("s1_int")

    sample_unmatched = unmatched.head(sample_size * 2)
    tg_needed = set(sample_unmatched["tg_int"])

    tg2 = pd.read_parquet(norm_path("train", 2), columns=cols)
    tg2["tg_int"] = id_to_int(tg2.entity_id.values)
    tg3 = pd.read_parquet(norm_path("train", 3), columns=cols)
    tg3["tg_int"] = id_to_int(tg3.entity_id.values)
    tg = pd.concat([tg2, tg3], ignore_index=True)
    tg = tg[tg["tg_int"].isin(tg_needed)].set_index("tg_int")

    print(f"\n{'='*75}")
    print(f"SAMPLE {sample_size} UNMATCHED GROUND TRUTH PAIRS (MISSED BY BASELINE)")
    print(f"{'='*75}")

    count = 0
    for _, row in sample_unmatched.iterrows():
        s1_id = row["s1_int"]
        tg_id = row["tg_int"]
        if s1_id in s1_dev.index and tg_id in tg.index:
            s1_row = s1_dev.loc[s1_id]
            tg_row = tg.loc[tg_id]
            count += 1
            print(f"Pair #{count} [{s1_row['country']}]")
            print(f"  S1: '{s1_row['name_norm']}' (key: '{s1_row['name_key']}')")
            print(f"  TG: '{tg_row['name_norm']}' (key: '{tg_row['name_key']}')")
            print(f"  S1 Addr: '{s1_row['addr_norm']}' | House: '{s1_row['house_no']}' | State: '{s1_row['state']}'")
            print(f"  TG Addr: '{tg_row['addr_norm']}' | House: '{tg_row['house_no']}' | State: '{tg_row['state']}'")
            print("-" * 75)
            if count >= sample_size:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze unmatched ground truth pairs.")
    parser.add_argument("--sample", type=int, default=20, help="Number of sample pairs to print.")
    args = parser.parse_args()
    analyze_unmatched_pairs(sample_size=args.sample)
