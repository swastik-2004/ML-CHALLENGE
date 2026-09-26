"""
S1 ids to build training features for: the 100k dev S1s plus extra train S1s.
More training S1s help: 10k -> 20k added +0.23 pt macro F0.5 on dev (26 Sep), about the same
per doubling. Default 300k in total.

  python -m src.train_ids                 # -> data/train_s1_ids.csv (300k)
  python -m src.train_ids --n 200000
"""
import argparse

import pandas as pd

from .config import DEV_IDS_FILE, RANDOM_SEED, WORK_DIR
from .data import read_source

OUT = WORK_DIR / "train_s1_ids.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    a = ap.parse_args()
    dev = pd.read_csv(DEV_IDS_FILE, dtype=str).s1_id
    s1 = read_source("train", 1, usecols=["entity_id"]).entity_id
    extra = s1[~s1.isin(set(dev))].sample(n=max(a.n - len(dev), 0), random_state=RANDOM_SEED)
    ids = pd.concat([dev, extra], ignore_index=True)
    pd.DataFrame({"s1_id": ids}).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(ids):,} S1 ({len(dev):,} dev + {len(extra):,} extra)")
