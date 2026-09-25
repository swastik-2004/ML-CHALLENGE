"""
Recall check on the blocker's REAL output file (not a re-implementation).

  python -m src.blocking.check_recall            # train cache vs ground truth, dev 100k + all train
Needs data/cache/candidate_pairs_train.parquet (blocker) and data/cache/truth_train.parquet
(python -m src.baseline --stage truth).
"""
import pandas as pd

from ..config import DEV_IDS_FILE, WORK_DIR
from ..data import id_to_int

CACHE = WORK_DIR / "cache"

if __name__ == "__main__":
    cand = pd.read_parquet(CACHE / "candidate_pairs_train.parquet", columns=["s1_int", "tg_int"])
    truth = pd.read_parquet(CACHE / "truth_train.parquet")
    hit = truth.merge(cand, on=["s1_int", "tg_int"], how="left", indicator=True)["_merge"].eq("both")
    dev = set(id_to_int(pd.read_csv(DEV_IDS_FILE, dtype=str).s1_id.values))
    per_s1 = cand.groupby("s1_int").size()
    n_s1 = pd.read_parquet(CACHE / "ids_train_s1.parquet").shape[0] if (CACHE / "ids_train_s1.parquet").exists() else None
    for name, m in [("DEV 100k", truth.s1_int.isin(dev).values), ("ALL TRAIN", slice(None))]:
        print(f"{name:<9} pair recall {hit[m].mean():.2%}  ({int(hit[m].sum()):,} of {int(hit[m].size):,} true pairs)")
    print(f"candidates per S1: mean {per_s1.mean():.1f}, median {per_s1.median():.0f}, "
          f"p95 {per_s1.quantile(.95):.0f}, max {per_s1.max()}, total pairs {len(cand):,}")
    if n_s1:
        print(f"S1 with zero candidates: {1 - len(per_s1) / n_s1:.2%}")
