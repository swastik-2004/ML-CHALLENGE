"""
Build the shared validation split files (run once; everyone gets identical files).

  data/folds.csv       s1_id, fold (0..4)   every train S1, grouped by S1 so all of an
                                            entity's candidates stay in one fold
  data/dev_s1_ids.csv  s1_id                the shared 100k dev sample:
                                            read_ground_truth().sample(n=100_000, random_state=42)

Usage: python -m src.folds
"""
import numpy as np
import pandas as pd

from .config import WORK_DIR, FOLDS_FILE, DEV_IDS_FILE, RANDOM_SEED, N_FOLDS, DEV_SAMPLE_SIZE
from .data import read_ground_truth


def build():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    gt = read_ground_truth()
    rng = np.random.default_rng(RANDOM_SEED)
    folds = pd.DataFrame({"s1_id": gt["source1_entity_id"].values,
                          "fold": rng.permutation(len(gt)) % N_FOLDS})
    folds.to_csv(FOLDS_FILE, index=False)
    dev = gt.sample(n=DEV_SAMPLE_SIZE, random_state=RANDOM_SEED)["source1_entity_id"]
    dev.rename("s1_id").to_frame().to_csv(DEV_IDS_FILE, index=False)
    return folds, dev


if __name__ == "__main__":
    folds, dev = build()
    print(f"wrote {FOLDS_FILE} ({len(folds):,} rows) and {DEV_IDS_FILE} ({len(dev):,} ids)")
    print("fold sizes:", folds.fold.value_counts().sort_index().to_dict())
