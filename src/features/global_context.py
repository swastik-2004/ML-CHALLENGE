"""
Candidate-level context measured over the WHOLE blocker output of a split.

Why this exists: a candidate's competition (how many S1s share it, where this S1 ranks among
them) only means something on the full population. The first version computed it inside
pair_features on whatever pairs it was given, i.e. on the 100k dev S1s only: cand_n_s1 had
median 1 at train time but median 9 on the full split, so a model trained on dev features
would see out-of-range values on test. Both train and test now read the full cache
data/cache/candidate_pairs_{split}.parquet, so the feature means the same thing on both.

  cand_n_s1      number of S1s whose candidate list contains this S2/S3 record
  cand_rank_blk  rank of this S1 among them by blocker evidence (best_priority asc, n_rules desc);
                 1 = the S1 with the strongest blocking evidence for this record
(cand_rank_for_target, which needed features for every pair of the split, is retired.)

  python -m src.features.global_context --split train --feats data/feats/train_subset.parquet
      -> replaces the old context columns in an existing feature file (in place, atomic)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..config import WORK_DIR
from ..data import id_to_int, save_parquet_atomic

CTX_COLS = ["cand_n_s1", "cand_rank_blk"]
RETIRED_COLS = ["cand_rank_for_target"]


def cache_path(split: str):
    return WORK_DIR / "cache" / f"candidate_pairs_{split}.parquet"


def candidate_context(tg: np.ndarray, n_rules: np.ndarray, best_priority: np.ndarray):
    """(cand_n_s1, cand_rank_blk) as float32 arrays aligned with the input rows.
    Pure numpy sort, memory ~40 bytes/row (57M train rows ~ 2.3 GB peak)."""
    n = len(tg)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    # higher score = stronger evidence; n_rules <= 13 so it never outweighs one priority step
    score = n_rules.astype(np.int32) - best_priority.astype(np.int32) * 100
    order = np.lexsort((-score, tg))           # group by tg, strongest S1 first
    del score
    tg_s = tg[order]
    new_grp = np.empty(n, bool)
    new_grp[0] = True
    np.not_equal(tg_s[1:], tg_s[:-1], out=new_grp[1:])
    del tg_s
    sc_s = (n_rules.astype(np.int32) - best_priority.astype(np.int32) * 100)[order]
    new_val = new_grp.copy()
    new_val[1:] |= sc_s[1:] != sc_s[:-1]
    del sc_s
    idx = np.arange(n, dtype=np.int64)
    grp_start = np.maximum.accumulate(np.where(new_grp, idx, 0))
    val_start = np.maximum.accumulate(np.where(new_val, idx, 0))
    del new_val, idx
    rank = np.empty(n, np.float32)
    rank[order] = (val_start - grp_start + 1).astype(np.float32)   # 'min' rank among ties
    del val_start, grp_start
    gid = np.cumsum(new_grp) - 1
    del new_grp
    size = np.bincount(gid).astype(np.float32)
    n_s1 = np.empty(n, np.float32)
    n_s1[order] = size[gid]
    return n_s1, rank


def load_cache_with_context(split: str, tg_subset=None) -> pd.DataFrame:
    """Blocker cache (s1_int, tg_int, n_rules, best_priority) + CTX_COLS, sorted by s1_int.
    tg_subset: only load rows for these records. Exact, because a record's context depends only
    on its own rows; used to patch a dev feature file with less memory."""
    t0 = time.time()
    cols = ["s1_int", "tg_int", "n_rules", "best_priority"]
    if tg_subset is None:
        c = pd.read_parquet(cache_path(split), columns=cols)
    else:
        import pyarrow as pa
        import pyarrow.dataset as pads
        flt = pads.field("tg_int").isin(pa.array(np.unique(tg_subset), type=pa.int64()))
        c = pads.dataset(cache_path(split), format="parquet").to_table(columns=cols, filter=flt).to_pandas()
    c["n_rules"] = c.n_rules.astype(np.int8)
    c["best_priority"] = c.best_priority.astype(np.int8)
    n_s1, rank = candidate_context(c.tg_int.values, c.n_rules.values, c.best_priority.values)
    c["cand_n_s1"] = n_s1
    c["cand_rank_blk"] = rank
    if not c.s1_int.is_monotonic_increasing:
        c = c.sort_values("s1_int", kind="stable").reset_index(drop=True)
    print(f"  {split} cache: {len(c):,} pairs, {c.s1_int.nunique():,} S1 with candidates, "
          f"cand_n_s1 median {np.median(n_s1):.0f} ({time.time()-t0:.0f}s)", flush=True)
    return c


def patch_feature_file(split: str, feats_path) -> None:
    """Swap the old subset-computed context columns for the full-split ones."""
    feats = pd.read_parquet(feats_path)
    f_s1, f_tg = id_to_int(feats.s1_id.values), id_to_int(feats.cand_id.values)
    ctx = load_cache_with_context(split, tg_subset=f_tg)
    ctx = ctx[ctx.s1_int.isin(np.unique(f_s1))][["s1_int", "tg_int"] + CTX_COLS]
    m = pd.DataFrame({"s1_int": f_s1, "tg_int": f_tg}).merge(ctx, on=["s1_int", "tg_int"], how="left")
    if len(m) != len(feats) or m[CTX_COLS].isna().any().any():
        raise RuntimeError("feature rows not found in the blocker cache: features and cache are out of sync "
                           "(rebuild features from the current cache)")
    feats = feats.drop(columns=[c for c in CTX_COLS + RETIRED_COLS if c in feats.columns])
    for c in CTX_COLS:
        feats[c] = m[c].values.astype(np.float32)
    save_parquet_atomic(feats, feats_path)
    print(f"patched {feats_path}: {len(feats):,} rows; cand_n_s1 median now "
          f"{feats.cand_n_s1.median():.0f}, cand_rank_blk median {feats.cand_rank_blk.median():.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--feats", required=True, help="feature parquet to patch in place")
    a = ap.parse_args()
    patch_feature_file(a.split, a.feats)
