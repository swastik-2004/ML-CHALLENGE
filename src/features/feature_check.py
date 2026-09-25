"""
Feature sanity check on REAL train data, before any blocker exists.

Positives : every true (S1, S2/S3) pair for N dev S1s.
Negatives : same-country look-alikes that are NOT matches, sampled per S1 from two mini-blocks:
            same first name-key token, and same (house number, state). Up to 4 each.
Reports per-feature ROC-AUC (on rows where the feature is defined) and a quick grouped-CV
model AUC for name-only vs name+address features. Context features are excluded here: this
harness always contains the positives, so their distribution would be unrealistic.

  python -m src.features.feature_check --stage scan --n-s1 5000
  python -m src.features.feature_check --stage pairs
  python -m src.features.feature_check --stage report
"""
import argparse
import time

import numpy as np
import pandas as pd

from ..config import DEV_IDS_FILE, WORK_DIR, RANDOM_SEED
from ..data import read_ground_truth, gt_pairs, save_parquet_atomic
from ..normalize import norm_path
from .pair_features import REC_COLS, compute_features

CACHE = WORK_DIR / "cache"
PER_KEY_CAP = 8


def _keys(df):
    first = df.name_key.str.split(" ", n=1).str[0]
    nkey = df.country + "|" + first
    akey = df.country + "|" + df.house_no + "|" + df.state
    akey = akey.where((df.house_no != "") & (df.state != ""), "")
    return nkey.where(first != "", ""), akey


def stage_scan(n_s1: int) -> None:
    """Pick N dev S1s and keep their true matches + capped look-alike targets (one pass)."""
    t0 = time.time()
    dev = pd.read_csv(DEV_IDS_FILE, dtype=str).s1_id.sample(n=n_s1, random_state=RANDOM_SEED)
    gt = read_ground_truth()
    pos = gt_pairs(gt[gt.source1_entity_id.isin(set(dev))])
    pos_ids = set(pos.match_id)
    s1 = pd.concat([pd.read_parquet(f, columns=REC_COLS)
                    for f in sorted(norm_path("train", 1).glob("part-*.parquet"))])
    s1 = s1[s1.entity_id.isin(set(dev))].copy()
    s1["nkey"], s1["akey"] = _keys(s1)
    nset, aset = set(s1.nkey) - {""}, set(s1.akey) - {""}
    kept = []
    for s in (2, 3):
        for f in sorted(norm_path("train", s).glob("part-*.parquet")):
            t = pd.read_parquet(f, columns=REC_COLS)
            t["nkey"], t["akey"] = _keys(t)
            kept += [t[t.nkey.isin(nset)].groupby("nkey").head(PER_KEY_CAP),
                     t[t.akey.isin(aset)].groupby("akey").head(PER_KEY_CAP),
                     t[t.entity_id.isin(pos_ids)]]
    tg = pd.concat(kept).drop_duplicates("entity_id")
    save_parquet_atomic(s1, CACHE / "fc_s1.parquet")
    save_parquet_atomic(tg, CACHE / "fc_tg.parquet")
    save_parquet_atomic(pos, CACHE / "fc_pos.parquet")
    print(f"scan: {len(s1):,} S1, {len(pos):,} true pairs, {len(tg):,} target rows kept  {time.time()-t0:.0f}s")


def stage_pairs() -> None:
    """Vectorised negative sampling: join S1 keys to target keys, drop true pairs, keep <=4/key/S1."""
    t0 = time.time()
    s1 = pd.read_parquet(CACHE / "fc_s1.parquet")
    tg = pd.read_parquet(CACHE / "fc_tg.parquet")
    pos = pd.read_parquet(CACHE / "fc_pos.parquet")
    negs = []
    for k in ("nkey", "akey"):
        a = s1.loc[s1[k] != "", ["entity_id", k]].rename(columns={"entity_id": "s1_id"})
        b = tg.loc[tg[k] != "", ["entity_id", k]].rename(columns={"entity_id": "cand_id"})
        m = a.merge(b, on=k)[["s1_id", "cand_id"]]
        m = m.sample(frac=1.0, random_state=RANDOM_SEED)
        negs.append(m)
    neg = pd.concat(negs).drop_duplicates()
    truth = pos.rename(columns={"match_id": "cand_id"}).assign(label=1)
    neg = neg.merge(truth, on=["s1_id", "cand_id"], how="left")
    neg = neg[neg.label.isna()].groupby("s1_id").head(8).assign(label=0)
    pairs = pd.concat([truth, neg], ignore_index=True)
    pairs["label"] = pairs.label.astype(int)
    rec = pd.concat([s1[REC_COLS], tg[REC_COLS]]).drop_duplicates("entity_id")
    save_parquet_atomic(pairs, CACHE / "fc_pairs.parquet")
    save_parquet_atomic(rec, CACHE / "fc_records.parquet")
    print(f"pairs: {len(pairs):,} ({pairs.label.sum():,} true, {(pairs.label == 0).sum():,} look-alikes, "
          f"{pairs.s1_id.nunique():,} S1)  {time.time()-t0:.0f}s")


def stage_report() -> None:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    pairs = pd.read_parquet(CACHE / "fc_pairs.parquet")
    rec = pd.read_parquet(CACHE / "fc_records.parquet").set_index("entity_id")
    t0 = time.time()
    X = compute_features(pairs, rec)
    print(f"{len(X):,} pairs x {X.shape[1]} features in {time.time()-t0:.1f}s "
          f"({len(X)/(time.time()-t0):,.0f} pairs/s)")
    y = pairs.label.values
    rows = []
    for c in X.columns:
        m = X[c].notna().values
        if m.sum() > 100 and len(set(y[m])) == 2 and X[c][m].nunique() > 1:
            auc = roc_auc_score(y[m], X[c].values[m])
            rows.append((c, m.mean(), auc, max(auc, 1 - auc)))
    r = pd.DataFrame(rows, columns=["feature", "coverage", "auc", "strength"]).sort_values("strength", ascending=False)
    print("\nper-feature ROC-AUC (true pair vs look-alike; 0.5 = useless, <0.5 = inverse signal):")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    name_cols = [c for c in X.columns if c.startswith(("name_", "legal", "cand_is", "script", "cand_non"))]
    groups = pairs.s1_id.values
    for label, cols in [("name only", name_cols), ("name + address + all", list(X.columns))]:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(3).split(X, y, groups):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=RANDOM_SEED)
            m.fit(X.iloc[tr][cols], y[tr])
            oof[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
        print(f"\nquick model [{label}]: grouped-CV ROC-AUC {roc_auc_score(y, oof):.4f} | "
              f"avg precision {average_precision_score(y, oof):.4f} | "
              f"precision@p>0.5 {y[oof > 0.5].mean():.4f} recall@p>0.5 {(oof[y == 1] > 0.5).mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "pairs", "report"], required=True)
    ap.add_argument("--n-s1", type=int, default=5000)
    a = ap.parse_args()
    {"scan": lambda: stage_scan(a.n_s1), "pairs": stage_pairs, "report": stage_report}[a.stage]()
