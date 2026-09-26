"""
Validation F0.5 by country for a trained classifier, on EXACTLY the validation S1s that
classifier.train held out (same GroupShuffleSplit, same seed). Also saves every validation
prediction so errors can be analysed afterwards.

  python -m src.models.country_breakdown --features-path data/feats/train_300k.parquet ^
      --eval-s1-ids data/train_s1_ids.csv --model output/classifier_model_300k.joblib ^
      --threshold output/optimal_threshold_300k.json --tag 300k_oldblocker

Writes data/preds/val_preds_{tag}.parquet (s1_id, cand_id, p, y, selected, country)
and reports/country_breakdown_{tag}.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from classifier.config import RANDOM_SEED, VALIDATION_SPLIT
from classifier.model import ClassifierModel
from src.config import PROJECT_ROOT, WORK_DIR
from src.data import id_to_int, save_parquet_atomic
from src.models.decision import macro_f05, one_owner, select_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-path", required=True)
    ap.add_argument("--eval-s1-ids", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", required=True, help="optimal_threshold*.json written by classifier.train")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()

    # the same validation S1s as classifier.train (sorted universe, GroupShuffleSplit, seed)
    universe = pd.Series(sorted(set(pd.read_csv(a.eval_s1_ids, dtype=str)["s1_id"])))
    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    _, val_pos = next(gss.split(universe, groups=universe))
    val_ids = universe.iloc[val_pos].values
    val_int = id_to_int(val_ids)

    import pyarrow as pa, pyarrow.dataset as pads
    f = pads.dataset(a.features_path).to_table(filter=pads.field("s1_id").isin(pa.array(val_ids))).to_pandas()
    model = ClassifierModel.load(a.model)
    thr = float(json.loads(Path(a.threshold).read_text())["best_threshold"])
    p = model.predict_proba(f[model.feature_names_])
    print(f"{len(f):,} validation pairs for {len(val_ids):,} S1, threshold {thr} ({time.time()-t0:.0f}s)", flush=True)

    truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
    truth = truth[truth.s1_int.isin(set(val_int))]
    n_true = truth.groupby("s1_int").size().reindex(val_int, fill_value=0).values
    d = pd.DataFrame({"s1_int": id_to_int(f.s1_id.values), "tg_int": id_to_int(f.cand_id.values), "p": p})
    d = d.merge(truth.assign(y=1), on=["s1_int", "tg_int"], how="left").fillna({"y": 0})
    sel = select_threshold(one_owner(d), thr)
    f05, _ = macro_f05(sel, val_int, n_true)

    s1c = pads.dataset(str(WORK_DIR / "norm" / "train_s1")).to_table(
        columns=["entity_id", "country"], filter=pads.field("entity_id").isin(pa.array(val_ids))).to_pandas()
    ctry = pd.Series(s1c.country.values, index=id_to_int(s1c.entity_id.values)).reindex(val_int).values
    in_cands = d.groupby("s1_int").y.sum().reindex(val_int, fill_value=0).values
    rc = np.where(n_true > 0, in_cands / np.maximum(n_true, 1), 0)
    ceiling = np.where(n_true == 0, 1.0, np.where(in_cands > 0, 1.25 * rc / (0.25 + rc), 0.0))

    rows = {"ALL": (f05.mean(), ceiling.mean(), len(val_int))}
    for c in sorted(set(ctry)):
        k = ctry == c
        rows[c] = (f05[k].mean(), ceiling[k].mean(), int(k.sum()))
    print(f"\n{'country':8s} {'F0.5':>7s} {'ceiling':>8s} {'gap':>6s} {'S1':>8s}")
    for c, (v, cl, n) in rows.items():
        print(f"{c:8s} {v:7.4f} {cl:8.4f} {cl - v:6.4f} {n:8,d}")

    sel_key = set(zip(sel.s1_int.values, sel.tg_int.values))
    out = pd.DataFrame({"s1_id": f.s1_id.values, "cand_id": f.cand_id.values, "p": p.astype(np.float32),
                        "y": d.y.values.astype(np.int8)})
    out["selected"] = [(a_, b_) in sel_key for a_, b_ in zip(d.s1_int.values, d.tg_int.values)]
    out["country"] = pd.Series(ctry, index=val_int).reindex(d.s1_int.values).values
    pred_path = WORK_DIR / "preds" / f"val_preds_{a.tag}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet_atomic(out, pred_path)
    rep = PROJECT_ROOT / "reports" / f"country_breakdown_{a.tag}.json"
    rep.write_text(json.dumps({c: {"f05": round(float(v), 4), "ceiling": round(float(cl), 4), "n_s1": n}
                               for c, (v, cl, n) in rows.items()} | {"threshold": thr}, indent=2))
    print(f"\nsaved {pred_path} and {rep} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
