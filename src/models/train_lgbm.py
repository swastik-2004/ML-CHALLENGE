"""
LightGBM matcher + per-S1 decision layer, scored with macro F0.5 on the dev S1 set.

1. Label each candidate pair from the ground truth.
2. Out-of-fold (OOF) probabilities: train on 4 fold-groups, predict the 5th (folds.csv, by S1),
   so every S1 is scored by a model that never saw it.
3. One-owner rule: each S2/S3 record stays only under the S1 that gives it the highest probability.
4. Choose each S1's output set, two ways, tuned on OOF and compared:
   - threshold: keep candidates with p >= t
   - expected F0.5: per S1, add candidates in probability order while expected F0.5 rises;
     predict nothing when P(no match) beats every non-empty set (singletons).
5. Score macro F0.5 over ALL dev S1s (S1s with no candidates count as empty predictions).

  python -m src.models.train_lgbm                                   # dev 100k features
  python -m src.models.train_lgbm --sample-s1 5000 --rounds 150 --folds 2   # quick smoke test
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import DEV_IDS_FILE, FOLDS_FILE, RANDOM_SEED, WORK_DIR
from ..data import id_to_int, save_parquet_atomic
from ..evaluate import f05_from_counts
from ..normalize import norm_path

ID_COLS = ["s1_id", "cand_id"]
MODEL_DIR = WORK_DIR / "models"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=RANDOM_SEED, num_threads=0)


def load(feats_path, sample_s1):
    dev = pd.read_csv(DEV_IDS_FILE, dtype=str).s1_id
    if sample_s1:
        dev = dev.sample(n=sample_s1, random_state=RANDOM_SEED)
    f = pd.read_parquet(feats_path)
    f = f[f.s1_id.isin(set(dev))].reset_index(drop=True)
    truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
    s1i, tgi = id_to_int(f.s1_id.values), id_to_int(f.cand_id.values)
    y = (pd.DataFrame({"s1_int": s1i, "tg_int": tgi})
         .merge(truth.assign(y=1), on=["s1_int", "tg_int"], how="left").y.fillna(0).astype(np.int8).values)
    fold = f.s1_id.map(pd.read_csv(FOLDS_FILE, dtype={"s1_id": str}).set_index("s1_id").fold).values
    dev_ints = id_to_int(dev.values)
    n_true = truth[truth.s1_int.isin(set(dev_ints))].groupby("s1_int").size().reindex(dev_ints, fill_value=0).values
    return f, y, fold, s1i, tgi, dev_ints, n_true


def oof_predict(X, y, fold, rounds, k):
    oof, iters, imp = np.zeros(len(y)), [], np.zeros(X.shape[1])
    for g in range(k):
        tr, te = (fold % k) != g, (fold % k) == g
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds,
                      valid_sets=[lgb.Dataset(X[te], y[te])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        imp += m.feature_importance("gain")
        print(f"  fold {g}: {int(te.sum()):,} pairs, best iteration {m.best_iteration}", flush=True)
    return oof, iters, imp


def one_owner(d):
    best = d.groupby("tg_int").p.transform("max")
    return d[d.p == best].drop_duplicates("tg_int")


def select_threshold(d, t):
    return d[d.p >= t]


def select_expected_f(d, c, r):
    """Greedy expected-F0.5 set per S1. c: extra expected true matches outside the candidates
    (blocker misses); r: weight on the 'predict nothing' option (singletons)."""
    d = d.sort_values(["s1_int", "p"], ascending=[True, False]).copy()
    g = d.groupby("s1_int", sort=False)
    d["k"] = g.cumcount() + 1
    nhat = g.p.transform("sum") + c
    d["E"] = 1.25 * g.p.cumsum() / (0.25 * nhat + d.k)
    d["lq"] = np.log1p(-d.p.clip(upper=1 - 1e-6))
    e0 = np.exp(d.groupby("s1_int", sort=False).lq.transform("sum")) * r
    best = d.loc[d.groupby("s1_int", sort=False).E.idxmax(), ["s1_int", "k", "E"]]
    d = d.merge(best.rename(columns={"k": "kbest", "E": "Ebest"}), on="s1_int")
    d["e0"] = e0.values
    return d[(d.k <= d.kbest) & (d.Ebest > d.e0)]


def macro_f05(sel, dev_ints, n_true):
    g = sel.groupby("s1_int").agg(n_pred=("y", "size"), tp=("y", "sum")).reindex(dev_ints, fill_value=0)
    return f05_from_counts(n_true, g.n_pred.values, g.tp.values), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=str(WORK_DIR / "feats" / "train_subset.parquet"))
    ap.add_argument("--sample-s1", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    f, y, fold, s1i, tgi, dev_ints, n_true = load(a.feats, a.sample_s1)
    feat_cols = [c for c in f.columns if c not in ID_COLS]
    X = f[feat_cols].astype(np.float32).values
    print(f"{len(f):,} pairs, {y.mean():.2%} true, {len(feat_cols)} features, "
          f"{len(dev_ints):,} dev S1 ({time.time()-t0:.0f}s)", flush=True)

    oof, iters, imp = oof_predict(X, y, fold, a.rounds, a.folds)
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"OOF pair ROC-AUC {roc_auc_score(y, oof):.4f} | avg precision {average_precision_score(y, oof):.4f}")

    d = one_owner(pd.DataFrame({"s1_int": s1i, "tg_int": tgi, "p": oof, "y": y}))
    results = []
    for t in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9):
        results.append(("threshold", {"t": t}, macro_f05(select_threshold(d, t), dev_ints, n_true)[0].mean()))
    for c in (0.0, 0.5, 1.0):
        for r in (1.0, 1.5, 2.0, 3.0):
            results.append(("expected_f", {"c": c, "r": r},
                            macro_f05(select_expected_f(d, c, r), dev_ints, n_true)[0].mean()))
    res = pd.DataFrame(results, columns=["method", "params", "macro_f05"]).sort_values("macro_f05", ascending=False)
    print("\ndecision rules (dev macro F0.5, OOF):")
    print(res.head(8).to_string(index=False))
    method, params, best = res.iloc[0]
    sel = select_threshold(d, **params) if method == "threshold" else select_expected_f(d, **params)
    f05, g = macro_f05(sel, dev_ints, n_true)
    single = n_true == 0
    country = pd.read_parquet(norm_path("train", 1), columns=["entity_id", "country"])
    cmap = pd.Series(country.country.values, index=id_to_int(country.entity_id.values))
    ctry = cmap.reindex(dev_ints).values
    print(f"\nBEST: {method} {params} -> dev macro F0.5 = {f05.mean():.4f}   (baseline on same dev: 0.682)")
    print(f"  pair precision {sel.y.mean():.4f} | pair recall {sel.y.sum() / n_true.sum():.4f} "
          f"| predicted-empty {(g.n_pred.values == 0).mean():.1%}")
    print(f"  singletons {f05[single].mean():.4f} (n={single.sum():,}) | matched S1 {f05[~single].mean():.4f}")
    for cn in sorted(set(ctry)):
        print(f"  {cn}: {f05[ctry == cn].mean():.4f} (n={(ctry == cn).sum():,})")
    top = pd.Series(imp, index=feat_cols).sort_values(ascending=False)
    print("\ntop features by gain:", ", ".join(f"{k} {v/top.sum():.1%}" for k, v in top.head(12).items()))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet_atomic(pd.DataFrame({"s1_id": f.s1_id, "cand_id": f.cand_id, "p": oof, "y": y}),
                        WORK_DIR / "preds" / "lgbm_oof_dev.parquet")
    n_iter = int(np.mean(iters))
    final = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=n_iter)
    final.save_model(str(MODEL_DIR / "lgbm_v1.txt"))
    (MODEL_DIR / "lgbm_v1.json").write_text(json.dumps(
        {"features": feat_cols, "decision": {"method": method, **params}, "dev_macro_f05": round(float(best), 4),
         "n_iter": n_iter, "params": PARAMS}, indent=2))
    print(f"\nsaved model + decision to {MODEL_DIR}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
