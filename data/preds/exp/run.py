"""Phase 2/3 experiment: same 30k dev S1s, 3 folds by S1, out-of-fold macro F0.5 vs full truth."""
import sys, time, json, numpy as np, pandas as pd, lightgbm as lgb, pyarrow as pa, pyarrow.dataset as pads
from src.data import id_to_int
E = "data/preds/exp/"
CONFIGS = {
 "A_hem":      dict(v2=False, params=dict(objective="binary", learning_rate=0.05, max_depth=6, num_leaves=31,
                     feature_fraction=0.8, bagging_fraction=0.8, verbose=-1, seed=42, num_threads=2), rounds=300, es=False),
 "B_newparams":dict(v2=False, params=dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=2), rounds=3000, es=True),
}
CONFIGS["C_v2_newparams"] = dict(CONFIGS["B_newparams"], v2=True)
CONFIGS["D_v2_hem"] = dict(CONFIGS["A_hem"], v2=True)
CONFIGS["C_half"] = dict(CONFIGS["C_v2_newparams"], half=True)
for _sd in (7, 99):
    CONFIGS[f"C_seed{_sd}"] = dict(CONFIGS["C_v2_newparams"], params=dict(CONFIGS["C_v2_newparams"]["params"], seed=_sd))

def data():
    ids = pd.read_csv("data/exp30k_s1_ids.csv", dtype=str).s1_id
    f = pads.dataset("data/feats/train_subset.parquet").to_table(filter=pads.field("s1_id").isin(pa.array(ids.values))).to_pandas()
    v = pd.read_parquet("data/feats/v2_exp30k.parquet")
    f = f.merge(v, on=["s1_id", "cand_id"], how="left")
    tr = pd.read_parquet("data/cache/truth_train.parquet")
    s1i, tgi = id_to_int(f.s1_id.values), id_to_int(f.cand_id.values)
    y = pd.DataFrame({"s1_int": s1i, "tg_int": tgi}).merge(tr.assign(y=1), how="left", on=["s1_int", "tg_int"]).y.fillna(0).values
    fold = f.s1_id.map(pd.read_csv("data/folds.csv", dtype={"s1_id": str}).set_index("s1_id").fold).values % 3
    return f, y, fold, s1i, tgi, ids, tr

def fit_fold(cfg, g):
    f, y, fold, *_ = data()
    base = [c for c in f.columns if c not in ("s1_id", "cand_id")]
    v2cols = list(pd.read_parquet("data/feats/v2_exp30k.parquet").columns[2:])
    cols = base if cfg["v2"] else [c for c in base if c not in v2cols]
    X = f[cols].to_numpy(np.float32); tr, te = fold != g, fold == g
    if cfg.get("half"):   # train on half of the training S1s (same test fold) to see the data-size slope
        keep = pd.util.hash_array(f.s1_id.values) % 2 == 0
        tr = tr & keep
    t = time.time()
    kw = dict(valid_sets=[lgb.Dataset(X[te], y[te])], callbacks=[lgb.early_stopping(50, verbose=False)]) if cfg["es"] else {}
    m = lgb.train(cfg["params"], lgb.Dataset(X[tr], y[tr]), num_boost_round=cfg["rounds"], **kw)
    p = m.predict(X[te], num_iteration=m.best_iteration or None)
    np.save(f"{E}{name}_f{g}.npy", p)
    imp = pd.Series(m.feature_importance("gain"), index=cols)
    imp.to_json(f"{E}{name}_f{g}_imp.json")
    print(f"{name} fold {g}: {len(cols)} feats, iters {m.best_iteration or cfg['rounds']}, {time.time()-t:.0f}s")

def evaluate(names):
    from src.models.decision import one_owner, select_threshold, select_expected_f, macro_f05
    f, y, fold, s1i, tgi, ids, tr = data()
    dev = id_to_int(ids.values)
    n_true = tr[tr.s1_int.isin(set(dev))].groupby("s1_int").size().reindex(dev, fill_value=0).values
    ctry = pd.read_parquet("data/norm/train_s1", columns=["entity_id", "country"])
    ctry = pd.Series(ctry.country.values, index=id_to_int(ctry.entity_id.values)).reindex(dev).values
    for name in names:
        p = np.zeros(len(y))
        for g in range(3): p[fold == g] = np.load(f"{E}{name}_f{g}.npy")
        d = one_owner(pd.DataFrame({"s1_int": s1i, "tg_int": tgi, "p": p, "y": y}))
        res = [(f"thr {t:.2f}", select_threshold(d, t)) for t in np.arange(0.3, 0.92, 0.04)]
        res += [(f"expF c{c} r{r}", select_expected_f(d, c, r)) for c in (0, 0.5) for r in (1, 1.5, 2, 3)]
        sc = [(k, macro_f05(s, dev, n_true)[0]) for k, s in res]
        k, fb = max(sc, key=lambda z: z[1].mean())
        thr = max((z for z in sc if z[0].startswith("thr")), key=lambda z: z[1].mean())
        by = " ".join(f"{c} {fb[ctry == c].mean():.4f}" for c in ("India", "US"))
        print(f"{name:15s} best {fb.mean():.4f} ({k}) | best threshold {thr[1].mean():.4f} ({thr[0]}) | {by}")

if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(sys.argv[2:])
    else:
        name, g = sys.argv[1], int(sys.argv[2]); fit_fold(CONFIGS[name], g)
