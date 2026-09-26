"""Decision-layer + seed-average check with honest (cross-validated) tuning on the 30k OOF predictions."""
import sys, numpy as np, pandas as pd
sys.argv = ["x", "eval"]
import importlib.util
spec = importlib.util.spec_from_file_location("run", "data/preds/exp/run.py"); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
from src.data import id_to_int
from src.models.decision import one_owner, select_threshold, select_expected_f, macro_f05

f, y, fold, s1i, tgi, ids, tr = run.data()
dev = id_to_int(ids.values)
n_true = tr[tr.s1_int.isin(set(dev))].groupby("s1_int").size().reindex(dev, fill_value=0).values
ctry = pd.read_parquet("data/norm/train_s1", columns=["entity_id", "country"])
ctry = pd.Series(ctry.country.values, index=id_to_int(ctry.entity_id.values)).reindex(dev).values
s1fold = pd.Series(ids.values).map(pd.read_csv("data/folds.csv", dtype={"s1_id": str}).set_index("s1_id").fold).values % 3

def oof(name):
    p = np.zeros(len(y))
    for g in range(3): p[fold == g] = np.load(f"data/preds/exp/{name}_f{g}.npy")
    return p

def grid(p):
    d = one_owner(pd.DataFrame({"s1_int": s1i, "tg_int": tgi, "p": p, "y": y}))
    G = {}
    for t in np.round(np.arange(0.40, 0.96, 0.02), 2):
        G[("thr", t)] = macro_f05(select_threshold(d, t), dev, n_true)[0]
    for c in (0, 0.25, 0.5, 1.0):
        for r in (1, 1.5, 2, 2.5, 3, 4):
            G[("expF", c, r)] = macro_f05(select_expected_f(d, c, r), dev, n_true)[0]
    return G

def cv(G, keys, per_country):
    """pick the best key on 2 folds, score the 3rd; per_country picks separately for India / US."""
    out = np.zeros(len(dev)); picks = []
    for g in range(3):
        tr_, te_ = s1fold != g, s1fold == g
        groups = [ctry == c for c in ("India", "US")] if per_country else [np.ones(len(dev), bool)]
        for m in groups:
            k = max(keys, key=lambda k: G[k][tr_ & m].mean())
            out[te_ & m] = G[k][te_ & m]; picks.append(k)
    return out.mean(), picks

for name, p in [("single model", oof("C_v2_newparams")),
                ("3-seed average", (oof("C_v2_newparams") + oof("C_seed7") + oof("C_seed99")) / 3)]:
    G = grid(p)
    thr = [k for k in G if k[0] == "thr"]; allk = list(G)
    print(f"\n{name}:")
    for label, keys, pc in [("one threshold", thr, False), ("threshold per country", thr, True),
                            ("threshold or expected-F0.5", allk, False), ("either, per country", allk, True)]:
        sc, picks = cv(G, keys, pc)
        print(f"  {label:28s} {sc:.4f}   picks: {sorted(set(picks))[:4]}")
