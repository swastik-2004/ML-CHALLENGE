"""Per-country threshold vs one threshold, chosen on 2/3 of val S1s and scored on the other 1/3 (3 rotations)."""
import numpy as np, pandas as pd
from src.data import id_to_int
from src.models.decision import one_owner, select_threshold, macro_f05
d = pd.read_parquet("data/preds/val_preds_current150k.parquet")
tr = pd.read_parquet("data/cache/truth_train.parquet")
s1 = pd.Series(d.s1_id.unique()); s1i = id_to_int(s1.values)
n_true = tr[tr.s1_int.isin(set(s1i))].groupby("s1_int").size().reindex(s1i, fill_value=0).values
ctry = d.drop_duplicates("s1_id").set_index("s1_id").country.reindex(s1).values
fold = pd.util.hash_array(s1.values.astype(object)) % 3
x = one_owner(pd.DataFrame({"s1_int": id_to_int(d.s1_id.values), "tg_int": id_to_int(d.cand_id.values), "p": d.p.values, "y": d.y.values}))
T = np.round(np.arange(0.50, 0.92, 0.02), 2)
G = {t: macro_f05(select_threshold(x, t), s1i, n_true)[0] for t in T}
def cv(per_country):
    out = np.zeros(len(s1i)); picks = []
    for g in range(3):
        trn, tst = fold != g, fold == g
        groups = [ctry == c for c in ("India", "US")] if per_country else [np.ones(len(s1i), bool)]
        for m in groups:
            t = max(T, key=lambda t: G[t][trn & m].mean()); out[tst & m] = G[t][tst & m]; picks.append(t)
    return out, picks
one, p1 = cv(False); per, p2 = cv(True)
print(f"one threshold      {one.mean():.4f}  picks {sorted(set(p1))}")
print(f"per-country        {per.mean():.4f}  picks {sorted(set(p2))}")
for c in ("India", "US"):
    k = ctry == c; print(f"  {c}: one {one[k].mean():.4f} -> per-country {per[k].mean():.4f}")
best_in = {c: max(T, key=lambda t: G[t][ctry == c].mean()) for c in ("India", "US")}
print("best threshold on all val S1s:", best_in)
