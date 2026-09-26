"""Does 'similarity to already-accepted siblings' separate true from false among uncertain pairs?"""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as pads
from rapidfuzz import fuzz
d = pd.read_parquet("data/preds/val_preds_current150k.parquet")
band = d[(d.p >= 0.2) & (d.p < 0.68) & ~d.selected]
pos = band[band.y == 1]; pos = pos.sample(min(3000, len(pos)), random_state=0); neg = band[band.y == 0]; neg = neg.sample(min(3000, len(neg)), random_state=0); print("sample", len(pos), len(neg))
smp = pd.concat([pos, neg])
sel = d[d.selected & d.s1_id.isin(set(smp.s1_id))][["s1_id", "cand_id"]]
ids = pd.unique(pd.concat([smp.cand_id, sel.cand_id]))
rec = []
for s in (2, 3):
    rec.append(pads.dataset(f"data/norm/train_s{s}").to_table(columns=["entity_id", "name_core", "addr_norm"],
               filter=pads.field("entity_id").isin(pa.array([i for i in ids if i.startswith(f"S{s}")]))).to_pandas())
rec = pd.concat(rec).set_index("entity_id")
sib = sel.groupby("s1_id").cand_id.apply(list).to_dict()
rows = []
for s1, c, y, p in zip(smp.s1_id, smp.cand_id, smp.y, smp.p):
    sb = [x for x in sib.get(s1, []) if x != c]
    if not sb: rows.append((y, p, np.nan, np.nan, 0, 0)); continue
    rn, ra = rec.at[c, "name_core"], rec.at[c, "addr_norm"]
    ns = max(fuzz.token_set_ratio(rn, rec.at[x, "name_core"]) for x in sb) / 100
    as_ = max(fuzz.token_set_ratio(ra, rec.at[x, "addr_norm"]) for x in sb) / 100
    other = sum(1 for x in sb if x[1] != c[1])        # accepted siblings from the OTHER source
    rows.append((y, p, ns, as_, len(sb), other))
r = pd.DataFrame(rows, columns=["y", "p", "sib_name", "sib_addr", "n_sib", "n_sib_other_src"])
print(r.groupby("y")[["p", "sib_name", "sib_addr", "n_sib", "n_sib_other_src"]].mean().round(3).to_string())
from sklearn.metrics import roc_auc_score
m = r.dropna()
for c in ["p", "sib_name", "sib_addr"]:
    print(f"AUC of {c:9s} for true vs false in this band: {roc_auc_score(m.y, m[c]):.3f}")
m = m.assign(combo=m.p + m.sib_addr + m.sib_name)
print(f"AUC of p + sibling similarity: {roc_auc_score(m.y, m.combo):.3f}")
r.to_parquet("reports/error_analysis/current150k/sibling_signal.parquet")
