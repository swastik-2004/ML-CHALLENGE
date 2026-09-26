"""#3 error analysis on validation predictions (model-side only: pairs the blocker DID find)."""
import sys, numpy as np, pandas as pd
P = sys.argv[1] if len(sys.argv) > 1 else "data/preds/val_preds_current150k.parquet"
d = pd.read_parquet(P); thr = 0.68
d["fn"] = (d.y == 1) & ~d.selected; d["fp"] = (d.y == 0) & d.selected
d["fn_reject"] = d.fn & (d.p < thr); d["fn_owner"] = d.fn & (d.p >= thr)   # lost to another S1 by one-owner
g = d.groupby("s1_id")
d["s1_sel"] = g.selected.transform("sum"); d["s1_true"] = g.y.transform("sum")
d["rank_p"] = g.p.rank(ascending=False, method="first")
out = []
def pr(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
pr(f"pairs {len(d):,} | true in candidates {int(d.y.sum()):,} | selected {int(d.selected.sum()):,}")
pr(f"missed true pairs (FN) {int(d.fn.sum()):,}: rejected p<{thr} {int(d.fn_reject.sum()):,}, lost to one-owner {int(d.fn_owner.sum()):,}")
pr(f"wrong matches (FP) {int(d.fp.sum()):,}")
pr("\nby country:"); pr(d.groupby("country")[["fn_reject","fn_owner","fp"]].sum().to_string())
fn = d[d.fn_reject]
pr("\nrejected true pairs by probability band:")
pr(pd.cut(fn.p, [0, .05, .2, .4, .55, .7]).value_counts().sort_index().to_string())
pr(f"\nrejected true pairs where the SAME S1 has other selected matches: {(fn.s1_sel > 0).mean():.1%}")
pr(f"rejected true pairs whose S1 got NOTHING selected (whole entity missed): {(fn.s1_sel == 0).mean():.1%}")
pr(f"rank of the rejected true pair among its S1's candidates (by p): median {fn.rank_p.median():.0f}")
fp = d[d.fp]
pr(f"\nFP: {(fp.s1_true == 0).mean():.1%} are on S1s with no true candidate at all (singletons or blocker-missed)")
# per-S1 loss: which S1 types lose most F0.5
fn.to_parquet("reports/error_analysis/current150k/fn_rejected.parquet"); fp.to_parquet("reports/error_analysis/current150k/fp.parquet")
open("reports/error_analysis/current150k/summary.txt", "w").write("\n".join(out))
