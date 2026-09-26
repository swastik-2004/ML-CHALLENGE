import pandas as pd, pyarrow as pa, pyarrow.dataset as pads
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 48)
fn = pd.read_parquet("reports/error_analysis/current150k/fn_rejected.parquet"); fp = pd.read_parquet("reports/error_analysis/current150k/fp.parquet")
fn = fn[(fn.p > .3)].sample(12, random_state=1); fp = fp.sample(8, random_state=1)
ids = pd.unique(pd.concat([fn.s1_id, fn.cand_id, fp.s1_id, fp.cand_id]))
rec = pd.concat([pads.dataset(f"data/norm/train_s{s}").to_table(columns=["entity_id", "name_core", "addr_norm"],
      filter=pads.field("entity_id").isin(pa.array([i for i in ids if i.startswith(f"S{s}")]))).to_pandas() for s in (1, 2, 3)]).set_index("entity_id")
def show(df, title):
    print(f"\n=== {title}")
    for r in df.itertuples():
        a, b = rec.loc[r.s1_id], rec.loc[r.cand_id]
        print(f"[{r.country} p={r.p:.2f}] S1: {a.name_core} | {a.addr_norm}\n{'':18s}{r.cand_id[:2]}: {b.name_core} | {b.addr_norm}")
show(fn, "TRUE pairs the model REJECTED (p 0.3-0.7)"); show(fp, "WRONG pairs the model ACCEPTED")
