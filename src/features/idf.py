"""
Token document frequencies per country, counted over the S2+S3 records of a split.
Used by pair_features for rarity-weighted overlap: sharing "tarabanahalli" is strong evidence,
sharing "road" or "private" is not.

  python -m src.features.idf --split train      # -> data/cache/idf_train.parquet
  python -m src.features.idf --split test

Only tokens seen in >= 2 records are stored (singletons dominate the vocabulary and are all
"maximally rare"); a missing token is treated as document frequency 1.
Columns: field ('addr' | 'name'), country, tok, df. Row tok='' holds the number of records.
"""
from __future__ import annotations

import argparse
import glob
import math
import time

import numpy as np
import pandas as pd

from ..config import WORK_DIR
from ..normalize import norm_path

FIELDS = {"addr": "addr_norm", "name": "name_key"}


def idf_path(split: str):
    return WORK_DIR / "cache" / f"idf_{split}.parquet"


def _count_source(split: str, s: int) -> pd.DataFrame:
    """Token counts for one source (cached: data/cache/idf_{split}_s{s}.parquet, so a rerun resumes)."""
    from ..data import save_parquet_atomic
    cache = WORK_DIR / "cache" / f"idf_{split}_s{s}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    parts, nrec = {f: [] for f in FIELDS}, []
    for p in sorted(glob.glob(str(norm_path(split, s) / "*.parquet"))):
        d = pd.read_parquet(p, columns=["country", *FIELDS.values()])
        nrec.append(d.country.value_counts())
        for f, col in FIELDS.items():
            e = d[["country"]].assign(tok=d[col].str.split()).explode("tok").dropna()
            parts[f].append(e.groupby(["country", "tok"]).size())
    out = []
    for f in FIELDS:
        c = pd.concat(parts[f]).groupby(level=[0, 1]).sum().rename("df").reset_index()
        c.insert(0, "field", f)
        out.append(c)
    n = pd.concat(nrec).groupby(level=0).sum()
    for f in FIELDS:
        out.append(pd.DataFrame({"field": f, "country": n.index.values, "tok": "", "df": n.values}))
    res = pd.concat(out, ignore_index=True)
    res["df"] = res.df.astype(np.int32)
    save_parquet_atomic(res, cache)
    return res


def build_idf(split: str) -> pd.DataFrame:
    t0 = time.time()
    per = []
    for s in (2, 3):
        per.append(_count_source(split, s))
        print(f"  source {s} counted ({time.time() - t0:.0f}s)", flush=True)
    t = pd.concat(per, ignore_index=True).groupby(["field", "country", "tok"], sort=False).df.sum().reset_index()
    t = t[(t.df >= 2) | (t.tok == "")].reset_index(drop=True)
    t["df"] = t.df.astype(np.int32)
    return t


_CACHE: dict = {}


def load_idf(split: str):
    """{(field, country): (dict tok->df, n_records)}; cached per process."""
    if split not in _CACHE:
        if not idf_path(split).exists():
            raise FileNotFoundError(f"{idf_path(split)} missing: run  python -m src.features.idf --split {split}")
        t = pd.read_parquet(idf_path(split))
        tabs = {}
        for (f, c), g in t.groupby(["field", "country"], sort=False):
            n = int(g.loc[g.tok == "", "df"].sum()) or 1
            g = g[g.tok != ""]
            tabs[(f, c)] = (dict(zip(g.tok.values, g.df.values.tolist())), n)
        _CACHE[split] = tabs
    return _CACHE[split]


def idf_weight(df: int, n: int) -> float:
    return math.log((n + 1) / (df + 1)) + 1.0


if __name__ == "__main__":
    from ..data import save_parquet_atomic
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    a = ap.parse_args()
    t = build_idf(a.split)
    save_parquet_atomic(t, idf_path(a.split))
    print(t.groupby(["field", "country"]).size().to_string())
    print(f"wrote {idf_path(a.split)}: {len(t):,} rows")
