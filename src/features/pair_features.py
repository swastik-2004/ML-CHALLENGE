"""
Pairwise features for candidate pairs (Task A).

Contract (agreed with the blocker, see team plan section 3):
  input   pairs DataFrame with columns  s1_id, cand_id  (+ optional passthrough columns kept
          as features, e.g. the blocker's n_rules / best_priority). One row per pair.
          The blocker's native cache (data/cache/candidate_pairs_{split}.parquet with integer
          s1_int / tg_int) is accepted directly and converted by to_pair_ids().
  records normalised rows from data/norm/{split}_s{n} (src.preprocessing.normalize).
  output  pairs + float32 feature columns, same row order.

Missing = NaN, never 0: an empty address is "not comparable", not "0% similar".
LightGBM handles NaN natively. Agreement features use 1 = agree, 0 = conflict, NaN = missing.

  python -m src.features.pair_features --split train                       # blocker cache, all S1
  python -m src.features.pair_features --split train --s1-ids data/dev_s1_ids.csv   # 100k dev only
  python -m src.features.pair_features --split test
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from ..normalize import norm_path

REC_COLS = ["entity_id", "country", "name_core", "name_key", "name_compact", "legal",
            "name_is_domain", "name_script", "addr_norm", "addr_key", "house_no", "numbers",
            "postal", "state", "addr_empty"]
LANDMARK_MARKERS = {"nr", "opp", "bhnd", "bsd", "adj"}
NAN = np.nan
F32 = np.float32


# ----------------------------------------------------------------------------- inputs
def _ints_to_ids(a) -> np.ndarray:
    a = pd.Series(a, dtype="int64")
    return ("S" + (a // 10**11).astype(str) + "-" + (a % 10**11).astype(str)).values


def to_pair_ids(p: pd.DataFrame) -> pd.DataFrame:
    """Accept either s1_id/cand_id strings or the blocker's s1_int/tg_int integers."""
    if {"s1_id", "cand_id"} <= set(p.columns):
        return p
    if {"s1_int", "tg_int"} <= set(p.columns):
        out = p.drop(columns=["s1_int", "tg_int"]).copy()
        out.insert(0, "cand_id", _ints_to_ids(p.tg_int.values))
        out.insert(0, "s1_id", _ints_to_ids(p.s1_int.values))
        return out
    raise ValueError(f"pairs need s1_id/cand_id or s1_int/tg_int columns, got {list(p.columns)}")


# ----------------------------------------------------------------------------- records
def gather_records(split: str, ids: Iterable[str]) -> pd.DataFrame:
    """Normalised records for the given entity ids, indexed by entity_id.
    The id filter runs inside pyarrow while scanning, so only the needed rows are ever
    converted to pandas (the old per-part pandas filter took ~11 min for 1.4M records)."""
    import pyarrow as pa
    import pyarrow.dataset as pads
    need = pd.unique(pd.Series(list(ids), dtype=object))
    value_set = pa.array(need, type=pa.string())
    parts = []
    for s in sorted({int(i[1]) for i in need}):
        dset = pads.dataset(str(norm_path(split, s)), format="parquet")   # ignores _SUCCESS
        tbl = dset.to_table(columns=REC_COLS, filter=pads.field("entity_id").isin(value_set))
        if tbl.num_rows:
            parts.append(tbl.to_pandas())
    rec = pd.concat(parts, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
    missing = pd.Index(need).difference(rec.index)
    if len(missing):
        raise ValueError(f"{len(missing)} ids not found in data/norm/{split}_*, e.g. {list(missing[:3])}")
    return rec


# ----------------------------------------------------------------------------- helpers
def _fuzzy(func, A, B, scale=100.0):
    return np.array([func(a, b) / scale if a and b else NAN for a, b in zip(A, B)], dtype=F32)


def _eq(A, B):
    return np.array([float(a == b) if a and b else NAN for a, b in zip(A, B)], dtype=F32)


def _jacc(A, B):
    out = np.full(len(A), NAN, dtype=F32)
    for i, (a, b) in enumerate(zip(A, B)):
        if a and b:
            x, y = set(a.split()), set(b.split())
            out[i] = len(x & y) / len(x | y)
    return out


def _grams(s, n=3):
    return {s} if len(s) < n else {s[i:i + n] for i in range(len(s) - n + 1)}


def _gram_jacc(A, B):
    out = np.full(len(A), NAN, dtype=F32)
    for i, (a, b) in enumerate(zip(A, B)):
        if a and b:
            x, y = _grams(a), _grams(b)
            out[i] = len(x & y) / len(x | y)
    return out


def _agree_sets(A, B):
    """1 same set, 0 disjoint, fraction if partial overlap, NaN if either side empty."""
    return _jacc(A, B)


def _len_ratio(A, B, tokens=False):
    out = np.full(len(A), NAN, dtype=F32)
    for i, (a, b) in enumerate(zip(A, B)):
        if a and b:
            la, lb = (len(a.split()), len(b.split())) if tokens else (len(a), len(b))
            out[i] = min(la, lb) / max(la, lb)
    return out


def _house(A, B):
    """house_eq (1/0/NaN), house_suffix (one is a proper suffix of the other: dropped digits,
    '12032' vs '2032'), house_sim (normalised Levenshtein, catches '649' vs '644')."""
    n = len(A)
    eq, suf, sim = (np.full(n, NAN, dtype=F32) for _ in range(3))
    for i, (a, b) in enumerate(zip(A, B)):
        if a and b:
            eq[i] = float(a == b)
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            suf[i] = float(a != b and len(short) >= 2 and long_.endswith(short))
            sim[i] = Levenshtein.normalized_similarity(a, b)
    return eq, suf, sim


def _landmarks(addr: str) -> str:
    toks = addr.split()
    out = []
    for i, t in enumerate(toks):
        if t in LANDMARK_MARKERS:
            out += [x for x in toks[i + 1:i + 3] if x not in LANDMARK_MARKERS and not x.isdigit()]
    return " ".join(out)



# ----------------------------------------------------------------------------- v2 features
# Targeted at the errors seen in missed / mis-scored dev pairs (26 Sep analysis):
#  * rarity: sharing "tarabanahalli" (28 records) is strong, sharing "nagar" (590k) is not
#  * digit damage: 8162 vs 162, 40800 vs 4080; house numbers may also sit anywhere in the address
#  * transliterated names: "sky technology" vs "skai teknoloji"
import re as _re

_HONORIFICS = {"sri", "shri", "sree", "shree", "smt", "m", "s", "ms"}
_SK_PAIRS = (("ph", "f"), ("ck", "k"), ("ch", "k"), ("sh", "s"), ("th", "t"), ("kh", "k"),
             ("gh", "g"), ("bh", "b"), ("dh", "d"), ("jh", "g"))
_SK_TABLE = str.maketrans({"c": "k", "q": "k", "j": "g", "z": "s", "w": "v", "x": "ks"})
_VOWELS = _re.compile(r"[aeiouy]")
_REPEAT = _re.compile(r"(.)\1+")


def _skel_token(t: str) -> str:
    for a, b in _SK_PAIRS:
        t = t.replace(a, b)
    t = _VOWELS.sub("", t.translate(_SK_TABLE))
    return _REPEAT.sub(r"\1", t)


def name_skeleton(name: str) -> str:
    """Consonant skeleton of a name ('skai teknoloji' -> 'sk tknlg'); honorific prefixes dropped."""
    toks = [t for t in (name or "").split() if not t.isdigit()]
    while toks and toks[0] in _HONORIFICS:
        toks = toks[1:]
    return " ".join(k for k in (_skel_token(t) for t in toks) if k)


def _near_number(a: str, b: str) -> bool:
    """Same number allowing one dropped / changed digit (length >= 3)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 3:
        return False
    return Levenshtein.distance(a, b, score_cutoff=1) <= 1


def _v2_features(L, R, split, f):
    from .idf import idf_weight, load_idf
    tabs = load_idf(split)
    n = len(L)
    out = {k: np.full(n, NAN, dtype=F32) for k in (
        "addr_idf_jacc", "addr_rare_shared", "addr_idf_max_shared", "name_idf_jacc",
        "name_rare_shared", "house_in_numbers", "numbers_near_frac", "name_skel_ratio", "name_skel_eq")}
    ctry = L.country.values
    la, ra = L.addr_norm.values, R.addr_norm.values
    ln, rn = L.name_key.values, R.name_key.values
    lh, rh = L.house_no.values, R.house_no.values
    lnum, rnum = L.numbers.values, R.numbers.values
    lcore, rcore = L.name_core.values, R.name_core.values
    for i in range(n):
        c = ctry[i]
        # address rarity
        at, an = tabs.get(("addr", c), ({}, 1))
        if la[i] and ra[i]:
            x, y = set(la[i].split()), set(ra[i].split())
            w = {t: idf_weight(at.get(t, 1), an) for t in x | y}
            inter = x & y
            out["addr_idf_jacc"][i] = sum(w[t] for t in inter) / sum(w.values())
            out["addr_rare_shared"][i] = sum(1 for t in inter if not t.isdigit() and len(t) >= 4
                                            and at.get(t, 1) <= 50)
            out["addr_idf_max_shared"][i] = max((w[t] for t in inter if not t.isdigit()), default=0.0)
        # name rarity
        nt, nn = tabs.get(("name", c), ({}, 1))
        if ln[i] and rn[i]:
            x, y = set(ln[i].split()), set(rn[i].split())
            w = {t: idf_weight(nt.get(t, 1), nn) for t in x | y}
            inter = x & y
            out["name_idf_jacc"][i] = sum(w[t] for t in inter) / sum(w.values())
            out["name_rare_shared"][i] = sum(1 for t in inter if nt.get(t, 1) <= 50)
        # numbers anywhere in the address, tolerant to one damaged digit
        a_nums = lnum[i].split() if lnum[i] else []
        b_nums = rnum[i].split() if rnum[i] else []
        if b_nums and (lh[i] or rh[i]):
            hits = [h for h in (lh[i], rh[i]) if h]
            other = [b_nums, a_nums]
            out["house_in_numbers"][i] = float(any(any(_near_number(h, o) for o in other[k])
                                                   for k, h in enumerate(hits) if other[k]))
        big = [a for a in a_nums if len(a) >= 3]
        if big and b_nums:
            out["numbers_near_frac"][i] = sum(any(_near_number(a, b) for b in b_nums) for a in big) / len(big)
        # transliteration-proof name similarity
        sa, sb = name_skeleton(lcore[i]), name_skeleton(rcore[i])
        if sa and sb:
            out["name_skel_ratio"][i] = fuzz.token_sort_ratio(sa, sb) / 100.0
            out["name_skel_eq"][i] = float(sa == sb)
    f.update(out)

# ----------------------------------------------------------------------------- features
def compute_features(pairs: pd.DataFrame, rec: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Row-aligned feature frame for pairs[s1_id, cand_id] using records `rec`.
    With `split`, also the v2 rarity / number / skeleton features (needs data/cache/idf_{split}.parquet)."""
    L = rec.reindex(pairs["s1_id"].values)
    R = rec.reindex(pairs["cand_id"].values)
    if L.name_key.isna().any() or R.name_key.isna().any():
        raise ValueError("pairs reference ids missing from rec (use gather_records)")
    f = {}
    ln, rn = L.name_core.values, R.name_core.values
    # --- name
    f["name_key_eq"] = _eq(L.name_key.values, R.name_key.values)
    f["name_compact_eq"] = _eq(L.name_compact.values, R.name_compact.values)
    f["name_ratio"] = _fuzzy(fuzz.ratio, ln, rn)
    f["name_token_sort"] = _fuzzy(fuzz.token_sort_ratio, ln, rn)
    f["name_token_set"] = _fuzzy(fuzz.token_set_ratio, ln, rn)
    f["name_partial"] = _fuzzy(fuzz.partial_ratio, ln, rn)
    f["name_jw"] = _fuzzy(JaroWinkler.normalized_similarity, ln, rn, scale=1.0)
    f["name_tok_jacc"] = _jacc(L.name_key.values, R.name_key.values)
    f["name_3gram_jacc"] = _gram_jacc(L.name_compact.values, R.name_compact.values)
    f["name_compact_ratio"] = _fuzzy(fuzz.ratio, L.name_compact.values, R.name_compact.values)
    f["name_len_ratio"] = _len_ratio(ln, rn)
    f["name_ntok_ratio"] = _len_ratio(ln, rn, tokens=True)
    f["legal_agree"] = _agree_sets(L.legal.values, R.legal.values)
    f["cand_is_domain"] = R.name_is_domain.values.astype(F32)
    f["script_diff"] = (L.name_script.values != R.name_script.values).astype(F32)
    f["cand_non_latin"] = (R.name_script.values != "latin").astype(F32)
    # --- address
    la, ra = L.addr_norm.values, R.addr_norm.values
    f["addr_key_eq"] = _eq(L.addr_key.values, R.addr_key.values)
    f["addr_ratio"] = _fuzzy(fuzz.ratio, la, ra)
    f["addr_token_set"] = _fuzzy(fuzz.token_set_ratio, la, ra)
    f["addr_token_sort"] = _fuzzy(fuzz.token_sort_ratio, la, ra)
    f["addr_tok_jacc"] = _jacc(L.addr_key.values, R.addr_key.values)
    f["addr_len_ratio"] = _len_ratio(la, ra)
    f["house_eq"], f["house_suffix"], f["house_sim"] = _house(L.house_no.values, R.house_no.values)
    f["numbers_jacc"] = _jacc(L.numbers.values, R.numbers.values)
    f["state_agree"] = _eq(L.state.values, R.state.values)
    f["postal_agree"] = _eq(L.postal.values, R.postal.values)
    f["landmark_jacc"] = _jacc([_landmarks(a) for a in la], [_landmarks(a) for a in ra])
    f["addr_missing"] = ((L.addr_empty.values == 1) | (R.addr_empty.values == 1)).astype(F32)
    # --- source
    f["cand_source"] = np.array([float(c[1]) for c in pairs["cand_id"].values], dtype=F32)
    if split is not None:
        _v2_features(L, R, split, f)
    return pd.DataFrame(f, index=pairs.index)


def add_context_features(df: pd.DataFrame,
                         score_cols=("name_token_set", "addr_token_set", "name_ratio")) -> pd.DataFrame:
    """Per-S1 context (all pairs of an S1 must be in df). Only meaningful on REAL blocker output:
    the candidate distribution at train time must match test time, so compute on both the same way."""
    df = df.copy()
    df["n_cands"] = df.groupby("s1_id", sort=False)["cand_id"].transform("size").astype(F32)
    for c in score_cols:
        if c in df.columns:
            s = df[c].fillna(-1.0)
            g = s.groupby(df["s1_id"], sort=False)
            df[f"{c}_rank"] = g.rank(ascending=False, method="min").astype(F32)
            df[f"{c}_gap"] = (g.transform("max") - s).astype(F32)
    # Candidate-level context (cand_n_s1, cand_rank_blk) is NOT computed here: it must be measured
    # on the full split, not on the subset passed in. See src/features/global_context.py.
    return df


def _task(args):
    """Worker entry point (top-level so Windows 'spawn' workers can import it)."""
    part, sub, split = args
    return pd.concat([part, compute_features(part, sub, split).reset_index(drop=True)], axis=1)


def build(split: str, pairs: pd.DataFrame, chunk: int = 250_000, context: bool = True,
          workers: int = 4, v2: bool = True) -> pd.DataFrame:
    """Features for all pairs. Chunks are processed in parallel; each worker only receives the
    records its chunk needs. Tasks are sent in small batches to keep memory flat."""
    from multiprocessing import get_context
    t0 = time.time()
    rec = gather_records(split, pd.concat([pairs.s1_id, pairs.cand_id]).unique())
    print(f"  gathered {len(rec):,} records in {time.time()-t0:.0f}s", flush=True)

    def job(i):
        part = pairs.iloc[i:i + chunk].reset_index(drop=True)
        ids = pd.unique(np.concatenate([part.s1_id.values, part.cand_id.values]))
        return part, rec.loc[ids], (split if v2 else None)

    starts = list(range(0, len(pairs), chunk))
    out = []
    if workers > 1 and len(starts) > 1:
        with get_context("spawn").Pool(workers) as pool:
            for b in range(0, len(starts), workers * 2):
                out += pool.map(_task, [job(i) for i in starts[b:b + workers * 2]])
                print(f"  features {min(starts[min(b + workers * 2, len(starts)) - 1] + chunk, len(pairs)):,}"
                      f"/{len(pairs):,}  {time.time()-t0:.0f}s", flush=True)
    else:
        for i in starts:
            out.append(_task(job(i)))
            print(f"  features {min(i + chunk, len(pairs)):,}/{len(pairs):,}  {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(out, ignore_index=True)
    return add_context_features(df) if context else df


if __name__ == "__main__":
    from ..config import WORK_DIR
    from ..data import save_parquet_atomic
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--pairs", help="default: data/cache/candidate_pairs_{split}.parquet (blocker output)")
    ap.add_argument("--s1-ids", help="csv with column s1_id: only build features for these S1 (e.g. dev)")
    ap.add_argument("--out", help="default: data/feats/{split}[_subset].parquet")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src = a.pairs or WORK_DIR / "cache" / f"candidate_pairs_{a.split}.parquet"
    if a.pairs:
        p = to_pair_ids(pd.read_parquet(src))
    else:   # blocker cache: attach candidate context measured on the FULL split before subsetting
        from .global_context import load_cache_with_context
        p = to_pair_ids(load_cache_with_context(a.split))
    if a.s1_ids:
        keep = set(pd.read_csv(a.s1_ids, dtype=str).s1_id)
        p = p[p.s1_id.isin(keep)].reset_index(drop=True)
    out = a.out or WORK_DIR / "feats" / f"{a.split}{'_subset' if a.s1_ids else ''}.parquet"
    print(f"{len(p):,} pairs for {p.s1_id.nunique():,} S1 from {src}", flush=True)
    feats = build(a.split, p, context=not a.no_context, workers=a.workers)
    save_parquet_atomic(feats, out)
    print(f"wrote {out}: {len(feats):,} rows x {feats.shape[1]} cols")
