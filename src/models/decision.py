"""
Decision layer (P1): turn per-pair match probabilities into each S1's final match set.

Input frame columns: s1_int, tg_int, p (probability) [+ y label when scoring on train].
  one_owner          each S2/S3 record stays only under the S1 that gives it the highest p
                     (every record belongs to at most one S1 - confirmed on all 7.6M train pairs)
  select_threshold   keep pairs with p >= t
  select_expected_f  per S1, add candidates in p order while expected F0.5 rises; predict
                     nothing when P(no match) * r beats every non-empty set (singletons)
  apply_decision     apply a saved {"method": ..., params} dict (used for the test submission)
  macro_f05          per-S1 F0.5 over a fixed S1 list (S1s without rows = empty prediction)
Owned by P1; the model (train_lgbm.py) only produces p.
"""
import numpy as np
import pandas as pd

from ..evaluate import f05_from_counts


def one_owner(d: pd.DataFrame) -> pd.DataFrame:
    best = d.groupby("tg_int").p.transform("max")
    return d[d.p == best].drop_duplicates("tg_int")


def select_threshold(d: pd.DataFrame, t: float) -> pd.DataFrame:
    return d[d.p >= t]


def select_expected_f(d: pd.DataFrame, c: float = 0.0, r: float = 1.0) -> pd.DataFrame:
    """c: extra expected true matches outside the candidate list (blocker misses);
    r: weight on the 'predict nothing' option."""
    if d.empty:
        return d
    d = d.sort_values(["s1_int", "p"], ascending=[True, False]).copy()
    g = d.groupby("s1_int", sort=False)
    d["k"] = g.cumcount() + 1
    nhat = g.p.transform("sum") + c
    d["E"] = 1.25 * g.p.cumsum() / (0.25 * nhat + d.k)
    d["lq"] = np.log1p(-d.p.clip(upper=1 - 1e-6))
    d["e0"] = np.exp(d.groupby("s1_int", sort=False).lq.transform("sum")) * r
    best = d.loc[d.groupby("s1_int", sort=False).E.idxmax(), ["s1_int", "k", "E"]]
    d = d.merge(best.rename(columns={"k": "kbest", "E": "Ebest"}), on="s1_int")
    return d[(d.k <= d.kbest) & (d.Ebest > d.e0)].drop(columns=["k", "E", "lq", "e0", "kbest", "Ebest"])


def apply_decision(d: pd.DataFrame, decision: dict) -> pd.DataFrame:
    """decision = {"method": "threshold", "t": 0.6} or {"method": "expected_f", "c": 0, "r": 1.5}."""
    params = {k: v for k, v in decision.items() if k != "method"}
    d = one_owner(d)
    if decision["method"] == "threshold":
        return select_threshold(d, **params)
    if decision["method"] == "expected_f":
        return select_expected_f(d, **params)
    raise ValueError(f"unknown decision method {decision['method']!r}")


def macro_f05(sel: pd.DataFrame, s1_ints, n_true):
    """Per-S1 F0.5 array (aligned to s1_ints) and the per-S1 counts. sel needs s1_int and y."""
    g = sel.groupby("s1_int").agg(n_pred=("y", "size"), tp=("y", "sum")).reindex(s1_ints, fill_value=0)
    return f05_from_counts(n_true, g.n_pred.values, g.tp.values), g
