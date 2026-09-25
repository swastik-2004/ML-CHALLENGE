"""
Macro F_0.5 evaluation, matching the challenge specification.

Per Source 1 entity:
  - true set empty (singleton): 1.0 if prediction empty, else 0.0
  - true set non-empty, prediction empty: 0.0   (statement is silent; safe reading)
  - otherwise F_0.5 = 1.25 * P * R / (0.25 * P + R)
Then averaged over every Source 1 entity in the evaluation set.

CLI:
  python -m src.evaluate --pred output/matching_results_train.tsv [--ids data/dev_s1_ids.csv]
"""
import argparse
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd

BETA = 0.5


def compute_entity_f_beta(pred_set: Set[str], true_set: Set[str], beta: float = BETA) -> float:
    """F_beta for a single Source 1 entity (see module docstring for edge cases)."""
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def f05_from_counts(n_true, n_pred, tp):
    """Vectorised per-S1 F0.5 from counts; identical rules to compute_entity_f_beta."""
    n_true, n_pred, tp = (np.asarray(x, dtype=np.float64) for x in (n_true, n_pred, tp))
    f = np.zeros(len(n_true))
    single = n_true == 0
    f[single] = (n_pred[single] == 0).astype(np.float64)
    m = (~single) & (tp > 0)
    P, R = tp[m] / n_pred[m], tp[m] / n_true[m]
    f[m] = (1 + BETA**2) * P * R / (BETA**2 * P + R)
    return f


def _to_map(df_or_map) -> Dict[str, Set[str]]:
    if isinstance(df_or_map, dict):
        return df_or_map
    ids = df_or_map["source1_entity_id"].astype(str).str.strip().values
    vals = df_or_map["matched_entity_ids"].fillna("").astype(str).values
    return {i: {x.strip() for x in v.split(",") if x.strip()} for i, v in zip(ids, vals)}


def score_entities(pred, gt, ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One row per evaluated S1: f05, tp, n_pred, n_true. ids restricts the evaluation set."""
    pred_map, gt_map = _to_map(pred), _to_map(gt)
    ids = list(gt_map.keys()) if ids is None else list(ids)
    empty: Set[str] = set()
    rows = []
    for s1 in ids:
        t = gt_map.get(s1, empty)
        p = pred_map.get(s1, empty)
        rows.append((s1, compute_entity_f_beta(p, t), len(p & t), len(p), len(t)))
    return pd.DataFrame(rows, columns=["s1_id", "f05", "tp", "n_pred", "n_true"])


def evaluate_macro_f05(predictions, ground_truth, ids: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Macro F_0.5 plus pair-level precision/recall (kept for backward compatibility)."""
    e = score_entities(predictions, ground_truth, ids)
    tp, npred, ntrue = e.tp.sum(), e.n_pred.sum(), e.n_true.sum()
    return {
        "macro_f05": float(e.f05.mean()) if len(e) else 0.0,
        "total_evaluated_entities": int(len(e)),
        "pair_precision": float(tp / npred) if npred else 0.0,
        "pair_recall": float(tp / ntrue) if ntrue else 0.0,
        "pred_empty_rate": float((e.n_pred == 0).mean()) if len(e) else 0.0,
    }


def score_report(predictions, ground_truth, ids: Optional[Iterable[str]] = None,
                 country: Optional[Dict[str, str]] = None) -> str:
    """Human-readable breakdown: overall, singletons vs matched, by #true matches, by country."""
    e = score_entities(predictions, ground_truth, ids)
    if country is not None:
        e["country"] = e.s1_id.map(country).fillna("?")
    e["bucket"] = pd.cut(e.n_true, [-1, 0, 1, 2, 3, 4, 5, 99],
                         labels=["0 (singleton)", "1", "2", "3", "4", "5", "6+"])
    tp, npred, ntrue = e.tp.sum(), e.n_pred.sum(), e.n_true.sum()
    lines = [f"macro F0.5 = {e.f05.mean():.4f}  on {len(e):,} S1  |  pair P = {tp/max(npred,1):.4f}  "
             f"pair R = {tp/max(ntrue,1):.4f}  |  predicted-empty = {(e.n_pred==0).mean():.1%}"]
    g = e.groupby("bucket", observed=True).agg(n=("f05", "size"), f05=("f05", "mean"),
                                               empty_pred=("n_pred", lambda s: (s == 0).mean()))
    lines.append("by true-match count:\n" + g.to_string(float_format=lambda x: f"{x:.4f}"))
    if country is not None:
        c = e.groupby("country").agg(n=("f05", "size"), f05=("f05", "mean"))
        lines.append("by country:\n" + c.to_string(float_format=lambda x: f"{x:.4f}"))
    return "\n".join(lines)


if __name__ == "__main__":
    from .data import read_ground_truth, READ_KW, read_source
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", help="csv with column s1_id restricting the evaluation set")
    a = ap.parse_args()
    pred = pd.read_csv(a.pred, **READ_KW)
    ids = pd.read_csv(a.ids, dtype=str)["s1_id"].tolist() if a.ids else None
    s1 = read_source("train", 1, usecols=["entity_id", "country"])
    print(score_report(pred, read_ground_truth(), ids, dict(zip(s1.entity_id, s1.country))))
