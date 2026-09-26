"""
Threshold Optimization Module for Macro F0.5 Metric Maximization.

Searches a configurable grid of probability thresholds to find the decision threshold
that maximizes the S1-entity level Macro F0.5 evaluation metric on validation data.

Does NOT execute search during import.
"""
import logging
from typing import Any, Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .config import THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP, S1_ID_COL, TARGET_ID_COL
from .evaluate import evaluate_macro_f05
from .utils import group_predictions_by_s1

logger = logging.getLogger(__name__)


def _optimize_threshold_loop(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float] = (THRESHOLD_MIN, THRESHOLD_MAX),
    threshold_step: float = THRESHOLD_STEP,
    s1_col: str = S1_ID_COL,
    target_id_col: str = TARGET_ID_COL,
) -> Dict[str, Any]:
    """
    Grid searches probability thresholds to maximize Macro F0.5 score.

    Args:
        val_df: Validation DataFrame containing candidate pairs (must contain s1_col and target_id_col).
        probabilities: 1D numpy array of predicted match probabilities P(match=1) for val_df rows.
        ground_truth: Dict mapping source1_entity_id -> set of true matching target IDs.
        threshold_range: Tuple of (min_threshold, max_threshold).
        threshold_step: Grid step increment.
        s1_col: Source 1 entity ID column name.
        target_id_col: Target candidate entity ID column name.

    Returns:
        Dictionary containing:
        - "best_threshold": Optimal probability threshold float
        - "best_macro_f05": Best validation Macro F0.5 score achieved
        - "threshold_grid_results": Dict mapping threshold string -> validation metrics report
    """
    if len(val_df) != len(probabilities):
        raise ValueError(
            f"Length mismatch: val_df has {len(val_df):,} rows, but probabilities has {len(probabilities):,} entries."
        )

    min_t, max_t = threshold_range
    thresholds = np.arange(min_t, max_t + 1e-9, threshold_step)

    # Attach probabilities to temporary validation DataFrame view
    eval_df = val_df[[s1_col, target_id_col]].copy()
    eval_df["probability"] = probabilities

    best_threshold = 0.5
    best_macro_f05 = -1.0
    grid_results = {}

    logger.info(
        f"Starting threshold grid search over {len(thresholds)} steps "
        f"range=[{min_t:.2f}, {max_t:.2f}] step={threshold_step:.2f}..."
    )

    for t in thresholds:
        t_val = round(float(t), 4)

        # Group predictions by S1 entity for threshold t
        pred_dict = group_predictions_by_s1(
            eval_df, probability_col="probability", threshold=t_val, s1_col=s1_col, target_id_col=target_id_col
        )

        # Compute Macro F0.5
        metrics = evaluate_macro_f05(ground_truth, pred_dict)
        macro_f05 = metrics["macro_f05"]
        grid_results[str(t_val)] = metrics

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = t_val

    logger.info(
        f"Threshold optimization complete. Best Threshold = {best_threshold:.4f}, "
        f"Best Validation Macro F0.5 = {best_macro_f05:.4f}"
    )

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "threshold_grid_results": grid_results,
    }


def optimize_threshold(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold_range: Tuple[float, float] = (THRESHOLD_MIN, THRESHOLD_MAX),
    threshold_step: float = THRESHOLD_STEP,
    s1_col: str = S1_ID_COL,
    target_id_col: str = TARGET_ID_COL,
) -> Dict[str, Any]:
    """Vectorised grid search, same inputs, outputs and rules as _optimize_threshold_loop
    (one-owner per target record, singleton scoring, macro F0.5 over ground-truth S1s plus
    validation S1s), but counts are computed with numpy per threshold instead of Python loops
    over every S1: seconds instead of ~14 min per 30k validation S1s."""
    if len(val_df) != len(probabilities):
        raise ValueError(
            f"Length mismatch: val_df has {len(val_df):,} rows, but probabilities has {len(probabilities):,} entries."
        )
    min_t, max_t = threshold_range
    thresholds = np.arange(min_t, max_t + 1e-9, threshold_step)

    d = pd.DataFrame({"s1": val_df[s1_col].astype(str).values,
                      "tg": val_df[target_id_col].astype(str).values,
                      "p": np.asarray(probabilities, dtype=float)})
    # one-owner: a record goes to the S1 with the highest probability. Filtering by the threshold
    # first and then taking the max keeps the same winner, so it can be done once for all thresholds.
    d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("tg")

    universe = pd.Index(sorted(set(ground_truth) | set(val_df[s1_col].astype(str).unique())))
    n_true = np.array([len(ground_truth.get(s, ())) for s in universe], dtype=np.int64)
    truth_pairs = {(s, t) for s, ts in ground_truth.items() for t in ts}
    y = np.fromiter(((s, t) in truth_pairs for s, t in zip(d.s1.values, d.tg.values)), dtype=bool, count=len(d))
    code = universe.get_indexer(d.s1.values)
    p = d.p.values
    n = len(universe)
    singletons = int((n_true == 0).sum())

    best_threshold, best_macro_f05, grid_results = 0.5, -1.0, {}
    for t in thresholds:
        t_val = round(float(t), 4)
        keep = p >= t_val
        n_pred = np.bincount(code[keep], minlength=n)
        tp = np.bincount(code[keep & y], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
            rec = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
            f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
        f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
        tot_tp, tot_pred, tot_true = int(tp.sum()), int(n_pred.sum()), int(n_true.sum())
        macro = float(f.mean()) if n else 0.0
        grid_results[str(t_val)] = {
            "macro_f05": macro, "total_entities": n, "singleton_entities": singletons,
            "precision_micro": tot_tp / tot_pred if tot_pred else 0.0,
            "recall_micro": tot_tp / tot_true if tot_true else 0.0,
            "total_tp": tot_tp, "total_fp": tot_pred - tot_tp, "total_fn": tot_true - tot_tp,
        }
        if macro > best_macro_f05:
            best_macro_f05, best_threshold = macro, t_val
    logger.info(f"Threshold optimization complete. Best Threshold = {best_threshold:.4f}, "
                f"Best Validation Macro F0.5 = {best_macro_f05:.4f}")
    return {"best_threshold": best_threshold, "best_macro_f05": best_macro_f05,
            "threshold_grid_results": grid_results}
