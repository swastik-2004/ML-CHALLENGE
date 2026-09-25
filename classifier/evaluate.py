"""
Evaluation Metrics Module for Business Entity Resolution (Macro F0.5).

Evaluates match predictions against ground truth at the Source 1 (S1) entity level.
Implements the exact challenge evaluation metric (Macro F0.5) with specific rules for singletons.
"""
import logging
from typing import Dict, Set, Any
import numpy as np

logger = logging.getLogger(__name__)


def compute_s1_entity_f05(
    true_targets: Set[str],
    pred_targets: Set[str],
    beta: float = 0.5
) -> float:
    """
    Computes the F0.5 score for a single Source 1 entity.

    Rules defined by challenge specification:
    1. Ground truth empty & Prediction empty (correct singleton): score = 1.0
    2. Ground truth empty & Prediction non-empty (false match): score = 0.0
    3. Ground truth non-empty & Prediction empty (missed match): score = 0.0
    4. General case: F_beta with beta = 0.5 (precision weighted 2x recall)

    Args:
        true_targets: Set of true matching target entity IDs (S2/S3).
        pred_targets: Set of predicted matching target entity IDs.
        beta: F-score beta parameter (default 0.5 for precision weighting).

    Returns:
        F0.5 score float in range [0.0, 1.0].
    """
    if not true_targets:
        # Ground truth is empty (singleton entity)
        return 1.0 if not pred_targets else 0.0

    if not pred_targets:
        # Ground truth has matches, but model predicted nothing
        return 0.0

    tp = len(true_targets & pred_targets)
    fp = len(pred_targets - true_targets)
    fn = len(true_targets - pred_targets)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return float(f_beta)


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Calculates Macro F0.5 metric across all Source 1 entities in the dataset.

    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matching target IDs.
        predictions: Dict mapping source1_entity_id -> set of predicted matching target IDs.
        beta: Metric beta weighting (default 0.5).

    Returns:
        Dictionary containing:
        - "macro_f05": Mean F0.5 score across all S1 entities
        - "total_entities": Number of evaluated S1 entities
        - "singleton_entities": Count of singletons in ground truth
        - "precision_micro": Overall micro-averaged precision
        - "recall_micro": Overall micro-averaged recall
    """
    all_s1_ids = set(ground_truth.keys()) | set(predictions.keys())
    if not all_s1_ids:
        logger.warning("No S1 entities provided for evaluation.")
        return {"macro_f05": 0.0, "total_entities": 0}

    s1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_count = 0

    for s1_id in all_s1_ids:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        if not gt_set:
            singleton_count += 1

        score = compute_s1_entity_f05(gt_set, pred_set, beta=beta)
        s1_scores.append(score)

        # Micro counts
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = float(np.mean(s1_scores)) if s1_scores else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    metrics_report = {
        "macro_f05": macro_f05,
        "total_entities": len(all_s1_ids),
        "singleton_entities": singleton_count,
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }

    logger.info(
        f"Evaluation Summary: Macro F0.5 = {macro_f05:.4f} "
        f"({len(all_s1_ids):,} S1 entities, {singleton_count:,} singletons)."
    )
    return metrics_report
