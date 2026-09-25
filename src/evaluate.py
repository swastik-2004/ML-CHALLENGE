"""
Macro F_0.5 evaluation metric implementation strictly matching challenge specifications.
"""
from typing import Dict, List, Set
import pandas as pd
import numpy as np


def compute_entity_f_beta(pred_set: Set[str], true_set: Set[str], beta: float = 0.5) -> float:
    """
    Compute F_beta for a single Source 1 entity.
    Handles singletons (true_set is empty):
      - 1.0 if pred_set is also empty
      - 0.0 if pred_set is non-empty
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        return 0.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    return float(f_beta)


def evaluate_macro_f05(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities.

    predictions_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    ground_truth_df: DataFrame with ['source1_entity_id', 'matched_entity_ids']
    """
    # Parse ground truth
    gt_map: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        val = row.get("matched_entity_ids")
        if pd.isna(val) or not str(val).strip():
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set(x.strip() for x in str(val).split(",") if x.strip())

    # Parse predictions
    pred_map: Dict[str, Set[str]] = {}
    for _, row in predictions_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        val = row.get("matched_entity_ids")
        if pd.isna(val) or not str(val).strip():
            pred_map[s1_id] = set()
        else:
            pred_map[s1_id] = set(x.strip() for x in str(val).split(",") if x.strip())

    # Calculate per entity score
    scores = []
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        score = compute_entity_f_beta(pred_set, true_set, beta=0.5)
        scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    return {
        "macro_f05": macro_f05,
        "total_evaluated_entities": len(scores)
    }
