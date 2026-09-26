"""
Classifier Training Pipeline.

Orchestrates data loading, schema validation, entity-grouped train/validation splitting,
gradient boosting model fitting, threshold optimization, and artifact serialization.

NOTE: This file provides the clean pipeline structure. The pairwise developer will integrate
their feature output at the indicated TODO integration points.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    INPUT_FEATURE_PATH,
    MODEL_ARTIFACT_PATH,
    THRESHOLD_ARTIFACT_PATH,
    S1_ID_COL,
    TARGET_ID_COL,
    LABEL_COL,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)
from .feature_schema import validate_features, get_feature_columns, standardize_columns
from .model import ClassifierModel
from .threshold import optimize_threshold
from .utils import (
    setup_logger,
    load_dataframe,
    separate_identifiers_and_features,
    save_threshold_artifact,
)

logger = setup_logger(__name__)


def build_ground_truth_dict(
    df: pd.DataFrame,
    s1_col: str = S1_ID_COL,
    target_id_col: str = TARGET_ID_COL,
    label_col: str = LABEL_COL,
    all_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Constructs a ground truth dictionary (s1_id -> set of true matching target IDs) from labeled data.

    Args:
        df: Labeled feature dataframe.
        s1_col: S1 entity ID column.
        target_id_col: Target candidate ID column.
        label_col: Binary label column (1 = match, 0 = non-match).
        all_s1_ids: Optional complete set of S1 entity IDs in the validation split (including those with 0 candidates).

    Returns:
        Dict mapping source1_entity_id -> set of true matching target IDs.
    """
    gt_dict: Dict[str, Set[str]] = {}

    s1_universe = set(df[s1_col].unique())
    if all_s1_ids:
        s1_universe = s1_universe | set(all_s1_ids)

    # Initialize all S1 entities with empty sets
    for s1_id in s1_universe:
        gt_dict[str(s1_id)] = set()

    # Filter true matches
    matches = df[df[label_col] == 1]
    for s1_id, group in matches.groupby(s1_col):
        gt_dict[str(s1_id)] = set(group[target_id_col].astype(str).tolist())

    return gt_dict


def train(
    feature_data_path: Optional[Union[str, Path]] = None,
    model_output_path: Optional[Union[str, Path]] = None,
    threshold_output_path: Optional[Union[str, Path]] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Tuple[ClassifierModel, float]:
    """
    Executes the classifier training and threshold optimization pipeline.

    Args:
        feature_data_path: Path to the pairwise feature dataset. Defaults to config.INPUT_FEATURE_PATH.
        model_output_path: Destination path for trained model artifact. Defaults to config.MODEL_ARTIFACT_PATH.
        threshold_output_path: Destination path for threshold artifact. Defaults to config.THRESHOLD_ARTIFACT_PATH.
        config_override: Optional dictionary of hyperparameter overrides.

    Returns:
        Tuple of (trained_model_instance, optimal_threshold_float).

    Steps:
        1. Load pairwise feature dataset (TODO: Pairwise Dev connects output here)
        2. Validate schema
        3. Perform entity-grouped train/validation split (Split by S1 entity, NEVER candidate rows)
        4. Train ClassifierModel
        5. Optimize threshold for Macro F0.5
        6. Persist artifacts
    """
    path = feature_data_path or INPUT_FEATURE_PATH
    m_path = model_output_path or MODEL_ARTIFACT_PATH
    t_path = threshold_output_path or THRESHOLD_ARTIFACT_PATH

    logger.info("=== Starting Classifier Training Pipeline ===")

    # -------------------------------------------------------------------------
    # TODO (Pairwise Developer Integration Point 1):
    # Ensure the feature file exists at `feature_data_path` or pass the dataframe directly.
    # -------------------------------------------------------------------------
    logger.info(f"[Step 1/6] Loading pairwise feature dataset from: {path}")
    df = standardize_columns(load_dataframe(path))

    # -------------------------------------------------------------------------
    # Step 2: Validate Schema
    # -------------------------------------------------------------------------
    logger.info("[Step 2/6] Validating feature schema...")
    validate_features(df)
    feature_cols = get_feature_columns(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' missing from input dataframe for training.")

    # -------------------------------------------------------------------------
    # Step 3: Entity-Grouped Train / Validation Split
    # CRITICAL: We split BY S1 ENTITY (`source1_entity_id`), NOT by candidate rows!
    # This prevents data leakage across pairwise candidates of the same S1 entity.
    # -------------------------------------------------------------------------
    logger.info(f"[Step 3/6] Performing S1-entity grouped train/val split (val_ratio={VALIDATION_SPLIT})...")
    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)

    groups = df[S1_ID_COL]
    train_idx, val_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    logger.info(
        f"Split complete: Train set = {len(train_df):,} candidate pairs "
        f"({train_df[S1_ID_COL].nunique():,} S1 entities), "
        f"Val set = {len(val_df):,} candidate pairs ({val_df[S1_ID_COL].nunique():,} S1 entities)."
    )

    # Separate features and labels
    _, X_train = separate_identifiers_and_features(train_df, feature_columns=feature_cols)
    y_train = train_df[LABEL_COL]

    _, X_val = separate_identifiers_and_features(val_df, feature_columns=feature_cols)
    y_val = val_df[LABEL_COL]

    # -------------------------------------------------------------------------
    # Step 4: Fit Model
    # -------------------------------------------------------------------------
    logger.info("[Step 4/6] Fitting classifier model...")
    model_wrapper = ClassifierModel(params=config_override)
    model_wrapper.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Step 5: Threshold Optimization for Macro F0.5
    # -------------------------------------------------------------------------
    logger.info("[Step 5/6] Optimizing decision threshold for Macro F0.5 metric...")
    val_probs = model_wrapper.predict_proba(X_val)
    val_gt = build_ground_truth_dict(val_df)

    opt_results = optimize_threshold(val_df, val_probs, val_gt)
    optimal_threshold = opt_results["best_threshold"]

    # -------------------------------------------------------------------------
    # Step 6: Save Model & Threshold Artifacts
    # -------------------------------------------------------------------------
    logger.info("[Step 6/6] Saving trained model and threshold artifacts...")
    model_wrapper.save(m_path)
    save_threshold_artifact(opt_results, t_path)

    logger.info("=== Classifier Training Pipeline Completed Successfully ===")
    return model_wrapper, optimal_threshold


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Business Entity Resolution Classifier")
    parser.add_argument("--features-path", type=str, default=str(INPUT_FEATURE_PATH))
    parser.add_argument("--model-out", type=str, default=str(MODEL_ARTIFACT_PATH))
    parser.add_argument("--threshold-out", type=str, default=str(THRESHOLD_ARTIFACT_PATH))
    args = parser.parse_args()

    train(
        feature_data_path=args.features_path,
        model_output_path=args.model_out,
        threshold_output_path=args.threshold_out,
    )
