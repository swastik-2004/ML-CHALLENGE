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
from .evaluate import evaluate_macro_f05
from .model import ClassifierModel
from .threshold import optimize_threshold
from .utils import (
    setup_logger,
    load_dataframe,
    separate_identifiers_and_features,
    group_predictions_by_s1,
    save_threshold_artifact,
)

logger = setup_logger(__name__)


def load_full_ground_truth(s1_ids: Set[str]) -> Dict[str, Set[str]]:
    """Complete true match sets from train_ground_truth.tsv for the given S1s (empty set for
    singletons), including matches the blocker never proposed as candidates."""
    from src.data import read_ground_truth, parse_id_list
    gt = read_ground_truth()
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]
    return {s1: set(parse_id_list(m)) for s1, m in
            zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values)}


def build_ground_truth_dict(
    df: pd.DataFrame,
    s1_col: str = S1_ID_COL,
    target_id_col: str = TARGET_ID_COL,
    label_col: str = LABEL_COL,
    all_s1_ids: Optional[Set[str]] = None,
    full_truth: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """
    Constructs a ground truth dictionary (s1_id -> set of true matching target IDs) from labeled data.

    Args:
        df: Labeled feature dataframe.
        s1_col: S1 entity ID column.
        target_id_col: Target candidate ID column.
        label_col: Binary label column (1 = match, 0 = non-match).
        all_s1_ids: Optional complete set of S1 entity IDs in the validation split (including those with 0 candidates).
        full_truth: Optional complete ground truth (load_full_ground_truth). When given, every S1 gets
                    its FULL true match set, so matches the blocker missed count as misses and S1s
                    without candidates are scored like on the leaderboard. Without it, only matches
                    that appear as labeled candidate rows are known (optimistic).

    Returns:
        Dict mapping source1_entity_id -> set of true matching target IDs.
    """
    gt_dict: Dict[str, Set[str]] = {}

    s1_universe = set(df[s1_col].unique())
    if all_s1_ids:
        s1_universe = s1_universe | set(all_s1_ids)

    if full_truth is not None:
        return {str(s1): set(full_truth.get(s1, set())) for s1 in s1_universe}

    # Initialize all S1 entities with empty sets
    for s1_id in s1_universe:
        gt_dict[str(s1_id)] = set()

    # Filter true matches
    matches = df[df[label_col] == 1]
    for s1_id, group in matches.groupby(s1_col):
        gt_dict[str(s1_id)] = set(group[target_id_col].astype(str).tolist())

    return gt_dict


def _evaluation_universe(
    df: pd.DataFrame, eval_s1_ids_path: Optional[Union[str, Path]]
) -> Tuple[Set[str], Optional[Dict[str, Set[str]]]]:
    """(S1 universe for the split, full ground truth or None).

    Uses the eval S1 list (default data/dev_s1_ids.csv) when every S1 in the features is in it,
    and the full train ground truth when every S1 of the universe is a train S1. Otherwise (e.g.
    synthetic or non-train feature files) falls back to the S1s present in the features and the
    candidate labels, with a warning that the validation score is then optimistic."""
    from src.config import DEV_IDS_FILE
    feat_s1 = set(df[S1_ID_COL].astype(str).unique())
    universe = feat_s1
    ids_path = Path(eval_s1_ids_path) if eval_s1_ids_path else DEV_IDS_FILE
    if ids_path.exists():
        listed = set(pd.read_csv(ids_path, dtype=str)["s1_id"])
        if feat_s1 <= listed:
            universe = listed
        else:
            logger.warning(f"Feature S1s are not all in {ids_path}: splitting over feature S1s only.")
    else:
        logger.warning(f"{ids_path} not found: splitting over feature S1s only (S1s without "
                       f"candidates are left out of validation).")

    try:
        full_truth = load_full_ground_truth(universe)
    except FileNotFoundError:
        full_truth = {}
    if len(full_truth) != len(universe):
        logger.warning("Not every S1 is in train_ground_truth.tsv: validation uses candidate labels "
                       "only, so the validation F0.5 ignores blocker misses (optimistic).")
        return feat_s1, None
    logger.info(f"Validation universe: {len(universe):,} S1 ({len(universe) - len(feat_s1):,} without "
                f"candidates), scored against the full ground truth.")
    return universe, full_truth


def train(
    feature_data_path: Optional[Union[str, Path]] = None,
    model_output_path: Optional[Union[str, Path]] = None,
    threshold_output_path: Optional[Union[str, Path]] = None,
    config_override: Optional[Dict[str, Any]] = None,
    eval_s1_ids_path: Optional[Union[str, Path]] = None,
) -> Tuple[ClassifierModel, float]:
    """
    Executes the classifier training and threshold optimization pipeline.

    Args:
        feature_data_path: Path to the pairwise feature dataset. Defaults to config.INPUT_FEATURE_PATH.
        model_output_path: Destination path for trained model artifact. Defaults to config.MODEL_ARTIFACT_PATH.
        threshold_output_path: Destination path for threshold artifact. Defaults to config.THRESHOLD_ARTIFACT_PATH.
        config_override: Optional dictionary of hyperparameter overrides.
        eval_s1_ids_path: CSV (column s1_id) of ALL S1s the feature file was built for, including
                          those the blocker gave no candidates. Defaults to data/dev_s1_ids.csv.
                          The train/val split is drawn from this list, and validation is scored
                          against the full ground truth, so the validation macro F0.5 estimates
                          the leaderboard score (blocker misses included).

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
        # src.features.pair_features writes unlabeled pairs: label them from the train ground truth
        from src.data import gt_pairs
        logger.info(f"'{LABEL_COL}' column missing: labeling pairs from train_ground_truth.tsv...")
        truth = gt_pairs().rename(columns={"s1_id": S1_ID_COL, "match_id": TARGET_ID_COL})
        truth[LABEL_COL] = 1
        df = df.merge(truth, on=[S1_ID_COL, TARGET_ID_COL], how="left")
        df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)
        logger.info(f"Labeled {len(df):,} pairs: {df[LABEL_COL].sum():,} true matches ({df[LABEL_COL].mean():.2%}).")

    # -------------------------------------------------------------------------
    # Step 3: Entity-Grouped Train / Validation Split
    # CRITICAL: We split BY S1 ENTITY (`source1_entity_id`), NOT by candidate rows!
    # This prevents data leakage across pairwise candidates of the same S1 entity.
    # -------------------------------------------------------------------------
    logger.info(f"[Step 3/6] Performing S1-entity grouped train/val split (val_ratio={VALIDATION_SPLIT})...")
    # The S1 universe is every S1 the features were built for (e.g. the 100k dev S1s), not just
    # those with candidates: S1s the blocker found nothing for still count on the leaderboard.
    s1_universe, full_truth = _evaluation_universe(df, eval_s1_ids_path)
    gss = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    universe = pd.Series(sorted(s1_universe))
    _, val_pos = next(gss.split(universe, groups=universe))
    val_s1 = set(universe.iloc[val_pos])

    is_val = df[S1_ID_COL].isin(val_s1)
    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()

    logger.info(
        f"Split complete: Train set = {len(train_df):,} candidate pairs "
        f"({train_df[S1_ID_COL].nunique():,} S1 entities), "
        f"Val set = {len(val_df):,} candidate pairs ({len(val_s1):,} S1 entities, "
        f"{len(val_s1) - val_df[S1_ID_COL].nunique():,} of them without candidates)."
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
    val_gt = build_ground_truth_dict(val_df, all_s1_ids=val_s1, full_truth=full_truth)

    opt_results = optimize_threshold(val_df, val_probs, val_gt)
    optimal_threshold = opt_results["best_threshold"]

    if full_truth is not None:
        # for reference: the old, candidate-only score at the chosen threshold (ignores blocker misses)
        cand_only = evaluate_macro_f05(
            build_ground_truth_dict(val_df),
            group_predictions_by_s1(val_df.assign(probability=val_probs), threshold=optimal_threshold),
        )["macro_f05"]
        n_true = sum(len(v) for v in val_gt.values())
        n_true_in_cands = int(val_df[LABEL_COL].sum())
        opt_results["scored_against"] = "full_ground_truth"
        opt_results["macro_f05_candidates_only"] = cand_only
        opt_results["val_recall_ceiling"] = n_true_in_cands / max(n_true, 1)
        logger.info(
            f"Validation macro F0.5 = {opt_results['best_macro_f05']:.4f} against the FULL ground truth "
            f"(leaderboard estimate); {cand_only:.4f} counting only matches present as candidates. "
            f"Blocker kept {n_true_in_cands:,}/{n_true:,} true matches of the val S1s "
            f"({opt_results['val_recall_ceiling']:.1%} recall ceiling)."
        )
    else:
        opt_results["scored_against"] = "candidate_labels_only"

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
    parser.add_argument("--eval-s1-ids", type=str, default=None,
                        help="CSV (s1_id) of all S1s the features were built for; default data/dev_s1_ids.csv")
    args = parser.parse_args()

    train(
        feature_data_path=args.features_path,
        model_output_path=args.model_out,
        threshold_output_path=args.threshold_out,
        eval_s1_ids_path=args.eval_s1_ids,
    )
