"""
Classifier Inference Pipeline.

Loads trained model and threshold artifacts, scores candidate pairs from pairwise features,
applies the decision threshold, and outputs predictions.

NOTE: Does NOT enforce one-to-one matching constraints. One S1 entity can match 0, 1, or multiple targets.
"""
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .config import (
    MODEL_ARTIFACT_PATH,
    THRESHOLD_ARTIFACT_PATH,
    PREDICTION_OUTPUT_PATH,
    S1_ID_COL,
    TARGET_ID_COL,
    TARGET_SOURCE_COL,
    DEFAULT_THRESHOLD,
)
from .feature_schema import validate_features, get_feature_columns, standardize_columns
from .model import ClassifierModel
from .utils import (
    setup_logger,
    load_dataframe,
    separate_identifiers_and_features,
    group_predictions_by_s1,
    save_predictions_tsv,
    load_threshold_artifact,
)

logger = setup_logger(__name__)


def predict(
    feature_data_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    threshold_path: Optional[Union[str, Path]] = None,
    threshold_override: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    enforce_one_owner: bool = True,
) -> pd.DataFrame:
    """
    Executes prediction pipeline over input candidate pairwise features.

    Args:
        feature_data_path: Path to the test candidate pairwise feature dataset.
        model_path: Optional model artifact path. Defaults to config.MODEL_ARTIFACT_PATH.
        threshold_path: Optional threshold artifact path. Defaults to config.THRESHOLD_ARTIFACT_PATH.
        threshold_override: Optional explicit probability threshold float.
        output_path: Optional destination TSV file path (`matching_results.tsv`).
        enforce_one_owner: Whether to assign each candidate target to at most one S1 entity.

    Returns:
        DataFrame containing candidate scores with columns:
        [source1_entity_id, target_entity_id, target_source, probability, prediction]
    """
    m_path = model_path or MODEL_ARTIFACT_PATH
    t_path = threshold_path or THRESHOLD_ARTIFACT_PATH
    out_path = output_path or PREDICTION_OUTPUT_PATH

    logger.info("=== Starting Classifier Inference Pipeline ===")

    # 1. Load Trained Model Artifact
    logger.info(f"[Step 1/5] Loading trained model from: {m_path}")
    model = ClassifierModel.load(m_path)

    # 2. Determine Decision Threshold
    if threshold_override is not None:
        threshold = float(threshold_override)
        logger.info(f"Using explicitly provided threshold: {threshold:.4f}")
    elif Path(t_path).exists():
        thresh_artifact = load_threshold_artifact(t_path)
        threshold = float(thresh_artifact.get("best_threshold", DEFAULT_THRESHOLD))
        logger.info(f"Loaded optimal threshold from artifact: {threshold:.4f}")
    else:
        threshold = DEFAULT_THRESHOLD
        logger.warning(f"No threshold artifact found. Using default threshold: {threshold:.4f}")

    # 3. Load Pairwise Features
    logger.info(f"[Step 2/5] Loading test candidate pairwise features from: {feature_data_path}")
    df_features_raw = standardize_columns(load_dataframe(feature_data_path))

    # 4. Validate Feature Schema
    logger.info("[Step 3/5] Validating feature schema...")
    validate_features(df_features_raw)
    feature_cols = get_feature_columns(df_features_raw)

    df_ids, X_test = separate_identifiers_and_features(df_features_raw, feature_columns=feature_cols)

    # 5. Model Inference (predict_proba)
    logger.info(f"[Step 4/5] Scoring {len(X_test):,} candidate pairs...")
    probabilities = model.predict_proba(X_test)
    binary_preds = (probabilities >= threshold).astype(int)

    # Build detailed output dataframe
    results_df = df_ids.copy()
    results_df["probability"] = probabilities
    results_df["prediction"] = binary_preds

    # 6. Format and Save Official Submissions (matching_results.tsv)
    logger.info(f"[Step 5/5] Grouping predictions (one_owner={enforce_one_owner}) and writing output to: {out_path}")
    grouped_preds = group_predictions_by_s1(
        results_df,
        probability_col="probability",
        threshold=threshold,
        s1_col=S1_ID_COL,
        target_id_col=TARGET_ID_COL,
        enforce_one_owner=enforce_one_owner,
    )

    save_predictions_tsv(grouped_preds, out_path, s1_id_col=S1_ID_COL)

    logger.info(
        f"Inference complete: Processed {len(results_df):,} candidate pairs. "
        f"Matched S1 entities = {sum(1 for s in grouped_preds.values() if s):,}, "
        f"Singletons = {sum(1 for s in grouped_preds.values() if not s):,}."
    )

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Classifier Inference on Candidate Pairs")
    parser.add_argument("--features-path", type=str, required=True, help="Path to pairwise feature dataset")
    parser.add_argument("--model-path", type=str, default=str(MODEL_ARTIFACT_PATH))
    parser.add_argument("--threshold-path", type=str, default=str(THRESHOLD_ARTIFACT_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-path", type=str, default=str(PREDICTION_OUTPUT_PATH))
    args = parser.parse_args()

    predict(
        feature_data_path=args.features_path,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        threshold_override=args.threshold,
        output_path=args.output_path,
    )
