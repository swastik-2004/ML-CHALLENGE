"""
Feature Schema and Validation Interface for the Classifier Module.

This module validates that incoming pairwise feature dataframes contain all required
entity identifier columns and numerical similarity feature columns expected by the classifier.

It performs NO feature calculations and makes NO assumptions about feature logic.
"""
import logging
from typing import List, Optional, Set
import pandas as pd

from .config import REQUIRED_ID_COLUMNS, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class FeatureSchemaError(ValueError):
    """Custom exception raised when a dataframe fails feature schema validation."""
    pass


def validate_required_columns(
    df: pd.DataFrame, required_cols: Optional[List[str]] = None
) -> None:
    """
    Validates that specified required columns exist in the dataframe.

    Args:
        df: Input pandas DataFrame.
        required_cols: List of column names that must be present.
                       Defaults to REQUIRED_ID_COLUMNS from config.

    Raises:
        FeatureSchemaError: If any required column is missing.
    """
    if required_cols is None:
        required_cols = REQUIRED_ID_COLUMNS

    missing_cols: Set[str] = set(required_cols) - set(df.columns)
    if missing_cols:
        error_msg = f"Dataframe is missing required columns: {sorted(list(missing_cols))}"
        logger.error(error_msg)
        raise FeatureSchemaError(error_msg)


def get_feature_columns(
    df: pd.DataFrame, feature_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Extracts or resolves the active feature column names from the dataframe.

    If feature_cols is provided (or configured in config.py), validates their presence.
    Otherwise, returns all non-identifier numerical columns.

    Args:
        df: Input pandas DataFrame.
        feature_cols: Explicit list of feature column names.

    Returns:
        List of validated feature column names.

    Raises:
        FeatureSchemaError: If specified feature columns are missing or empty.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    if feature_cols:
        validate_required_columns(df, feature_cols)
        return feature_cols

    # Fallback: Infer feature columns as non-identifier numeric columns
    non_feature_cols = set(REQUIRED_ID_COLUMNS)
    inferred_features = [
        col for col in df.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not inferred_features:
        error_msg = (
            "No feature columns specified in config.FEATURE_COLUMNS and no numeric "
            "pairwise feature columns could be inferred from the input dataframe."
        )
        logger.error(error_msg)
        raise FeatureSchemaError(error_msg)

    logger.info(f"Inferred {len(inferred_features)} feature columns from dataframe.")
    return inferred_features


def validate_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    required_id_cols: Optional[List[str]] = None,
) -> bool:
    """
    Complete validation check for incoming pairwise feature dataframes.

    Checks:
    1. Required entity identifier columns exist.
    2. Specified or inferred feature columns exist and are non-empty.
    3. Dataframe is non-empty.

    Args:
        df: Input pandas DataFrame containing candidate pairs and features.
        feature_cols: Optional explicit feature column list.
        required_id_cols: Optional explicit required ID column list.

    Returns:
        True if validation succeeds.

    Raises:
        FeatureSchemaError: If validation fails.
    """
    if df is None or df.empty:
        raise FeatureSchemaError("Input feature dataframe is empty or None.")

    validate_required_columns(df, required_id_cols)
    active_features = get_feature_columns(df, feature_cols)

    logger.info(
        f"Feature schema validation PASSED ({len(df):,} rows, "
        f"{len(active_features)} feature columns)."
    )
    return True
