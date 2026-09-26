"""
Utility functions for the Classifier Module.

Provides generic data loading, grouping, saving, and artifact handling helpers.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import joblib

from .config import REQUIRED_ID_COLUMNS, S1_ID_COL, TARGET_ID_COL

logger = logging.getLogger(__name__)


def setup_logger(name: str = "classifier", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger instance."""
    logger_inst = logging.getLogger(name)
    if not logger_inst.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger_inst.addHandler(handler)
        logger_inst.setLevel(level)
    return logger_inst


def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Generic data loader supporting Parquet, CSV, and TSV file formats.

    Args:
        file_path: Path to the input dataset file.

    Returns:
        Loaded pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist at: {path}")

    suffix = path.suffix.lower()
    logger.info(f"Loading dataframe from: {path}")

    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        # Fallback attempting TSV then CSV
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)


def separate_identifiers_and_features(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates identifier columns (e.g. S1 entity ID, target entity ID) from numerical feature columns.

    Args:
        df: Input pandas DataFrame.
        id_columns: List of identifier column names. Defaults to REQUIRED_ID_COLUMNS.
        feature_columns: Explicit list of feature column names. If None, infers remaining columns.

    Returns:
        Tuple of (id_dataframe, feature_dataframe).
    """
    if id_columns is None:
        id_columns = [col for col in REQUIRED_ID_COLUMNS if col in df.columns]

    df_ids = df[id_columns].copy()

    if feature_columns is not None:
        df_features = df[feature_columns].copy()
    else:
        df_features = df.drop(columns=id_columns, errors="ignore").select_dtypes(include=["number"]).copy()

    return df_ids, df_features


def group_predictions_by_s1(
    df: pd.DataFrame,
    probability_col: str = "probability",
    threshold: float = 0.5,
    s1_col: str = S1_ID_COL,
    target_id_col: str = TARGET_ID_COL,
    enforce_one_owner: bool = True,
) -> Dict[str, Set[str]]:
    """
    Groups predicted matches by Source 1 entity ID based on a probability decision threshold,
    enforcing the one-owner rule (each S2/S3 record stays under at most one S1).

    Args:
        df: DataFrame containing predictions with s1_col, target_id_col, and probability_col.
        probability_col: Name of predicted match probability column.
        threshold: Minimum probability required to count as a match.
        s1_col: Source 1 entity ID column name.
        target_id_col: Target candidate ID column name.
        enforce_one_owner: Whether to assign each target record only to its highest-probability S1.

    Returns:
        Dict mapping source1_entity_id -> set of predicted matching target IDs.
    """
    grouped_preds: Dict[str, Set[str]] = {}

    if df is None or df.empty:
        return grouped_preds

    # Filter rows satisfying decision threshold
    matched_df = df[df[probability_col] >= threshold]

    # Enforce One-Owner Rule (each target entity belongs to at most one S1 entity)
    if enforce_one_owner and not matched_df.empty and target_id_col in matched_df.columns:
        best_p = matched_df.groupby(target_id_col)[probability_col].transform("max")
        matched_df = matched_df[matched_df[probability_col] == best_p].drop_duplicates(target_id_col)

    for s1_id, group in matched_df.groupby(s1_col):
        grouped_preds[str(s1_id)] = set(group[target_id_col].astype(str).tolist())

    return grouped_preds


def save_predictions_tsv(
    predictions_dict: Dict[str, Set[str]],
    output_path: Union[str, Path],
    s1_id_col: str = S1_ID_COL,
    matched_col: str = "matched_entity_ids",
) -> None:
    """
    Saves grouped predictions into the official TSV format (`matching_results.tsv`).

    Args:
        predictions_dict: Dict mapping source1_entity_id -> set of predicted target IDs.
        output_path: Destination TSV file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, matched_set in predictions_dict.items():
        matched_str = ",".join(sorted(list(matched_set))) if matched_set else ""
        rows.append({s1_id_col: s1_id, matched_col: matched_str})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(path, sep="\t", index=False)
    logger.info(f"Saved matching results to: {path}")


def save_model_artifact(model_wrapper: Any, path: Union[str, Path]) -> None:
    """Saves classifier model artifact via model wrapper or joblib."""
    if hasattr(model_wrapper, "save"):
        model_wrapper.save(path)
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_wrapper, path)
        logger.info(f"Model artifact saved to: {path}")


def load_model_artifact(path: Union[str, Path]) -> Any:
    """Loads classifier model artifact from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing at: {path}")
    return joblib.load(path)


def save_threshold_artifact(threshold_data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves optimal threshold optimization results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Threshold artifact saved to: {path}")


def load_threshold_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads optimal threshold artifact JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Threshold artifact missing at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
