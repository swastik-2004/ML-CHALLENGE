"""
Generic Binary Classifier Wrapper for Business Entity Resolution.

Provides a unified scikit-learn compatible interface for tabular gradient boosting models
(LightGBM with sklearn HistGradientBoostingClassifier / RandomForestClassifier fallback).

Does NOT generate mock data or execute training upon import.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import joblib

from .config import MODEL_PARAMS

logger = logging.getLogger(__name__)


def _get_base_classifier(params: Dict[str, Any]) -> Any:
    """
    Instantiates LightGBM classifier if available, falling back to sklearn HistGradientBoosting.

    Args:
        params: Model hyperparameter dictionary.

    Returns:
        Scikit-learn compatible estimator object.
    """
    try:
        from lightgbm import LGBMClassifier
        logger.info("Using LightGBM classifier (LGBMClassifier).")
        return LGBMClassifier(**params)
    except ImportError:
        logger.warning("LightGBM not installed. Falling back to sklearn HistGradientBoostingClassifier.")
        from sklearn.ensemble import HistGradientBoostingClassifier
        # Map compatible hyperparameters
        hist_params = {
            "max_iter": params.get("n_estimators", 300),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_leaf_nodes": params.get("num_leaves", 31),
            "random_state": params.get("random_state", 42),
        }
        return HistGradientBoostingClassifier(**hist_params)


class ClassifierModel:
    """
    Wrapper class managing classifier initialization, training, scoring, and persistence.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the classifier model wrapper.

        Args:
            params: Optional model hyperparameter dictionary. Defaults to config.MODEL_PARAMS.
        """
        self.params: Dict[str, Any] = params if params is not None else MODEL_PARAMS.copy()
        self.model: Optional[Any] = None
        self.feature_names_: Optional[list] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
            X_val: Optional[pd.DataFrame] = None, y_val=None) -> "ClassifierModel":
        """
        Fits the binary classifier model on candidate pair features X and binary labels y.

        Args:
            X: Feature pandas DataFrame (numerical features).
            y: Target binary labels (1 for true match, 0 for non-match).

        Returns:
            Self instance.
        """
        if X is None or len(X) == 0:
            raise ValueError("Training feature set X cannot be empty.")
        if y is None or len(y) == 0:
            raise ValueError("Training target y cannot be empty.")

        self.feature_names_ = list(X.columns)
        self.model = _get_base_classifier(self.params)

        logger.info(f"Training classifier on {len(X):,} candidate pairs and {X.shape[1]} features...")
        if X_val is not None and len(X_val) and type(self.model).__name__ == "LGBMClassifier":
            import lightgbm as lgb
            from .config import EARLY_STOPPING_ROUNDS
            self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                           callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            logger.info(f"Early stopping: best iteration {self.model.best_iteration_}")
        else:
            self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Classifier model training complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts match probabilities for pairwise candidates.

        Args:
            X: Feature pandas DataFrame matching the feature schema used during fit.

        Returns:
            1D numpy array of predicted match probabilities P(match=1).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Classifier model is not fitted yet. Call fit() or load() first.")

        if X is None or len(X) == 0:
            return np.array([])

        probabilities = self.model.predict_proba(X)
        # Return probability column for positive match class (1)
        return probabilities[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary match decision (0 or 1) by applying a decision threshold.

        Args:
            X: Feature pandas DataFrame.
            threshold: Probability decision threshold.

        Returns:
            1D numpy array of binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the fitted model artifact to disk using joblib.

        Args:
            path: Destination file path for model serialization.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names_, "params": self.params}, path)
        logger.info(f"Classifier model saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassifierModel":
        """
        Loads a trained model artifact from disk.

        Args:
            path: Source file path of saved model artifact.

        Returns:
            Loaded ClassifierModel instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        artifact = joblib.load(path)
        instance = cls(params=artifact.get("params"))
        instance.model = artifact["model"]
        instance.feature_names_ = artifact.get("feature_names")
        instance.is_fitted = True
        logger.info(f"Classifier model loaded successfully from: {path}")
        return instance
