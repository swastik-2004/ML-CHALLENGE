"""
Entity matching classification model and threshold optimization.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


class EntityMatcher:
    """
    Supervised pairwise matching classifier.
    Learns whether a candidate pair (S1, S2/S3) represents the same business entity.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.model = LogisticRegression(class_weight="balanced", max_iter=500)
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the matcher on labeled candidate pair features."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Output probability of positive match."""
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Output binary decision based on precision-tuned threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray, beta: float = 0.5) -> float:
        """
        Find decision threshold that maximizes F_beta (default beta=0.5).
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
        """
        probs = self.predict_proba(X_val)
        best_thresh = 0.5
        best_score = -1.0

        for thresh in np.arange(0.3, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f_beta = ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall)
            else:
                f_beta = 0.0

            if f_beta > best_score:
                best_score = f_beta
                best_thresh = thresh

        self.threshold = best_thresh
        return best_thresh

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({"model": self.model, "threshold": self.threshold, "features": self.feature_names}, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.threshold = data["threshold"]
        self.feature_names = data["features"]
