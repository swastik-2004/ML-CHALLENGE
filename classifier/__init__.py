"""
Classifier Module for Business Entity Resolution (Amazon ML Challenge 2026).

This module provides the tabular machine learning classification stage of the pipeline.
It consumes numerical pairwise similarity features produced downstream of blocking,
trains a high-precision binary classifier, optimizes probability decision thresholds
for Macro F0.5 maximization, and formats submission outputs.
"""
from .config import (
    FEATURE_COLUMNS,
    REQUIRED_ID_COLUMNS,
    MODEL_PARAMS,
    S1_ID_COL,
    TARGET_ID_COL,
    TARGET_SOURCE_COL,
    LABEL_COL,
)
from .feature_schema import validate_features, get_feature_columns, FeatureSchemaError
from .model import ClassifierModel
from .evaluate import evaluate_macro_f05, compute_s1_entity_f05
from .threshold import optimize_threshold
from .train import train
from .predict import predict

__all__ = [
    "ClassifierModel",
    "validate_features",
    "get_feature_columns",
    "FeatureSchemaError",
    "evaluate_macro_f05",
    "compute_s1_entity_f05",
    "optimize_threshold",
    "train",
    "predict",
    "FEATURE_COLUMNS",
    "REQUIRED_ID_COLUMNS",
    "MODEL_PARAMS",
    "S1_ID_COL",
    "TARGET_ID_COL",
    "TARGET_SOURCE_COL",
    "LABEL_COL",
]
