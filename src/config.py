"""
Configuration settings, file paths, and hyperparameters for Entity Resolution.

Data location (first match wins):
  1. $ER_DATA_DIR environment variable
  2. <repo>/student_resource/dataset
  3. <repo>/../student_resource/dataset   (dataset unzipped next to the repo)
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_data_dir() -> Path:
    env = os.environ.get("ER_DATA_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        PROJECT_ROOT / "student_resource" / "dataset",
        PROJECT_ROOT.parent / "student_resource" / "dataset",
    ]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    return candidates[0]


DATA_DIR = _find_data_dir()
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Working data (gitignored): folds, dev sample, normalised parquet, caches
WORK_DIR = PROJECT_ROOT / "data"
NORM_DIR = WORK_DIR / "norm"
FOLDS_FILE = WORK_DIR / "folds.csv"
DEV_IDS_FILE = WORK_DIR / "dev_s1_ids.csv"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

# Training Data Files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test Data Files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Pipeline Parameters
RANDOM_SEED = 42
N_FOLDS = 5
DEV_SAMPLE_SIZE = 100_000
VALIDATION_SPLIT = 0.2
BETA = 0.5  # Precision-heavy F_beta metric
