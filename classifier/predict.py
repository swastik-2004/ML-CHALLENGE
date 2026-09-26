"""
Classifier Inference Pipeline.

Loads trained model and threshold artifacts, scores candidate pairs from pairwise features,
applies the decision threshold and the one-owner rule (each S2/S3 record goes to at most one S1;
an S1 can still match 0, 1 or many records), and writes matching_results.tsv with one row per S1.

  python -m classifier.predict --split test                      # batched, from the blocker cache
  python -m classifier.predict --features-path <feats.parquet>   # prebuilt feature file, unbatched
"""
import logging
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from src.config import TEST_SOURCE1
from src.data import READ_KW

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


def resolve_threshold(threshold_path: Union[str, Path], threshold_override: Optional[float] = None) -> float:
    """Explicit override > optimal_threshold.json > DEFAULT_THRESHOLD."""
    if threshold_override is not None:
        threshold = float(threshold_override)
        logger.info(f"Using explicitly provided threshold: {threshold:.4f}")
    elif Path(threshold_path).exists():
        thresh_artifact = load_threshold_artifact(threshold_path)
        threshold = float(thresh_artifact.get("best_threshold", DEFAULT_THRESHOLD))
        logger.info(f"Loaded optimal threshold from artifact: {threshold:.4f}")
    else:
        threshold = DEFAULT_THRESHOLD
        logger.warning(f"No threshold artifact found. Using default threshold: {threshold:.4f}")
    return threshold


def predict(
    feature_data_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    threshold_path: Optional[Union[str, Path]] = None,
    threshold_override: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    enforce_one_owner: bool = True,
    all_s1_ids: Optional[List[str]] = None,
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
        all_s1_ids: Every Source 1 id that must get a row (the submission needs one row per
                    test S1). S1s with no candidates or no match above the threshold are written
                    with an empty match list. If None, only S1s with a match are written.

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
    threshold = resolve_threshold(t_path, threshold_override)

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

    if all_s1_ids is not None:
        # group_predictions_by_s1 only returns S1s with a match; singletons need an empty row
        grouped_preds = {s1: grouped_preds.get(s1, set()) for s1 in all_s1_ids}

    save_predictions_tsv(grouped_preds, out_path, s1_id_col=S1_ID_COL)

    logger.info(
        f"Inference complete: Processed {len(results_df):,} candidate pairs. "
        f"Matched S1 entities = {sum(1 for s in grouped_preds.values() if s):,}, "
        f"Singletons = {sum(1 for s in grouped_preds.values() if not s):,}."
    )

    return results_df


def predict_from_cache(
    split: str = "test",
    model_path: Optional[Union[str, Path]] = None,
    threshold_path: Optional[Union[str, Path]] = None,
    threshold_override: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
    enforce_one_owner: bool = True,
    all_s1_ids: Optional[List[str]] = None,
    batch_s1: int = 100_000,
    workers: int = 4,
    fresh: bool = False,
) -> pd.DataFrame:
    """
    Batched inference straight from the blocker cache (data/cache/candidate_pairs_{split}.parquet):
    for every batch of `batch_s1` S1 entities, build the pairwise features (same code as training:
    src.features.pair_features + full-split candidate context), score them, and keep only pairs at
    or above the threshold. Memory stays at about one batch of features, so the full test split
    (tens of millions of pairs) fits in RAM.

    Resumable: each batch's kept pairs are saved to data/preds/classifier_{split}_parts/<key>/.
    The key changes with the model file, threshold, batch size and candidate count, so a retrained
    model or a new blocker run never reuses stale results. `fresh=True` recomputes every batch.

    Returns the kept pairs (source1_entity_id, target_entity_id, probability) before one-owner.
    """
    import hashlib
    import json
    import shutil
    import numpy as np
    from src.config import WORK_DIR
    from src.data import save_parquet_atomic
    from src.features.global_context import load_cache_with_context
    from src.features.pair_features import build, to_pair_ids

    m_path = Path(model_path or MODEL_ARTIFACT_PATH)
    t_path = threshold_path or THRESHOLD_ARTIFACT_PATH
    out_path = output_path or PREDICTION_OUTPUT_PATH

    logger.info(f"=== Starting Batched Classifier Inference ({split} blocker cache) ===")
    model = ClassifierModel.load(m_path)
    features = list(model.feature_names_)
    threshold = resolve_threshold(t_path, threshold_override)

    cache = load_cache_with_context(split)          # sorted by s1_int, context on the full split
    s1_sorted = cache.s1_int.values
    uniq = np.unique(s1_sorted)
    edges = [uniq[k] for k in range(0, len(uniq), batch_s1)] + [uniq[-1] + 1] if len(uniq) else []
    n_batches = max(len(edges) - 1, 0)

    manifest = {"model": str(m_path.resolve()), "model_mtime": m_path.stat().st_mtime,
                "threshold": threshold, "batch_s1": batch_s1, "n_pairs": int(len(cache)),
                "features": features}
    key = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:10]
    parts_dir = WORK_DIR / "preds" / f"classifier_{split}_parts" / key
    if fresh and parts_dir.exists():
        shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    logger.info(f"{len(cache):,} candidate pairs, {len(uniq):,} S1 with candidates, "
                f"{n_batches} batches of {batch_s1:,} S1 -> {parts_dir}")

    for b in range(n_batches):
        part = parts_dir / f"part-{b:04d}.parquet"
        if part.exists():
            continue
        lo, hi = np.searchsorted(s1_sorted, [edges[b], edges[b + 1]])
        feats = build(split, to_pair_ids(cache.iloc[lo:hi].reset_index(drop=True)),
                      context=True, workers=workers)
        missing = [c for c in features if c not in feats.columns]
        if missing:
            raise RuntimeError(f"model expects features the pipeline did not produce: {missing}. "
                               "Retrain the classifier on features built by the current code.")
        p = model.predict_proba(feats[features].astype(np.float32))
        keep = p >= threshold
        save_parquet_atomic(pd.DataFrame({S1_ID_COL: feats.s1_id.values[keep],
                                          TARGET_ID_COL: feats.cand_id.values[keep],
                                          "probability": p[keep].astype(np.float32)}), part)
        logger.info(f"batch {b + 1}/{n_batches}: {hi - lo:,} pairs scored, {int(keep.sum()):,} >= {threshold:.2f}")
        del feats

    kept = pd.concat([pd.read_parquet(parts_dir / f"part-{b:04d}.parquet") for b in range(n_batches)],
                     ignore_index=True) if n_batches else pd.DataFrame(
        columns=[S1_ID_COL, TARGET_ID_COL, "probability"])

    grouped_preds = group_predictions_by_s1(
        kept, probability_col="probability", threshold=threshold, s1_col=S1_ID_COL,
        target_id_col=TARGET_ID_COL, enforce_one_owner=enforce_one_owner,
    )
    if all_s1_ids is not None:
        grouped_preds = {s1: grouped_preds.get(s1, set()) for s1 in all_s1_ids}
    save_predictions_tsv(grouped_preds, out_path, s1_id_col=S1_ID_COL)
    logger.info(
        f"Inference complete: {len(cache):,} candidate pairs scored, {len(kept):,} above threshold. "
        f"Matched S1 entities = {sum(1 for s in grouped_preds.values() if s):,}, "
        f"Empty (singleton) = {sum(1 for s in grouped_preds.values() if not s):,}."
    )
    return kept


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Classifier Inference on Candidate Pairs")
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--features-path", type=str, help="Prebuilt pairwise feature file (unbatched)")
    src_grp.add_argument("--split", choices=["train", "test"],
                         help="Batched: build features from data/cache/candidate_pairs_{split}.parquet")
    parser.add_argument("--model-path", type=str, default=str(MODEL_ARTIFACT_PATH))
    parser.add_argument("--threshold-path", type=str, default=str(THRESHOLD_ARTIFACT_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-path", type=str, default=str(PREDICTION_OUTPUT_PATH))
    parser.add_argument("--s1-file", type=str, default=str(TEST_SOURCE1),
                        help="TSV whose entity_id column lists every S1 that needs a row "
                             "(default: test_source1.tsv). Pass '' to write matched S1s only.")
    parser.add_argument("--batch-s1", type=int, default=100_000, help="--split mode: S1s per batch")
    parser.add_argument("--workers", type=int, default=4, help="--split mode: feature workers")
    parser.add_argument("--fresh", action="store_true", help="--split mode: recompute every batch")
    args = parser.parse_args()

    s1_ids = None
    if args.s1_file:
        s1_ids = pd.read_csv(args.s1_file, usecols=["entity_id"], **READ_KW)["entity_id"].tolist()

    if args.split:
        predict_from_cache(
            split=args.split,
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            threshold_override=args.threshold,
            output_path=args.output_path,
            all_s1_ids=s1_ids,
            batch_s1=args.batch_s1,
            workers=args.workers,
            fresh=args.fresh,
        )
    else:
        predict(
            feature_data_path=args.features_path,
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            threshold_override=args.threshold,
            output_path=args.output_path,
            all_s1_ids=s1_ids,
        )
