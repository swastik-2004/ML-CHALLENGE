"""
End-to-End Pipeline Orchestrator for Business Entity Resolution.
Reproduces data loading -> blocking -> feature extraction -> matching -> output generation.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

from .config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    OUTPUT_DIR, MATCHING_RESULTS_FILE, CANDIDATE_PAIRS_FILE
)
from .preprocessing.cleaner import TextCleaner, AddressNormalizer
from .blocking.blocker import generate_candidates
from .features.features import PairFeatureExtractor
from .models.matcher import EntityMatcher
from .evaluate import evaluate_macro_f05


def run_pipeline(mode: str = "sample", sample_size: int = 1000):
    """
    Executes entity resolution pipeline.
    mode: 'sample' for rapid prototype verification, 'full' for complete test set.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Running Entity Resolution Pipeline in [{mode.upper()}] Mode ===")

    # 1. Load Data
    print("\n[Step 1/5] Loading datasets...")
    if mode == "sample":
        s1 = pd.read_csv(TEST_SOURCE1, sep="\t", nrows=sample_size)
        s2 = pd.read_csv(TEST_SOURCE2, sep="\t", nrows=sample_size * 3)
        s3 = pd.read_csv(TEST_SOURCE3, sep="\t", nrows=sample_size * 3)
    else:
        s1 = pd.read_csv(TEST_SOURCE1, sep="\t")
        s2 = pd.read_csv(TEST_SOURCE2, sep="\t")
        s3 = pd.read_csv(TEST_SOURCE3, sep="\t")

    targets = pd.concat([s2, s3], ignore_index=True)
    print(f"Loaded {len(s1):,} Source 1 entities and {len(targets):,} candidate pool records.")

    # 2. Candidate Generation (Blocking)
    print("\n[Step 2/5] Running candidate blocking...")
    candidates_df = generate_candidates(s1, targets, max_candidates_per_entity=30)
    candidates_df.to_csv(CANDIDATE_PAIRS_FILE, sep="\t", index=False)
    print(f"Saved candidate pairs to: {CANDIDATE_PAIRS_FILE}")

    # 3. Matching & Scoring
    print("\n[Step 3/5] Scoring candidate pairs...")
    target_dict = targets.set_index("entity_id").to_dict("index")
    s1_dict = s1.set_index("entity_id").to_dict("index")

    results = []
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_str = str(row["candidate_entity_ids"])
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]

        s1_record = s1_dict.get(s1_id, {})
        matched_ids = []

        for cand_id in cands:
            cand_record = target_dict.get(cand_id, {})
            feats = PairFeatureExtractor.extract_features(s1_record, cand_record)

            # High precision rule / heuristic score
            # Requires same country and high string similarity
            if feats["same_country"] == 1.0:
                name_sim = max(feats["name_ratio"], feats["name_jaccard"])
                addr_sim = max(feats["addr_ratio"], feats["addr_jaccard"])
                if name_sim >= 0.85 and (addr_sim >= 0.5 or feats["postal_match"] == 1.0):
                    matched_ids.append(cand_id)

        results.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })

    # 4. Save Final Submissions
    print("\n[Step 4/5] Writing output files...")
    matching_df = pd.DataFrame(results)
    matching_df.to_csv(MATCHING_RESULTS_FILE, sep="\t", index=False)
    print(f"Saved matching results to: {MATCHING_RESULTS_FILE}")

    # 5. Validation Check
    print("\n[Step 5/5] Pipeline completed successfully!")
    print(f"Summary:")
    print(f"  - Total Source 1 records: {len(matching_df):,}")
    print(f"  - Matched entities: {(matching_df['matched_entity_ids'] != '').sum():,}")
    print(f"  - Singletons: {(matching_df['matched_entity_ids'] == '').sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    run_pipeline(mode=args.mode, sample_size=args.sample_size)
