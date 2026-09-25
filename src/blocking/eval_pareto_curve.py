"""
Evaluate Recall@K and Precision@K trade-off curve on 100k Dev set
to optimize candidate pool size for Amazon's scalability ranking criterion.
"""
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import WORK_DIR, DEV_IDS_FILE, OUTPUT_DIR, PROJECT_ROOT
from src.data import id_to_int

print("Loading Dev IDs and Truth...")
dev_s1_ids = pd.read_csv(DEV_IDS_FILE)["s1_id"].values
dev_s1_int_set = set(id_to_int(dev_s1_ids))

truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
dev_truth = truth[truth["s1_int"].isin(dev_s1_int_set)].copy()
n_dev_truth = len(dev_truth)
print(f"Total dev truth pairs: {n_dev_truth:,}")

print("Loading cached dev candidate pairs from training blocking cache...")
cache_file = WORK_DIR / "cache" / "candidate_pairs_train.parquet"
if not cache_file.exists():
    print(f"Cache {cache_file} not found!")
    sys.exit(1)

cand_df = pd.read_parquet(cache_file)
# Filter to dev
cand_dev = cand_df[cand_df["s1_int"].isin(dev_s1_int_set)].copy()
print(f"Loaded {len(cand_dev):,} candidate pairs for dev entities.")

# Re-compute rank within s1_int if not present
if "rank" not in cand_dev.columns:
    cand_dev.sort_values(
        by=["s1_int", "best_priority", "n_rules"],
        ascending=[True, True, False],
        inplace=True
    )
    cand_dev["rank"] = cand_dev.groupby("s1_int").cumcount() + 1
else:
    cand_dev["rank"] = cand_dev["rank"] + 1

# Evaluate for K in [1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
k_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
k_results = []

print(f"\n{'K (Max Cap)':<12} | {'Recall@K':<12} | {'Total Pairs':<12} | {'Mean Cand/S1':<12} | {'Median Cand':<12} | {'P90 Cand'}")
print("-" * 75)

for k in k_values:
    capped = cand_dev[cand_dev["rank"] <= k]
    # Count matches with truth
    m = dev_truth.merge(capped[["s1_int", "tg_int"]], on=["s1_int", "tg_int"])
    recall_k = len(m) / n_dev_truth
    
    # Candidate pool stats across all 100k dev entities
    counts = capped.groupby("s1_int")["tg_int"].count()
    # Fill in entities with 0 candidates
    total_entities = len(dev_s1_ids)
    mean_cand = len(capped) / total_entities
    
    # Distribution among entities with candidates
    median_cand = int(counts.median())
    p90_cand = int(counts.quantile(0.90))
    
    print(f"{k:<12} | {recall_k*100:>10.2f}% | {len(capped):<12,d} | {mean_cand:>10.2f} | {median_cand:>11} | {p90_cand:>8}")
    k_results.append({
        "k": k,
        "recall_at_k": round(recall_k * 100, 2),
        "total_pairs": len(capped),
        "mean_candidates_per_entity": round(mean_cand, 2),
        "median_candidates": median_cand,
        "p90_candidates": p90_cand
    })

# Save JSON results
out_json = OUTPUT_DIR / "recall_at_k_tradeoff.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(k_results, f, indent=2)
print(f"\nSaved trade-off metrics to {out_json}")

# Plot Pareto Curve: Candidate Pool Size vs Recall@K
fig, ax1 = plt.subplots(figsize=(9, 5))
recalls = [r["recall_at_k"] for r in k_results]
means = [r["mean_candidates_per_entity"] for r in k_results]

color = '#1f77b4'
ax1.set_xlabel('Candidate Pool Size Cap (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall@K (%)', color=color, fontsize=12, fontweight='bold')
line1 = ax1.plot(k_values, recalls, color=color, marker='o', linewidth=2.5, label='Recall@K (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(40, 90)

# Highlight sweet spots
ax1.axvline(15, color='#2ca02c', linestyle='--', alpha=0.7, label='Optimal Pareto Operating Point (K=15)')
ax1.axvline(50, color='#d62728', linestyle=':', alpha=0.7, label='Maximum Allowed Cap (K=50)')

ax2 = ax1.twinx()
color = '#ff7f0e'
ax2.set_ylabel('Mean Candidates / S1 Entity', color=color, fontsize=12, fontweight='bold')
line2 = ax2.plot(k_values, means, color=color, marker='s', linestyle='--', linewidth=2, label='Mean Candidates / Entity')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 20)

plt.title('Pareto Frontier: Candidate Recall vs. Pool Size (Amazon Scalability Criterion)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()

REPORTS_FIG = PROJECT_ROOT / "reports" / "figures"
REPORTS_FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(REPORTS_FIG / "pareto_recall_vs_candidate_size.png", dpi=300)
fig.savefig(OUTPUT_DIR / "pareto_recall_vs_candidate_size.png", dpi=300)
plt.close(fig)
print(f"Saved Pareto figure to {REPORTS_FIG / 'pareto_recall_vs_candidate_size.png'}")
