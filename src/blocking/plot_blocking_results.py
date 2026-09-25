"""
Generate publication-quality charts for the Candidate Blocker findings.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..config import OUTPUT_DIR, PROJECT_ROOT

REPORTS_FIG = PROJECT_ROOT / "reports" / "figures"
REPORTS_FIG.mkdir(parents=True, exist_ok=True)

results_file = OUTPUT_DIR / "blocking_benchmark_results.json"
with open(results_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rules = [r["rule"].replace("_", " ").title() for r in data["rule_metrics"]]
rule_recalls = [r["rule_recall"] for r in data["rule_metrics"]]
cum_recalls = [r["cumulative_recall"] for r in data["rule_metrics"]]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Plot 1: Cumulative Recall Progression
fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(rules))
width = 0.35

rects1 = ax1.bar(x - width/2, rule_recalls, width, label='Individual Rule Recall (%)', color='#4A90E2', alpha=0.85)
rects2 = ax1.bar(x + width/2, cum_recalls, width, label='Cumulative Blocker Recall (%)', color='#50E3C2', alpha=0.9)

ax1.axhline(63.26, color='#E74C3C', linestyle='--', linewidth=1.5, label='Baseline Rule Recall (63.26%)')
ax1.set_ylabel('Recall Ceiling (%)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Pass Blocker: Recall Progression on 100k Dev Set', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(rules, rotation=25, ha='right', fontsize=10)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

for rect in rects2:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(REPORTS_FIG / "blocking_recall_progression.png", dpi=300)
fig.savefig(OUTPUT_DIR / "blocking_recall_progression.png", dpi=300)
plt.close(fig)
print(f"Saved {REPORTS_FIG / 'blocking_recall_progression.png'}")

# Plot 2: Candidate Pool Sizing Metrics
stats = data["pool_statistics"]
fig, ax2 = plt.subplots(figsize=(8, 4.5))
percentiles = ['Median (P50)', 'P75', 'P90', 'P95', 'P99', 'Max Allowed']
values = [stats['median_candidates'], stats['p75_candidates'], stats['p90_candidates'], stats['p95_candidates'], stats['p99_candidates'], 50]
colors = ['#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C', '#34495E']

bars = ax2.bar(percentiles, values, color=colors, alpha=0.85, width=0.55)
ax2.axhline(50, color='#E74C3C', linestyle=':', linewidth=1.5, label='Max Submission Cap (50)')
ax2.set_ylabel('Candidates per Entity', fontsize=12, fontweight='bold')
ax2.set_title(f"Candidate Pool Distribution (Mean: {stats['mean_candidates_per_entity']:.1f}, Coverage: {stats['coverage_pct']}%)", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(0, 65)

for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(REPORTS_FIG / "candidate_pool_distribution.png", dpi=300)
fig.savefig(OUTPUT_DIR / "candidate_pool_distribution.png", dpi=300)
plt.close(fig)
print(f"Saved {REPORTS_FIG / 'candidate_pool_distribution.png'}")
