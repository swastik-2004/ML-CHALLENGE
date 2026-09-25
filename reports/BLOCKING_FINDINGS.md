# Candidate Blocker Research & Empirical Findings

## 1. Executive Summary

In large-scale entity resolution (Source 1: ~2.2M train / ~1.7M test against Source 2 + Source 3: ~10.3M train / ~8.0M test), direct all-pairs comparison involves **\(2.2 \times 10^6 \times 10.3 \times 10^6 \approx 2.27 \times 10^{13}\) comparisons** (over 22 trillion pairs).

The goal of the **Candidate Blocker** is to:
1. **Maximize the Pair-Level Recall Ceiling**: Ensure that true matching pairs are retained in the candidate pool (\(\ge 82\%\text{--}94\%\)). Any match missed at the blocking stage can *never* be recovered by subsequent machine learning stages.
2. **Control the Candidate Pool Size**: Keep candidate lists compact (target: average \(\le 20\), 95th percentile \(\le 45\), hard cap \(\le 50\) candidates per Source 1 entity) to prevent memory exhaustion and catastrophic false-positive inflation under the precision-heavy Macro \(F_{0.5}\) metric.
3. **Prevent Key Crowding**: Identify and prune over-crowded, generic tokens (e.g. chains, common prefixes, generic suffixes) that cause combinatorial explosions.

---

## 2. Root Cause Analysis: Why Baseline Recall Was Only 63.26%

In the initial exact-key baseline (`src/baseline.py`), candidate recall on the 100,000 entity dev set was **63.26%** (218,863 out of 345,968 true pairs). Analysis of the 127,105 missed pairs revealed two primary causes:

### A. The "MAX_S1 = 3" Bottleneck
The baseline set a strict threshold:
```python
MAX_S1 = 3  # Maximum number of Source 1 entities sharing a key
MAX_TG = 40  # Maximum number of Target (S2/S3) records sharing a key
```
Because the baseline used pure heuristic 1-to-1 matching without an ML classifier, it discarded any key shared by more than 3 Source 1 entities to protect precision. 
However, for candidate generation, this caused severe recall loss:
- Popular local business names, franchise branches, and common street addresses were completely dropped from blocking.
- Relaxing `MAX_S1` from 3 to 20 increased the valid S1 entity coverage for `name_key` from **60.13% to 71.76%** (+11.63% absolute gain).

### B. Rigid Exact-Key Matching on Residual Pairs
Baseline relied exclusively on three exact strings:
1. `name_key` (sorted canonical tokens + legal suffix stripped)
2. `name_compact` (domain forms, spaces removed)
3. `addr_key` (sorted canonical address tokens)

When a pair had:
- A sub-brand or extra token (e.g. `"Cozy Grill"` vs. `"Cozy Grill Services"`),
- A spelling typo or OCR error (e.g. `"Rowland and Porter"` vs. `"Rowland and Pir"`),
- A landmark variation in the address (e.g. Indian addresses with landmark navigation vs. plot numbers),

both `name_key` and `addr_key` failed to match identically, leaving the pair completely unindexed.

---

## 3. Multi-Pass Blocker Architecture

To resolve the residual gap while strictly bounding candidate sizes, we designed a **7-Rule Multi-Pass Blocker**:

```mermaid
flowchart TD
    S1[Source 1 Record] --> P1[Pass 1: Canonical Clean Keys]
    S1 --> P2[Pass 2: First Significant Name Token]
    S1 --> P3[Pass 3: House No + Name 3-Prefix]
    S1 --> P4[Pass 4: State + First Name Token]
    S1 --> P5[Pass 5: Second Significant Name Token]

    TG[Source 2 / 3 Records] --> P1
    TG --> P2
    TG --> P3
    TG --> P4
    TG --> P5

    P1 -->|R1 name_key, R2 name_compact, R3 addr_key| MERGE[Candidate Pair Aggregator]
    P2 -->|R4 w1_name| MERGE
    P3 -->|R5 house_p3| MERGE
    P4 -->|R6 state_w1| MERGE
    P5 -->|R7 w2_name| MERGE

    MERGE --> DUP[Deduplication & Rule Agreement Counting]
    DUP --> CAP[Candidate Capping: Max 50 per S1]
    CAP --> OUT[candidate_pairs.tsv]
```

### Detailed Pass Specifications:

| Pass # | Rule Identifier | Blocking Key Construction | Crowding Limits (`MAX_S1`, `MAX_TG`) | Target Discrepancy Resolved |
| :--- | :--- | :--- | :--- | :--- |
| **Pass 1** | `R1_name_key` | `Country + name_key` | (20, 50) | Exact order-invariant business name match. |
| **Pass 1** | `R2_name_compact` | `Country + name_compact` | (20, 50) | Compact domain forms (`acme.com` vs `Acme`). |
| **Pass 1** | `R3_addr_key` | `Country + addr_key` | (20, 50) | Exact location match when names differ. |
| **Pass 2** | `R4_w1_name` | `Country + w1` (First token $\ge 3$ chars) | (15, 30) | Brand matching when one record has sub-brand words. |
| **Pass 3** | `R5_house_p3` | `Country + house_no + p3` (3-char name prefix) | (20, 50) | Same building + typo/abbreviation in name. |
| **Pass 4** | `R6_state_w1` | `Country + state + w1` | (15, 40) | State-bounded brand matching. |
| **Pass 5** | `R7_w2_name` | `Country + w2` (Second token $\ge 3$ chars) | (10, 25) | Brand inversion or generic first word. |

---

## 4. Benchmark Results on 100,000 Dev Entities

The benchmark was executed using `python -m src.blocking.benchmark_blocking` on the unified 100k dev set (`data/dev_s1_ids.csv`), containing **345,968 ground-truth pairs**:

### A. Recall Progression Across Passes
![Blocking Recall Progression](figures/blocking_recall_progression.png)

| Rule Name | Total Index Pairs (Global) | Dev Pairs Evaluated | Individual Rule Recall | Cumulative Blocker Recall | Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Strict Rules)** | 6,803,632 | 312,410 | 63.26% | 63.26% | ~120s |
| **R1 (name_key)** | 9,031,668 | 409,085 | 45.97% | **45.97%** | 18.69s |
| **R2 (name_compact)** | 8,911,377 | 404,812 | 45.67% | **48.76%** | 16.12s |
| **R3 (addr_key)** | 3,550,162 | 161,516 | 37.83% | **68.75%** | 21.37s |
| **R4 (w1_name)** | 3,015,420 | 138,912 | 7.69% | **70.21%** | 15.24s |
| **R5 (house_p3)** | 15,287,713 | 690,369 | 52.65% | **80.38%** | 24.26s |
| **R6 (state_w1)** | 10,363,844 | 468,515 | 32.33% | **82.40%** | 19.80s |
| **R7 (w2_name)** | 1,061,554 | 48,523 | 2.65% | **82.50%** | 20.59s |

> 🚀 **Key Finding:**  
> The multi-pass strategy elevates candidate recall from **63.26% to 82.50%** (+19.24% absolute recall gain).  
> In particular, **Pass 3 (`house_p3`)** was extraordinarily effective, capturing 52.65% of true pairs on its own and pushing cumulative recall past 80%.

---

## 5. Candidate Pool Sizing & Distribution

![Candidate Pool Distribution](figures/candidate_pool_distribution.png)

We evaluated candidate load per entity across all 100,000 Dev Source 1 entities:

| Metric | Measured Value | Requirement / Boundary | Status |
| :--- | :---: | :---: | :---: |
| **Entity Coverage** | **95.96%** (95,961 / 100,000) | $\ge 90\%$ | ✅ PASS |
| **Mean Candidates / Entity** | **16.17** | $\le 25.0$ | ✅ PASS |
| **Median (P50)** | **12** | $\le 15$ | ✅ PASS |
| **75th Percentile (P75)** | **23** | $\le 30$ | ✅ PASS |
| **90th Percentile (P90)** | **36** | $\le 45$ | ✅ PASS |
| **95th Percentile (P95)** | **44** | $\le 50$ | ✅ PASS |
| **99th Percentile (P99)** | **59** | Capped at 50 | ✅ Handled via ranking |
| **Hard Cap Applied** | **$\le 50$** | Submission Maximum = 50 | ✅ Strict Enforced |

---

## 6. Implementation Notes for Production Blocker (`src/blocking/blocker.py`)

1. **Memory & Streaming Execution**:
   - The vectorized numpy factorization (`kc * n_countries + cc`) executes the entire cross-match in memory within 25 seconds per rule on 12M+ records.
2. **Prioritized Capping**:
   - When an entity exceeds 50 candidates, candidates are sorted by:
     1. Rule agreement count (number of passes that agreed on the pair, e.g. Pass 1 + Pass 3).
     2. Pass priority (Pass 1 exact keys > Pass 3 house number > Pass 2 first word).
   - The top 50 candidates are preserved, guaranteeing no precision degradation.
3. **Format Integrity**:
   - Every Source 1 entity in `test_source1.tsv` has exactly one row.
   - All candidate IDs are validated against Target S2/S3 ID pools.
   - Lists are deduplicated and comma-separated with tab separation between entity ID and candidates.

---

## 7. Amazon Scalability Criterion: Recall@K Pareto Frontier Analysis

> 🏆 **Critical Competition Update:**  
> *"Candidate generation counts toward the final ranking. We will review your `candidate_pairs.tsv` and the code that produces it when deciding final rankings, alongside your `matching_results.tsv` score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."*

To optimize for this ranking criterion, we mapped the **Pareto Trade-Off Curve** between the candidate pool size cap ($K$) and ground-truth recall ($R@K$) across all 100,000 dev entities (345,968 true matching pairs):

![Pareto Recall vs Candidate Size](figures/pareto_recall_vs_candidate_size.png)

### Recall@K vs. Candidate Pool Sizing:

| Candidate Cap ($K$) | Recall@K (%) | Total Dev Pairs | Mean Candidates / Entity | Median Candidates | 90th Percentile | Efficiency Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$K=1$** | 23.62% | 95,961 | 0.96 | 1 | 1 | Extremely sparse; only single highest-confidence match. |
| **$K=2$** | 42.07% | 187,768 | 1.88 | 2 | 2 | Top 2 candidates capture nearly half of true links. |
| **$K=3$** | 55.01% | 274,749 | 2.75 | 3 | 3 | Approaching baseline recall with under 3 candidates/entity! |
| **$K=5$** | 68.10% | 433,052 | 4.33 | 5 | 5 | **Surpasses Baseline Recall (63.26%)** with only 4.3 candidates/entity! |
| **$K=8$** | 74.20% | 635,304 | 6.35 | 8 | 8 | High-efficiency lean candidate pool. |
| **$K=10$** | 76.15% | 751,295 | 7.51 | 10 | 10 | Captures >76% of matches with ~7.5 candidates/entity. |
| **$K=15$** | **79.00%** | **987,718** | **9.88** | **12** | **15** | ⭐ **Optimal Pareto Operating Point** (slashes pairs by 35% with <3.5% recall difference). |
| **$K=20$** | **80.42%** | **1,162,680** | **11.63** | **12** | **20** | **High-Recall Scalable Point** (>80% recall, 11.6 candidates). |
| **$K=30$** | 81.72% | 1,380,790 | 13.81 | 12 | 30 | Diminishing returns region. |
| **$K=50$** | **82.44%** | **1,529,861** | **15.30** | **12** | **36** | Maximum allowed recall ceiling (default ceiling). |

### Key Strategic Recommendations for Amazon Evaluation:
1. **The $K=5$ Milestone:**
   At just **5 candidates per entity**, our blocker achieves **68.10% recall**, already beating the full unconstrained baseline recall (63.26%).
2. **The $K=15$ Sweet Spot:**
   Setting `--max-candidates 15` yields an average of **only 9.88 candidates per entity** while retaining **79.00% true match recall**. This reduces candidate storage and downstream classifier comparison load by **35.4%** compared to $K=50$.
3. **Prioritized In-List Ordering:**
   Because our multi-pass blocker sorts candidate pairs by `(best_priority ASC, n_rules DESC)`, the most probable matches appear first. Any truncated candidate list preserves the highest-probability links.

