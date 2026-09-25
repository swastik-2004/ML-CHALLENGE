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
