# Business Entity Resolution Challenge — Problem Statement & Objectives

> **Amazon ML Challenge 2026**  
> *A comprehensive guide to understanding the challenge, data sources, objectives, noise characteristics, constraints, and evaluation metric.*

---

## 1. Executive Problem Summary

In large-scale commercial and e-commerce platforms, business identity data originates from multiple disparate and independent sources (e.g., supplier registries, public records, seller registrations, invoices, and logistics feeds). Each source contributes partial, noisy, and fragmented information about real-world businesses. 

Crucially, **these fragments share no common identifiers** (such as universal registration IDs, tax numbers, or phone numbers). The challenge of programmatically determining which records across different sources refer to the exact same real-world business entity is known as **Entity Resolution (ER)** or **Record Linkage**.

The goal of this challenge is to construct an end-to-end Machine Learning pipeline that takes business records from **3 independent sources** with noisy and inconsistent fields, and accurately predicts which records refer to the same real-world business.

---

## 2. Core Objectives

1. **Reference-Based Entity Matching**:
   - **Source 1** serves as the **deduplicated reference source**.
   - Your objective is to query **Source 2** and **Source 3** for every entity in **Source 1** and output all matching record IDs.
   
2. **Handle Variable Match Cardinality (1-to-Many & Singletons)**:
   - A Source 1 entity may match:
     - **Zero records** (called **singletons** — ~5.6% of entities have no counterpart in Sources 2 or 3).
     - **Exactly one record** from Source 2 or Source 3.
     - **Multiple records** across Source 2, Source 3, or both (many entities have 3 to 5 matching records).

3. **High-Precision Candidate Filtering (Blocking)**:
   - Comparing all records pairwise is computationally impossible ($1.7 \times 10^6 \text{ test entities} \times 10^7 \text{ candidates} \approx 1.7 \times 10^{13}$ pairs).
   - The objective is to design a high-recall, high-reduction **blocking stage** that narrows down billions of comparisons to a focused candidate set ($\le 30\text{--}50$ candidates per entity) before classification.

4. **Precision-Heavy Classification**:
   - The competition metric is **$F_{0.5}$**, which weights **precision twice as heavily as recall**.
   - The objective is to eliminate false merges (linking two different businesses together) while capturing true matches.

---

## 3. Data Sources & Schema

All datasets are provided as **tab-separated files (`.tsv`)**. This is intentional because business addresses and ID lists contain commas; reading them with commas would corrupt column parsing.

### 3.1 Record Schema

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has four columns:

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`entity_id`** | String | Unique record identifier with source prefix (`S1-`, `S2-`, `S3-`) | `S1-714132312`, `S2-00047` |
| **`business_name`** | String | Name of the business entity (noisy, abbreviated, typos) | `Tata Consultancy Services Ltd`, `TCS Corp` |
| **`business_address`** | String | Free-form address (partial, landmark-based, reordered) | `Plot 42, Hitech City, Hyderabad, 500081` |
| **`country`** | String | Country label (`US`, `India`, `France`) | `India`, `US`, `France` |

### 3.2 Ground Truth Schema (`train_ground_truth.tsv`)

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **`source1_entity_id`** | String | The ID of the reference Source 1 entity | `S1-00001` |
| **`matched_entity_ids`** | String | Comma-separated list of matching IDs from S2 and S3 (empty if singleton) | `S2-00047,S2-00193,S3-00812` |

---

## 4. Key Data Insights & Noise Patterns to Handle

### 4.1 Name Variations
- **Legal Entity Suffixes**: Inconsistent abbreviations across sources:
  - `Corp` vs. `Corporation`
  - `Pvt Ltd` vs. `Private Limited`
  - `Inc` vs. `Incorporated`
  - `LLC` vs. `Limited Liability Company`
  - `Co.` vs. `Company`
- **Punctuation & Symbols**: Ampersands vs. words (`A & B` vs. `A and B`), hyphens, slashes, and periods.
- **Word Order & Transpositions**: `"National Bank of California"` vs. `"California National Bank"`.
- **Typographical Errors & Transliterations**: Spelling variations, missing vowels, phonetic variations.

### 4.2 Address Variations
- **Street / Way Abbreviations**: `St` vs `Street`, `Rd` vs `Road`, `Ave` vs `Avenue`, `Blvd` vs `Boulevard`, `Hwy` vs `Highway`.
- **Missing Administrative Components**: Missing postal/ZIP codes, missing city or state names.
- **Landmark-Based Addresses (India)**: References like `"Near SBI ATM"`, `"Opposite Railway Station"`, `"Behind Metro Pillar 12"`.
- **Suite / Unit Formats**: `Ste 400`, `Suite #400`, `Fl 4`, `Floor 4`, `Apt 12B`.

### 4.3 The "France" Out-of-Distribution Shift in the Test Set
- **Training Data**: Contains records from only **`US` (59.9%)** and **`India` (40.1%)**.
- **Test Data**: Contains **`India` (46.7%)**, **`US` (38.3%)**, and **`France` (15.0% — 259,452 entities)**.
- **Objective Requirement**: France does **not** exist in training. Your pipeline must **never** hardcode `US` or `India` and must not drop French entities. Preprocessing and feature engineering must handle open-ended international text representations.

---

## 5. Submission Requirements & Format

Each solution must produce **two tab-separated files (`.tsv`)** placed in the `output/` folder:

### 5.1 `matching_results.tsv` (Leaderboard Scored File)
Your final resolved matches. This file drives the public and private leaderboards.

```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Formatting Rules**:
1. Exactly one row for every Source 1 entity in the test set.
2. Leave `matched_entity_ids` empty for singletons (do not write `"None"` or `"NaN"`).
3. No duplicate IDs inside the comma-separated list.
4. Only IDs from Source 2 (`S2-...`) or Source 3 (`S3-...`) that exist in the test set are allowed. Self-matches to `S1` are rejected.

### 5.2 `candidate_pairs.tsv` (Audit & Blocking Analysis File)
The candidate set generated by your blocking stage **before** final classification.

```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

**Formatting Rules**:
- Every matched ID in `matching_results.tsv` should be a subset of `candidate_pairs.tsv`.
- Used to evaluate blocking recall ceiling and reduction ratio.

---

## 6. Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are scored using the **$F_{\beta}$ score with $\beta = 0.5$**:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why $F_{0.5}$ (Precision-Weighted)?
In commercial entity resolution, merging two different companies (a False Positive) damages platform trust and downstream contracts far more than missing a secondary record (a False Negative). Hence, **Precision is weighted $2\times$ over Recall**.

### Scoring Rules:
- **Macro-Averaged**: Calculated per Source 1 entity, then averaged over all entities in the evaluation set:
  $$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}^{(i)}$$
- **Singletons Scoring**:
  - If ground truth has no matches and you predict empty: **Score = 1.0** (full credit).
  - If ground truth has no matches and you predict any match: **Score = 0.0**.

---

## 7. Challenge Rules & Constraints

1. **Permitted Models**: Models must use an open-source license (MIT, Apache 2.0) and have **at most 8 Billion parameters**.
2. **⚠️ Strictly Prohibited: External Data Lookup**:
   - No external APIs, commercial entity resolution tools, geocoding services, or government business registration registries.
   - All predictions must be derived solely from the provided training and test datasets.
3. **Reproducibility**:
   - The top leaderboard submissions are reviewed by organizers. The code must be self-contained and reproducible using the provided codebase.
