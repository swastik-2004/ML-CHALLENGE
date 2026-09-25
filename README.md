# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Evaluation Metric: Macro F0.5](https://img.shields.io/badge/Metric-Macro%20F0.5-orange.svg)]()
[![Documentation: Problem Statement](https://img.shields.io/badge/Docs-Problem%20Statement-purple.svg)](PROBLEM_STATEMENT.md)
[![Documentation: EDA Findings](https://img.shields.io/badge/Docs-EDA%20Findings-teal.svg)](reports/EDA_FINDINGS.md)

An end-to-end Machine Learning solution for large-scale **Business Entity Resolution (ER)** across multi-source commercial data.

---

## 📖 Key Documentation Links

- 🎯 **[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)** — **Full deep-dive into the problem statement, business context, core objectives, constraints, data schemas, and the $F_{0.5}$ metric.**
- 📊 **[reports/EDA_FINDINGS.md](reports/EDA_FINDINGS.md)** — **Plain-English exploratory analysis, dataset numbers, and data visualizations.**

---

## 📌 Problem Overview & Objectives

In commercial platforms, business identity data arrives from multiple independent sources without any common unique identifiers.

- **Source 1**: Reference deduplicated business entities.
- **Source 2 & Source 3**: Incoming noisy records with abbreviations, typographical errors, and partial addresses.

**Core Objective**: For every entity in **Source 1**, identify all corresponding records from **Source 2** and **Source 3** that refer to the same real-world business entity.

### Essential Highlights:
- **Variable Cardinality**: Source 1 entities can match zero (5.58% singletons), one, or multiple records (up to 9 records).
- **The "France" Shift**: The training data contains `US` and `India`, whereas the test set introduces **`France`** (15% of test data). The pipeline must remain strictly country-agnostic.
- **Precision-Heavy Metric ($F_{0.5}$)**: False merges are penalized **twice as heavily** as missed links.
- **Scale**: Over **1.7 million test entities** evaluated against **~10 million candidate records** in Sources 2 & 3.

---

## 🗂️ Project Structure

```text
ML_CHALLENGE/
│
├── .gitignore                          # Excludes datasets, PDFs, venvs, cache, checkpoints
├── README.md                           # Main repository guide (this file)
├── PROBLEM_STATEMENT.md                # Comprehensive problem statement & objective breakdown
├── requirements.txt                    # Python dependencies
│
├── data/                               # Dataset documentation and local data references
│   └── README.md
│
├── src/                                # Modular Entity Resolution package
│   ├── __init__.py
│   ├── config.py                       # Project paths, parameters, and constants
│   ├── preprocessing/                  # Text and address normalization
│   │   ├── __init__.py
│   │   └── cleaner.py                  # Abbreviation expansion, postal code extraction
│   ├── blocking/                       # Candidate generation (blocking)
│   │   ├── __init__.py
│   │   └── blocker.py                  # Multi-key rule-based blocker
│   ├── features/                       # Pairwise similarity feature extraction
│   │   ├── __init__.py
│   │   └── features.py                 # Jaccard, character n-grams, Levenshtein, postal check
│   ├── models/                         # Match classification & threshold tuning
│   │   ├── __init__.py
│   │   └── matcher.py                  # Supervised matcher & F_0.5 threshold optimizer
│   ├── evaluate.py                     # Metric calculation (macro F_0.5)
│   └── pipeline.py                     # End-to-end execution script
│
├── notebooks/                          # Interactive exploration & experiments
│   └── 01_eda_and_visualization.ipynb
│
├── reports/                            # Analysis findings, documentation & figures
│   ├── EDA_FINDINGS.md                 # Visual and plain-English EDA report
│   └── figures/                        # Generated data visualization charts
│       ├── match_distribution.png
│       ├── country_distribution.png
│       ├── text_length_analysis.png
│       ├── entity_overlap_summary.png
│       ├── address_token_patterns.png
│       └── summary_stats.json
│
├── output/                             # Generated submission outputs (tab-separated .tsv)
│   └── .gitkeep
│
└── student_resource/                   # Challenge starter resources
    ├── dataset/                        # Raw TSV dataset files (git-ignored)
    ├── utils/
    │   └── validate_submission.py      # Official submission format checker
    ├── Documentation_template.md       # Solution methodology write-up template
    └── README.md                       # Original challenge prompt
```

---

## ⚙️ Environment Setup & Installation

### 1. Create and Activate Virtual Environment

After pushing to your repository, create an isolated virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Quick Verification Run (Sample Mode)
Test the entire end-to-end pipeline locally on a small subset of records:
```bash
python -m src.pipeline --mode sample --sample-size 1000
```

### Full Production Run
Run the full candidate generation and matching pipeline over the entire test set:
```bash
python -m src.pipeline --mode full
```

This generates the two required output files in `output/`:
- `output/matching_results.tsv` — Scored final matches
- `output/candidate_pairs.tsv` — Blocking candidate set

---

## 🧪 Validating Submissions

Validate the outputs against the official competition formatting rules before submission:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

A passing check prints `PASS` (exit code 0).

---

## 📦 Final Submission Package Structure

When creating the final competition zip archive (`<team_name>_submission.zip`):

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv            # Leaderboard submission file
│   └── candidate_pairs.tsv             # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                        # Complete source code
│       ├── README.md                   # Reproduction instructions
│       └── requirements.txt            # Pinned requirements
└── Documentation_template.md           # Completed methodology documentation
```

---

## 📐 Evaluation Metric

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Macro-average**: $F_{0.5}$ is calculated individually per Source 1 entity, then averaged over all Source 1 entities.
- **Precision Weighting**: Precision is weighted $2\times$ over Recall, penalizing false merges more severely than missed matches.
- **Singletons**: Source 1 entities with zero true matches score 1.0 when correctly predicted as empty, and 0.0 if any match is predicted.
