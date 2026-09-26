# Setup & Run Guide

How to go from a fresh clone to a validated submission (`output/matching_results.tsv` + `output/candidate_pairs.tsv`).

For *what* the problem is, see [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md). For *why* each stage looks the way it does, see [reports/EDA_FINDINGS.md](reports/EDA_FINDINGS.md), [reports/BLOCKING_FINDINGS.md](reports/BLOCKING_FINDINGS.md) and [reports/CLASSIFIER_ARCHITECTURE_AND_UPDATES.md](reports/CLASSIFIER_ARCHITECTURE_AND_UPDATES.md). This file covers only how to run it.

---

## 1. Architecture: test data → `matching_results.tsv`

```text
                 TRAIN (learn the model)                          TEST (make the submission)
                 ───────────────────────                          ──────────────────────────
 dataset/train/*.tsv                                   dataset/test/*.tsv
        │  src.normalize                                        │  src.normalize
        ▼                                                       ▼
 data/norm/train_s{1,2,3}                               data/norm/test_s{1,2,3}
        │  src.blocking.blocker --split train                   │  src.blocking.blocker --split test
        ▼                                                       ▼
 data/cache/candidate_pairs_train.parquet               data/cache/candidate_pairs_test.parquet
        │  + candidate context on the FULL split                │  + candidate context on the FULL split
        │    (src/features/global_context.py)                   │    (src/features/global_context.py)
        │  src.features.pair_features                           │  classifier.predict --split test,       
        │  (100k dev S1s only)                                  │    per batch of 100k S1: pair features
        ▼                                                       │    → P(match); then threshold + one-owner
 data/feats/train_subset.parquet                                │
        │  classifier.train                                     │
        │  (labels from ground truth, S1-grouped split,         │
        │   threshold search for macro F0.5)                    ▼
        ▼                                               output/matching_results.tsv  (one row per test S1)
 output/classifier_model.joblib  ─────────────────────► output/candidate_pairs.tsv   (exactly the scored pairs)
 output/optimal_threshold.json   ─────────────────────►
```

| Stage | Code | What it does |
| :--- | :--- | :--- |
| Normalise | [src/normalize.py](src/normalize.py), [src/preprocessing/normalize.py](src/preprocessing/normalize.py) | Accent folding, Indic transliteration, OCR fixes, legal-suffix split, address canonicalisation. Builds order-invariant `name_key` / `addr_key`, plus `house_no`, `state` and `postal`. |
| Blocking | [src/blocking/blocker.py](src/blocking/blocker.py) | 13 same-country key rules, skips crowded keys, merges pairs with memory-efficient bit-packed dedup, and keeps the top **20** candidates per S1 (default), ranked by rule specificity. |
| Candidate context | [src/features/global_context.py](src/features/global_context.py) | `cand_n_s1` (how many S1s compete for a record) and `cand_rank_blk` (this S1's rank among them by blocker evidence). Both are measured on the **whole** blocker cache of the split, so they mean the same thing in train and test. |
| Pair features | [src/features/pair_features.py](src/features/pair_features.py) | ~40 features per pair: rapidfuzz name/address ratios, Jaro-Winkler, token/3-gram Jaccard, house number, state, postal and landmark agreement, plus per-S1 rank/gap context for `name_token_set`, `addr_token_set` and `name_ratio`. |
| Classifier | [classifier/train.py](classifier/train.py), [classifier/model.py](classifier/model.py) | LightGBM on 39 features, 80/20 split grouped by S1 (no leakage), and a probability threshold grid-searched (0.05–0.95) for validation macro F0.5. |
| Test inference | [classifier/predict.py](classifier/predict.py) (`--split test`) | Builds features from the test blocker cache and scores them **in batches of 100k S1s**. Memory stays at about one batch, and a rerun resumes from the last finished batch. It then applies the tuned threshold and the one-owner rule (each S2/S3 record goes to its highest-probability S1 only) and writes `matching_results.tsv` with one row per test S1. |

`candidate_pairs.tsv` is written by the test blocker from the same cache the classifier scores, so it is exactly the set of pairs the classifier ran on, as the challenge rules require. Only the **test** files feed the two output files. The train files are used only to fit the model and choose the threshold.

---

## 2. Prerequisites

| Requirement | Notes |
| :--- | :--- |
| **Python 3.10+** | Tested on 3.14 (all 30 unit tests pass). |
| **RAM** | 16 GB recommended. The blocker loads all normalised records of a split, and test scoring holds about one 100k-S1 batch of features at a time (lower it with `--batch-s1`). |
| **Disk** | ~2.5 GB of raw TSVs, plus several GB of parquet caches under `data/`. |
| **CPU** | 4–12 cores. Normalisation and feature building run in parallel. |
| **The dataset** | Not in the repo (git-ignored). Download it from the challenge portal. See step 5. |

> **Windows users:** clone into a short path (e.g. `C:\ml\ML-CHALLENGE`). The `jupyter` dependency installs very deep file paths, and on a long base path `pip install` fails with `OSError: [Errno 2] ... Long Path support`. Alternatively [enable long paths](https://pip.pypa.io/warnings/enable-long-paths).

---

## 3. Clone

```bash
git clone https://github.com/swastik-2004/ML-CHALLENGE.git
cd ML-CHALLENGE
git checkout hem_devt
```

**Run every command from the repo root** (the folder containing `src/` and `classifier/`). All stages run as modules (`python -m ...`), and that only works from the root.

---

## 4. Create the environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest          # unit tests (not in requirements.txt)
```

`jupyter` / `ipykernel` are only needed for `notebooks/`. If they cause install problems, remove those two lines from `requirements.txt`.

---

## 5. Add the dataset

```text
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

[src/config.py](src/config.py) uses the first of these that contains both `train/` and `test/`:

1. The `ER_DATA_DIR` environment variable
2. `<repo>/student_resource/dataset`
3. `<repo>/../student_resource/dataset`

```bash
export ER_DATA_DIR=/path/to/dataset          # Linux/macOS
$env:ER_DATA_DIR = "D:\path\to\dataset"      # Windows PowerShell
```

---

## 6. Run everything: one command

```bash
python run_pipeline.py
```

[run_pipeline.py](run_pipeline.py) runs the 9 stages below in order. It stops at the first failure and prints the exact command to resume from there. Every stage caches its output, so a resumed run doesn't redo finished work.

```bash
python run_pipeline.py --list                    # stage names
python run_pipeline.py --from classifier_predict # resume from a stage
python run_pipeline.py --only classifier_predict validate
python run_pipeline.py --skip-tests --workers 8
python run_pipeline.py --max-candidates 15       # smaller candidate sets (train and test together)
python run_pipeline.py --dry-run                 # print the commands only
```

---

## 7. The stages, one by one

Use these to run or debug a single stage by hand. The order matters.

| # | Stage | Command | Output |
| :-: | :--- | :--- | :--- |
| 1 | `tests` | `pytest` | 30 unit tests (no dataset needed) |
| 2 | `folds` | `python -m src.folds` | `data/folds.csv`, `data/dev_s1_ids.csv` (100k dev S1s) |
| 3 | `normalize` | `python -m src.normalize --workers 6` | `data/norm/{train,test}_s{1,2,3}/` |
| 4 | `blocker_train` | `python -m src.blocking.blocker --split train --skip-tsv` | `data/cache/candidate_pairs_train.parquet` |
| 5 | `features_train` | `python -m src.features.pair_features --split train --s1-ids data/dev_s1_ids.csv` | `data/feats/train_subset.parquet` |
| 6 | `classifier_train` | `python -m classifier.train --features-path data/feats/train_subset.parquet` | `output/classifier_model.joblib`, `output/optimal_threshold.json` |
| 7 | `blocker_test` | `python -m src.blocking.blocker --split test` | `data/cache/candidate_pairs_test.parquet`, **`output/candidate_pairs.tsv`** |
| 8 | `classifier_predict` | `python -m classifier.predict --split test` | **`output/matching_results.tsv`**, batch results in `data/preds/classifier_test_parts/` |
| 9 | `validate` | `python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test` | prints `PASS` |

Notes:

- **Normalise (3)** is the slowest preprocessing stage and can be resumed. Rerun the same command to continue. `--max-seconds 600` runs it in time-boxed slices, `--force` redoes it from scratch, and `--splits` / `--sources` limit the scope.
- **Blocker (4, 7)** caps candidates at 20 per S1 by default. The organisers rank smaller candidate sets higher. If you change the cap, use the same value for train and test (`run_pipeline.py --max-candidates N` does this).
- **Train features (5)** load the full train blocker cache, attach the full-split candidate context, and only then keep the 100k dev S1s. This keeps `cand_n_s1` / `cand_rank_blk` comparable with test.
- **Classifier training (6)**:
  - Labels each pair from `train_ground_truth.tsv` (`is_match`) when the feature file has no labels.
  - Splits **all 100k dev S1s** 80/20, including S1s the blocker found no candidates for, trains on the 80%, and searches thresholds on the 20%.
  - Scores validation against the **full ground truth**, so true matches the blocker missed count as misses and S1s without candidates count as empty predictions. The logged validation macro F0.5 is therefore an estimate of the leaderboard score.
  - Also logs, and saves to `optimal_threshold.json`, the candidates-only F0.5 (`macro_f05_candidates_only`) and the blocker's recall ceiling on the validation S1s (`val_recall_ceiling`), so you can see how much the blocker costs.
- **Test inference (8)**:
  - Each finished batch is saved, so an interrupted run resumes where it stopped. `--fresh` recomputes everything.
  - The batch folder is keyed by the model file, threshold and batch settings, so a retrained model never reuses old results.
  - `--batch-s1 50000` lowers peak memory, `--workers N` sets feature parallelism, and `--threshold 0.9` overrides the tuned threshold.

**Upload `output/matching_results.tsv` to the leaderboard.** Both files go in the final zip.

---

## 8. Known limitations

None known in the main pipeline. The blocker stages load all normalised records of a split at once, so on 16 GB machines they are the ones to watch (see Troubleshooting).

---

## 9. Optional extras

| Command | Purpose |
| :--- | :--- |
| `python -m classifier.predict --features-path <feats.parquet>` | Unbatched scoring of a prebuilt feature file (for small or dev-sized sets; writes every `test_source1.tsv` S1, `--s1-file` to change) |
| `python -m src.baseline --stage truth` then `python -m src.blocking.check_recall --max-candidates 20` | Blocker recall and pool size on the dev set → `output/blocking_recall_check.json` |
| `python -m src.baseline ...` (see [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) stage 5) | Exact-key rule baseline (dev macro F0.5 ≈ 0.683), for comparison only. Pass `--only matching` to its `write` stage so it doesn't overwrite `candidate_pairs.tsv`. |
| `python -m src.eda_tasks_1_2_3`, `python -m src.blocking.benchmark_blocking`, `python -m src.blocking.eval_pareto_curve` | Regenerate the EDA and blocking reports and figures |

---

## 10. Where things end up

| Path | What | In git? |
| :--- | :--- | :---: |
| `output/matching_results.tsv` | Leaderboard submission | No |
| `output/candidate_pairs.tsv` | Candidate set (exactly what the classifier scored) | No |
| `output/classifier_model.joblib` | Trained LightGBM model | No |
| `output/optimal_threshold.json` | Chosen threshold + full grid-search metrics | Yes |
| `data/folds.csv`, `data/dev_s1_ids.csv` | Validation split | No |
| `data/norm/`, `data/cache/`, `data/feats/` | Normalised records, candidate caches, train feature table | No |
| `data/preds/classifier_test_parts/<key>/` | Per-batch test pairs above the threshold (resume state) | No |

To start over, delete the `data/` subfolders and rerun `python run_pipeline.py`. If you change `--max-candidates`, rerun from `blocker_train` so the caches match the new cap.

---

## 11. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` / `'classifier'` | Run from the repo root using `python -m ...`. |
| `ImportError: cannot import name 'generate_candidates'` | That's `src/pipeline.py`, an old prototype. Use `run_pipeline.py`. |
| pip `OSError ... Long Path support` (Windows) | Shorter clone path, enable long paths, or drop `jupyter`/`ipykernel` from `requirements.txt`. |
| `FileNotFoundError` for a dataset TSV | Check the layout in section 5 or set `ER_DATA_DIR`. |
| `FileNotFoundError` for `data/...` or `output/classifier_model.joblib` | An earlier stage hasn't run. Use `--from <stage>` with the first missing one. |
| `model expects features the pipeline did not produce` (classifier_predict) | The model was trained on features from older code. Rerun from `features_train`. |
| `MemoryError` in `classifier_predict` | Rerun it with a smaller batch: `python -m classifier.predict --split test --batch-s1 50000`. Finished batches are kept. |
| `MemoryError` in a blocker stage | Lower the cap and rebuild both sides: `python run_pipeline.py --from blocker_train --max-candidates 15`. |
