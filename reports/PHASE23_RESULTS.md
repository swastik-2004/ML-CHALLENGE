# Phase 2 + 3 results (26 Sep)

Measured on the **same 30,000 dev S1s** (candidates from the 13-rule blocker), 3 folds grouped by S1,
out-of-fold predictions, one-owner rule, best decision rule per setup, macro F0.5 against the
**full** ground truth (blocker misses count, like the leaderboard).

| Setup | Macro F0.5 | India | US |
|---|---|---|---|
| A: hem_devt model (300 trees, depth 6, 31 leaves) + 40 features | 0.9018 | 0.8436 | 0.9414 |
| D: A + 9 v2 features | 0.9048 | 0.8474 | 0.9439 |
| B: new model settings, old features | 0.9054 | 0.8473 | 0.9450 |
| **C: new settings + v2 features** | **0.9088** | **0.8521** | **0.9475** |
| C trained on half the S1s | 0.9065 | 0.8493 | 0.9455 |

- A's best threshold was 0.66, the same one the team found, so the test agrees with the real pipeline.
- More training S1s: about +0.23 pt per doubling, so 80k → 300k should add about +0.4.

## Phase 2: v2 features (`src/features/pair_features.py`, `src/features/idf.py`)
Needs the rarity tables `data/cache/idf_{split}.parquet` (`python -m src.features.idf --split train|test`).

| Feature | What it measures |
|---|---|
| `addr_idf_jacc` | address token overlap weighted by rarity (per country, counted over S2+S3) |
| `addr_rare_shared` | shared address words appearing in ≤ 50 records ("tarabanahalli", "vaniawad") |
| `addr_idf_max_shared` | rarity of the rarest shared address word |
| `name_idf_jacc`, `name_rare_shared` | the same for name tokens ("vijay" counts, "private" doesn't) |
| `house_in_numbers` | house number found anywhere among the other side's numbers, allowing one damaged digit |
| `numbers_near_frac` | share of numbers with ≥ 3 digits that have a match allowing one dropped or changed digit (8162/162) |
| `name_skel_ratio`, `name_skel_eq` | consonant-skeleton similarity: "sky technology" = "skai teknoloji" = `sk tknlg` |

## Phase 3: model (`classifier/config.py`, `classifier/model.py`, `classifier/train.py`)
- 127 leaves, `min_child_samples` 100, learning rate 0.05, early stopping (50 rounds) on the 20% validation S1s; typically stops at 200–400 trees.
- `subsample_freq=1`: before this, `subsample=0.8` did nothing, because LightGBM ignores `subsample` without it.
- Training S1s: 100k dev + 200k more (`python -m src.train_ids`, `run_pipeline.py --train-s1`).

Reproduce: `data/preds/exp/run.py` (configs A/B/C/D, `run.py eval ...`).

## Checked and NOT adopted (26 Sep, same 30k S1s, settings chosen on 2 folds and scored on the 3rd)
| Option | Macro F0.5 | vs one threshold |
|---|---|---|
| One threshold (current pipeline) | 0.9080 | — |
| Separate threshold for India / US | 0.9080 | ±0 |
| Expected-F0.5 set selection instead of a threshold | 0.9087 | +0.07 (within noise) |
| Average of 3 seeds, one threshold | 0.9084 | +0.04 for 3× training time |

- The threshold the new model picks is about 0.72–0.76, not 0.66. `classifier.train` searches it automatically, so nothing to change.
- None of these is worth the extra code or training time. The pipeline stays as committed.
