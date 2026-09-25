# Data Directory

This directory is intended for local datasets and intermediate processed files.

## Dataset Structure

The raw datasets are provided under `student_resource/dataset/`:

```
student_resource/dataset/
├── train/
│   ├── train_source1.tsv          # Reference Source 1 records (deduplicated)
│   ├── train_source2.tsv          # Source 2 records (noisy fragments)
│   ├── train_source3.tsv          # Source 3 records (noisy fragments)
│   └── train_ground_truth.tsv     # Ground truth matching entity IDs
└── test/
    ├── test_source1.tsv           # Source 1 records to evaluate (includes US, India, France)
    ├── test_source2.tsv           # Source 2 test records
    └── test_source3.tsv           # Source 3 test records
```

> **Note:** Raw `.tsv` datasets are excluded from Git version control via `.gitignore` to keep the repository lightweight.
