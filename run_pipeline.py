"""
One-command runner for the whole pipeline: raw TSVs -> validated submission.

  blocking -> pairwise features -> classifier (LightGBM) -> threshold + one-owner
           -> output/matching_results.tsv

The classifier is trained on the 100k train dev S1s (blocker candidates + features, labeled from
the ground truth), then classifier.predict scores every test candidate pair in batches of 100k S1s
(bounded memory, resumable per batch). Each stage runs as `python -m ...` with the current
interpreter and caches its output under data/ or output/, so a failed or interrupted run can be
resumed with --from <stage>.

  python run_pipeline.py                        # full run -> output/matching_results.tsv + candidate_pairs.tsv
  python run_pipeline.py --list                 # show stage names
  python run_pipeline.py --from classifier_predict  # resume from a stage
  python run_pipeline.py --only validate        # run selected stages only
  python run_pipeline.py --dry-run              # print the commands without running them
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_stages(a):
    """Ordered (name, argv) list. argv is run as [python, *argv] from the repo root."""
    from src.config import TEST_DIR   # honours ER_DATA_DIR, same as the pipeline itself

    # the same cap for train and test: the context features (rank/gap, n candidates) must
    # see the same candidate distribution at train and test time
    cap = ["--max-candidates", str(a.max_candidates)]
    workers = ["--workers", str(a.workers)]
    return [
        ("tests", ["-m", "pytest", "-q"]),
        # --- shared preparation
        ("folds", ["-m", "src.folds"]),
        ("normalize", ["-m", "src.normalize", *workers]),
        # --- train side: learn the classifier + threshold on the dev S1s
        ("idf_train", ["-m", "src.features.idf", "--split", "train"]),
        ("idf_test", ["-m", "src.features.idf", "--split", "test"]),
        ("train_ids", ["-m", "src.train_ids", "--n", str(a.train_s1)]),
        ("blocker_train", ["-m", "src.blocking.blocker", "--split", "train", "--skip-tsv", *cap]),
        ("features_train", ["-m", "src.features.pair_features", "--split", "train",
                            "--s1-ids", "data/train_s1_ids.csv", *workers]),
        ("classifier_train", ["-m", "classifier.train",
                              "--features-path", "data/feats/train_subset.parquet",
                              "--eval-s1-ids", "data/train_s1_ids.csv"]),
        # --- test side: candidates (+ candidate_pairs.tsv from the same cache the classifier
        # scores) -> features + classifier scoring in S1 batches -> matching_results.tsv
        ("blocker_test", ["-m", "src.blocking.blocker", "--split", "test", *cap]),
        ("classifier_predict", ["-m", "classifier.predict", "--split", "test",
                                "--s1-file", str(TEST_DIR / "test_source1.tsv"), *workers]),
        ("validate", ["student_resource/utils/validate_submission.py",
                      "--matching", "output/matching_results.tsv",
                      "--candidate", "output/candidate_pairs.tsv",
                      "--test-dir", str(TEST_DIR)]),
    ]


def main():
    ap = argparse.ArgumentParser(description="Run the entity resolution pipeline end to end.")
    ap.add_argument("--from", dest="start", help="resume from this stage (see --list)")
    ap.add_argument("--only", nargs="+", help="run only these stages")
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest stage")
    ap.add_argument("--workers", type=int, default=6, help="parallel workers for normalise / features")
    ap.add_argument("--max-candidates", type=int, default=20,
                    help="blocker cap per S1, used for both train and test (blocker default: 20)")
    ap.add_argument("--train-s1", type=int, default=300_000,
                    help="S1s to train on (100k dev + extra train S1s); features scale with this")
    ap.add_argument("--list", action="store_true", help="list stage names and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    a = ap.parse_args()

    stages = build_stages(a)
    if a.skip_tests:
        stages = [s for s in stages if s[0] != "tests"]
    names = [n for n, _ in stages]
    if a.list:
        print("\n".join(names))
        return
    if a.start:
        if a.start not in names:
            ap.error(f"unknown stage {a.start!r}; choose from: {', '.join(names)}")
        stages = stages[names.index(a.start):]
    if a.only:
        bad = set(a.only) - set(names)
        if bad:
            ap.error(f"unknown stage(s) {sorted(bad)}; choose from: {', '.join(names)}")
        stages = [s for s in stages if s[0] in a.only]

    t_all = time.time()
    for i, (name, argv) in enumerate(stages, 1):
        cmd = [sys.executable, *argv]
        print(f"\n{'=' * 78}\n[{i}/{len(stages)}] {name}: python {' '.join(argv)}\n{'=' * 78}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"\nFAILED at stage '{name}' (exit code {rc}) after {time.time() - t0:.0f}s.\n"
                  f"Fix the problem, then resume with:  python run_pipeline.py --from {name}", flush=True)
            sys.exit(rc)
        print(f"[{name} done in {time.time() - t0:.0f}s]", flush=True)

    if not a.dry_run:
        print(f"\nPipeline finished in {(time.time() - t_all) / 60:.1f} min.")
        if any(n == "classifier_predict" for n, _ in stages):
            print(f"Submission: {ROOT / 'output' / 'matching_results.tsv'}\n"
                  f"Candidates: {ROOT / 'output' / 'candidate_pairs.tsv'}")


if __name__ == "__main__":
    main()
