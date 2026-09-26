"""
Inference for a submission: blocker cache -> pair features (in S1 batches) -> LightGBM ->
decision layer -> output/matching_results.tsv + output/candidate_pairs.tsv -> validator.

  python -m src.predict_test                              # test split, data/models/lgbm_v1.txt
  python -m src.predict_test --workers 10 --batch-s1 150000
  python -m src.predict_test --fresh                      # recompute every batch (new model/features)

Self-check on train S1s the model never saw (scores itself against the ground truth, writes to
data/preds/selfcheck/ instead of output/):
  python -m src.predict_test --split train --s1-ids data/holdout_s1_ids.csv

Needs: data/cache/candidate_pairs_{split}.parquet (python -m src.blocking.blocker --split test
--skip-tsv), data/norm/{split}_s*, and a model + its .json (features, decision) from
src.models.train_lgbm.

Resumable: each batch's probabilities go to data/preds/{split}_parts/<run key>/part-XXXX.parquet.
A rerun skips finished batches. The run key changes with the model file, the feature list, the
batch size and the candidate count, so a retrained model never reuses old probabilities.
Memory stays at roughly one batch of features (~2.6M pairs for 100k S1) plus the (s1, record, p)
triples for the whole split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, TEST_DIR, WORK_DIR, PROJECT_ROOT
from .data import id_to_int, read_source, save_parquet_atomic
from .features.global_context import load_cache_with_context
from .features.pair_features import _ints_to_ids, build, to_pair_ids
from .models.decision import apply_decision, macro_f05

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]
VALIDATOR = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"


# ----------------------------------------------------------------------------- output files
def write_lists(path: Path, header, s1_ids, s1_ints, pair_s1, pair_tg, chunk_rows=5_000_000) -> int:
    """One line per S1 (in s1_ids order), record ids comma-joined, empty when none.
    UTF-8, tab, LF line endings, no quoting, duplicates removed, written atomically.
    Returns the number of S1 lines with an empty list."""
    pair_s1, pair_tg = np.asarray(pair_s1, np.int64), np.asarray(pair_tg, np.int64)
    order = np.lexsort((pair_tg, pair_s1))
    ps, pt = pair_s1[order], pair_tg[order]
    if len(ps):
        keep = np.r_[True, (ps[1:] != ps[:-1]) | (pt[1:] != pt[:-1])]
        ps, pt = ps[keep], pt[keep]
    joined = []
    i = 0
    while i < len(ps):   # chunks end on an S1 boundary so no list is split
        j = min(i + chunk_rows, len(ps))
        if j < len(ps):
            j2 = int(np.searchsorted(ps, ps[j], side="left"))
            j = j2 if j2 > i else int(np.searchsorted(ps, ps[j], side="right"))
        part = pd.DataFrame({"s1": ps[i:j], "tg": _ints_to_ids(pt[i:j])})
        joined.append(part.groupby("s1", sort=False).tg.agg(",".join))
        i = j
    joined = pd.concat(joined) if joined else pd.Series(dtype=object)
    lists = joined.reindex(s1_ints).fillna("").values
    tmp = Path(str(path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for a in range(0, len(s1_ids), 200_000):
            f.write("".join(f"{s}\t{l}\n" for s, l in zip(s1_ids[a:a + 200_000], lists[a:a + 200_000])))
    os.replace(tmp, path)
    return int((lists == "").sum())


# ----------------------------------------------------------------------------- scoring batches
def score_batches(split, cache, booster, features, parts_dir: Path, batch_s1: int, workers: int):
    s1_sorted = cache.s1_int.values
    uniq = np.unique(s1_sorted)
    edges = [uniq[k] for k in range(0, len(uniq), batch_s1)] + [uniq[-1] + 1]
    n = len(edges) - 1
    t0 = time.time()
    done_before = 0
    for b in range(n):
        out = parts_dir / f"part-{b:04d}.parquet"
        if out.exists():
            done_before += 1
            continue
        lo, hi = np.searchsorted(s1_sorted, [edges[b], edges[b + 1]])
        pairs = to_pair_ids(cache.iloc[lo:hi].reset_index(drop=True))
        tb = time.time()
        feats = build(split, pairs, context=True, workers=workers)
        missing = [c for c in features if c not in feats.columns]
        if missing:
            raise RuntimeError(f"model expects features the pipeline did not produce: {missing}. "
                               "Retrain the model on features built by the current code.")
        p = booster.predict(feats[features].to_numpy(np.float32), num_threads=0)
        save_parquet_atomic(pd.DataFrame({"s1_int": id_to_int(feats.s1_id.values),
                                          "tg_int": id_to_int(feats.cand_id.values),
                                          "p": p.astype(np.float32)}), out)
        del feats, pairs
        el = time.time() - t0
        k = b + 1 - done_before
        print(f"batch {b + 1}/{n}: {hi - lo:,} pairs in {time.time() - tb:.0f}s | "
              f"elapsed {el / 60:.1f} min, ETA {el / k * (n - b - 1) / 60:.1f} min", flush=True)
    return n


def load_parts(parts_dir: Path, n: int) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(parts_dir / f"part-{b:04d}.parquet") for b in range(n)],
                     ignore_index=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--model", default=str(WORK_DIR / "models" / "lgbm_v1.txt"))
    ap.add_argument("--s1-ids", help="csv with column s1_id: only these S1 (self-check on train)")
    ap.add_argument("--batch-s1", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-p", type=float, default=1e-4,
                    help="drop pairs below this probability before the decision layer (memory)")
    ap.add_argument("--fresh", action="store_true", help="recompute all batches")
    ap.add_argument("--out-dir", help="default: output/ for test, data/preds/selfcheck/ otherwise")
    a = ap.parse_args()
    t0 = time.time()

    model_path = Path(a.model)
    meta = json.loads(model_path.with_suffix(".json").read_text())
    features, decision = meta["features"], meta["decision"]
    booster = lgb.Booster(model_file=str(model_path))
    print(f"model {model_path.name}: {len(features)} features, decision {decision}, "
          f"dev F0.5 {meta.get('dev_macro_f05')}", flush=True)

    # every S1 of the split must get a line, in file order
    s1 = read_source(a.split, 1, usecols=["entity_id", "country"])
    if a.s1_ids:
        s1 = s1[s1.entity_id.isin(set(pd.read_csv(a.s1_ids, dtype=str).s1_id))].reset_index(drop=True)
    s1_ids, s1_ints = s1.entity_id.values, id_to_int(s1.entity_id.values)

    if a.s1_ids:   # load only the records these S1s touch (context stays exact, memory small)
        import pyarrow as pa
        import pyarrow.dataset as pads
        from .features.global_context import cache_path
        tg = pads.dataset(cache_path(a.split), format="parquet").to_table(
            columns=["tg_int"], filter=pads.field("s1_int").isin(pa.array(s1_ints, type=pa.int64()))
        ).column("tg_int").to_numpy()
        cache = load_cache_with_context(a.split, tg_subset=tg)
        cache = cache[cache.s1_int.isin(set(s1_ints))].reset_index(drop=True)
    else:
        cache = load_cache_with_context(a.split)

    # batch results live in a folder keyed by model + layout: a retrained model or a new blocker
    # run automatically starts a new folder, so stale probabilities are never mixed in
    tag = f"{a.split}_{Path(a.s1_ids).stem}" if a.s1_ids else a.split
    manifest = {"model": str(model_path.resolve()), "model_mtime": model_path.stat().st_mtime,
                "batch_s1": a.batch_s1, "n_pairs": int(len(cache)), "features": features}
    key = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:10]
    parts_dir = WORK_DIR / "preds" / f"{tag}_parts" / key
    if a.fresh and parts_dir.exists():
        try:
            shutil.rmtree(parts_dir)
        except OSError:
            parts_dir = parts_dir.with_name(f"{key}_{int(time.time())}")
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"batch results: {parts_dir}  (older folders next to it can be deleted)", flush=True)

    n = score_batches(a.split, cache, booster, features, parts_dir, a.batch_s1, a.workers)
    d = load_parts(parts_dir, n)
    n_all = len(d)
    d = d[d.p >= a.min_p].reset_index(drop=True)
    sel = apply_decision(d, decision)
    print(f"scored {n_all:,} pairs; {len(d):,} with p >= {a.min_p:g} go to the decision layer; "
          f"{len(sel):,} matches selected", flush=True)

    out_dir = Path(a.out_dir) if a.out_dir else (OUTPUT_DIR if a.split == "test" and not a.s1_ids
                                                  else WORK_DIR / "preds" / "selfcheck")
    m_path, c_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    n_empty = write_lists(m_path, MATCH_HEADER, s1_ids, s1_ints, sel.s1_int.values, sel.tg_int.values)
    n_nocand = write_lists(c_path, CAND_HEADER, s1_ids, s1_ints, cache.s1_int.values, cache.tg_int.values)
    print(f"wrote {m_path} ({len(s1_ids):,} S1, {n_empty:,} predicted empty) and {c_path} "
          f"({n_nocand:,} S1 without candidates)", flush=True)

    # per-country sanity table (France has no labels: its rates should look like US/India)
    npred = sel.groupby("s1_int").size().reindex(s1_ints, fill_value=0).values
    ncand = cache.groupby("s1_int").size().reindex(s1_ints, fill_value=0).values
    t = pd.DataFrame({"country": s1.country.values, "npred": npred, "ncand": ncand})
    print(t.groupby("country").agg(n_s1=("npred", "size"), mean_candidates=("ncand", "mean"),
                                   no_candidates=("ncand", lambda x: (x == 0).mean()),
                                   predicted_empty=("npred", lambda x: (x == 0).mean()),
                                   mean_matches=("npred", "mean")).round(3).to_string())

    if a.split == "train":   # self-check: score against the ground truth
        truth = pd.read_parquet(WORK_DIR / "cache" / "truth_train.parquet")
        truth = truth[truth.s1_int.isin(set(s1_ints))]
        n_true = truth.groupby("s1_int").size().reindex(s1_ints, fill_value=0).values
        sel = sel.merge(truth.assign(y=1), on=["s1_int", "tg_int"], how="left").fillna({"y": 0})
        f, _ = macro_f05(sel, s1_ints, n_true)
        tp_c = cache.merge(truth.assign(y=1), on=["s1_int", "tg_int"]).groupby("s1_int").size()
        tp_c = tp_c.reindex(s1_ints, fill_value=0).values
        rc = np.where(n_true > 0, tp_c / np.maximum(n_true, 1), 0)
        oracle = np.where(n_true == 0, 1.0, np.where(tp_c > 0, 1.25 * rc / (0.25 + rc), 0.0))
        print(f"\nSELF-CHECK macro F0.5 = {f.mean():.4f}   (perfect model on these candidates: "
              f"{oracle.mean():.4f}) | pair precision {sel.y.mean():.4f}, recall {sel.y.sum() / n_true.sum():.4f}")
        for cn in sorted(set(s1.country)):
            k = s1.country.values == cn
            print(f"  {cn}: {f[k].mean():.4f} (ceiling {oracle[k].mean():.4f}, n={k.sum():,})")
    elif not a.s1_ids and VALIDATOR.exists():
        print("\nrunning the official validator ...", flush=True)
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(m_path),
                            "--candidate", str(c_path), "--test-dir", str(TEST_DIR)])
        print("validator exit code", r.returncode, "(0 = PASS)")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
