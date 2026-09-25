"""
Exact-key baseline (P1): high-precision rules, no learning.

Rules, always within the same country string (100% of true train pairs share country):
  R1 name_key     equal   (order/duplicate/legal/accent/OCR-invariant name)
  R2 name_compact equal   (domain forms: 'indriyaclub.com' == 'Indriya Club Ltd')
  R3 addr_key     equal   (reorder-invariant normalised address, non-empty)
Keys shared by more than MAX_S1 S1s or MAX_TG targets are skipped (chains/generic names).
One-owner rule: each S2/S3 record kept under exactly one S1 (most rules agreeing; ties dropped).

Stages (each fits the laptop workspace's 3-minute call limit; all caches written atomically):
  python -m src.baseline --split train --stage ids
  python -m src.baseline --split train --stage rule --rule R1   (then R2, R3)
  python -m src.baseline --split train --stage combine
  python -m src.baseline --split train --stage truth
  python -m src.baseline --split train --stage eval
  ... same ids/rule/combine for --split test, then --stage write
"""
import argparse
import csv
import time

import numpy as np
import pandas as pd

from .config import DEV_IDS_FILE, OUTPUT_DIR, MATCHING_RESULTS_FILE, CANDIDATE_PAIRS_FILE, WORK_DIR
from .data import id_to_int, read_ground_truth, gt_pairs, save_parquet_atomic
from .evaluate import f05_from_counts
from .normalize import norm_path

RULES = {"R1": "name_key", "R2": "name_compact", "R3": "addr_key"}
MAX_S1, MAX_TG = 3, 40
CACHE = WORK_DIR / "cache"


def _load(split, cols):
    s1 = pd.read_parquet(norm_path(split, 1), columns=cols)
    tg = pd.concat([pd.read_parquet(norm_path(split, s), columns=cols) for s in (2, 3)],
                   ignore_index=True)
    return s1, tg


def stage_ids(split: str) -> None:
    """Row position -> int id arrays for S1 and targets (S2 then S3)."""
    s1, tg = _load(split, ["entity_id"])
    save_parquet_atomic(pd.DataFrame({"id": id_to_int(s1.entity_id.values)}), CACHE / f"ids_{split}_s1.parquet")
    save_parquet_atomic(pd.DataFrame({"id": id_to_int(tg.entity_id.values)}), CACHE / f"ids_{split}_tg.parquet")


def stage_rule(split: str, rule: str) -> int:
    """(s1_pos, tg_pos) whose (country, key) are equal and non-empty, skipping crowded keys."""
    col = RULES[rule]
    s1, tg = _load(split, ["country", col])
    n1 = len(s1)
    both = pd.concat([s1, tg], ignore_index=True)
    del s1, tg
    cc, _ = pd.factorize(both["country"])
    kc, _ = pd.factorize(both[col])
    key = kc.astype(np.int64) * (int(cc.max()) + 1) + cc
    key[both[col].values == ""] = -1
    del both
    ka, kb = key[:n1], key[n1:]
    nk = int(key.max()) + 1
    ca = np.bincount(ka[ka >= 0], minlength=nk)
    cb = np.bincount(kb[kb >= 0], minlength=nk)
    ok = (ca >= 1) & (ca <= MAX_S1) & (cb >= 1) & (cb <= MAX_TG)
    ia = np.flatnonzero((ka >= 0) & ok[np.maximum(ka, 0)])
    ib = np.flatnonzero((kb >= 0) & ok[np.maximum(kb, 0)])
    pairs = pd.DataFrame({"s1": ia, "k": ka[ia]}).merge(pd.DataFrame({"tg": ib, "k": kb[ib]}), on="k")
    pairs = pairs[["s1", "tg"]].astype(np.int32)
    save_parquet_atomic(pairs, CACHE / f"pairs_{split}_{rule}.parquet")
    return len(pairs)


def stage_combine(split: str) -> int:
    parts = []
    for r in RULES:
        p = pd.read_parquet(CACHE / f"pairs_{split}_{r}.parquet")
        p[r] = np.int8(1)
        parts.append(p)
    pairs = pd.concat(parts, ignore_index=True)
    for r in RULES:
        pairs[r] = pairs[r].fillna(0).astype(np.int8)
    pairs = pairs.groupby(["s1", "tg"], as_index=False, sort=False)[list(RULES)].max()
    pairs["n_rules"] = pairs[list(RULES)].sum(axis=1).astype(np.int8)
    s1_ids = pd.read_parquet(CACHE / f"ids_{split}_s1.parquet").id.values
    tg_ids = pd.read_parquet(CACHE / f"ids_{split}_tg.parquet").id.values
    pairs["s1_int"], pairs["tg_int"] = s1_ids[pairs.s1.values], tg_ids[pairs.tg.values]
    save_parquet_atomic(pairs, CACHE / f"pairs_{split}.parquet")
    return len(pairs)


def stage_truth() -> int:
    gp = gt_pairs(read_ground_truth())
    t = pd.DataFrame({"s1_int": id_to_int(gp.s1_id.values), "tg_int": id_to_int(gp.match_id.values)})
    save_parquet_atomic(t, CACHE / "truth_train.parquet")
    return len(t)


VARIANTS = ("all", "addr_first", "addr_only")


def apply_variant(pairs: pd.DataFrame, variant: str) -> pd.DataFrame:
    """all: every rule pair | addr_first: if an S1 has any address-exact pair (R3), drop its
    name-only pairs | addr_only: keep address-exact pairs only.
    Motivation (train, 25 Sep): name-only pairs are ~53% precise, address pairs 91-100%."""
    if variant == "all":
        return pairs
    if variant == "addr_only":
        return pairs[pairs.R3 == 1]
    has_addr = pairs.groupby("s1")["R3"].transform("max") == 1
    return pairs[(pairs.R3 == 1) | ~has_addr]


def one_owner(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep each target under the single S1 with the most agreeing rules; drop ties."""
    best = pairs.groupby("tg")["n_rules"].transform("max")
    top = pairs[pairs.n_rules == best]
    return top[top.groupby("tg")["s1"].transform("size") == 1]


def stage_eval() -> None:
    pairs = pd.read_parquet(CACHE / "pairs_train.parquet")
    truth = pd.read_parquet(CACHE / "truth_train.parquet")
    truth["true"] = True
    pairs = pairs.merge(truth, on=["s1_int", "tg_int"], how="left")
    pairs["true"] = pairs["true"].notna()
    print(f"candidate pairs: {len(pairs):,}  | true pairs overall: {len(truth):,}  "
          f"| candidate recall {pairs.true.sum()/len(truth):.1%}")
    print("pair precision by rule combination (before one-owner):")
    print(pairs.groupby(["R1", "R2", "R3"])["true"].agg(n="size", precision="mean").to_string())
    s1_all = pd.read_parquet(CACHE / "ids_train_s1.parquet").id.values
    n_true = truth.groupby("s1_int").size().reindex(s1_all, fill_value=0).values
    best, best_f = None, -1.0
    for v in VARIANTS:
        k = one_owner(apply_variant(pairs, v))
        g = k.groupby("s1_int").agg(n_pred=("tg_int", "size"), tp=("true", "sum")).reindex(s1_all, fill_value=0)
        f = f05_from_counts(n_true, g.n_pred.values, g.tp.values).mean()
        print(f"variant {v:<11} pairs {len(k):>10,}  precision {k.true.mean():.4f}  "
              f"recall {k.true.sum()/len(truth):.1%}  macro F0.5 {f:.4f}")
        if f > best_f:
            best, best_f = v, f
    print(f"best variant: {best}")
    (CACHE / "baseline_variant.txt").write_text(best)
    kept = one_owner(apply_variant(pairs, best))
    g = kept.groupby("s1_int").agg(n_pred=("tg_int", "size"), tp=("true", "sum")).reindex(s1_all, fill_value=0)
    rep = pd.DataFrame({"s1_int": s1_all, "n_true": n_true, "n_pred": g.n_pred.values,
                        "f05": f05_from_counts(n_true, g.n_pred.values, g.tp.values),
                        "country": pd.read_parquet(norm_path("train", 1), columns=["country"]).country.values})
    rep["bucket"] = pd.cut(rep.n_true, [-1, 0, 1, 2, 3, 4, 5, 99],
                           labels=["0 (singleton)", "1", "2", "3", "4", "5", "6+"])
    dev = set(id_to_int(pd.read_csv(DEV_IDS_FILE, dtype=str).s1_id.values))
    fmt = lambda x: f"{x:.4f}"
    for name, r in [("DEV 100k", rep[rep.s1_int.isin(dev)]), ("FULL TRAIN 2.2M", rep)]:
        print(f"\n=== {name}: macro F0.5 = {r.f05.mean():.4f} | predicted-empty {(r.n_pred == 0).mean():.1%} ===")
        print(r.groupby("bucket", observed=True).agg(n=("f05", "size"), f05=("f05", "mean"),
              empty_pred=("n_pred", lambda s: (s == 0).mean())).to_string(float_format=fmt))
        print(r.groupby("country").agg(n=("f05", "size"), f05=("f05", "mean")).to_string(float_format=fmt))
    save_parquet_atomic(rep[["s1_int", "n_true", "n_pred", "f05"]], CACHE / "baseline_scores_train.parquet")


def stage_write(only: str = "both") -> None:
    """Write the leaderboard file and/or the candidate file from cached test pairs."""
    pairs = pd.read_parquet(CACHE / "pairs_test.parquet", columns=["s1", "tg", "R1", "R2", "R3", "n_rules"])
    variant = (CACHE / "baseline_variant.txt").read_text().strip()
    s1_ids = pd.read_parquet(norm_path("test", 1), columns=["entity_id"]).entity_id.values
    tg_ids = np.concatenate([pd.read_parquet(norm_path("test", s), columns=["entity_id"]).entity_id.values
                             for s in (2, 3)])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    if only in ("both", "matching"):
        jobs.append((MATCHING_RESULTS_FILE, "matched_entity_ids", one_owner(apply_variant(pairs, variant))))
        print(f"variant used: {variant}")
    if only in ("both", "candidate"):
        jobs.append((CANDIDATE_PAIRS_FILE, "candidate_entity_ids", pairs))
    for path, col, df in jobs:
        lists = (pd.DataFrame({"s1": df.s1.values, "t": tg_ids[df.tg.values]})
                 .sort_values(["s1", "t"]).groupby("s1", sort=False)["t"].agg(",".join))
        vals = np.full(len(s1_ids), "", dtype=object)
        vals[lists.index.values] = lists.values
        out = pd.DataFrame({"source1_entity_id": s1_ids, col: vals})
        tmp = path.with_name(path.name + ".tmp")
        out.to_csv(tmp, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
        tmp.replace(path)
        print(f"wrote {path.name}: {len(out):,} rows, non-empty {(out[col] != '').mean():.1%}, "
              f"ids {len(df):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--stage", choices=["ids", "rule", "combine", "truth", "eval", "write"], required=True)
    ap.add_argument("--rule", choices=list(RULES))
    ap.add_argument("--only", choices=["both", "matching", "candidate"], default="both")
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "ids":
        stage_ids(a.split)
    elif a.stage == "rule":
        print(f"{a.split} {a.rule}: {stage_rule(a.split, a.rule):,} pairs")
    elif a.stage == "combine":
        print(f"{a.split}: {stage_combine(a.split):,} unique candidate pairs")
    elif a.stage == "truth":
        print(f"truth pairs: {stage_truth():,}")
    elif a.stage == "eval":
        stage_eval()
    else:
        stage_write(a.only)
    print(f"[{a.stage} done in {time.time()-t0:.0f}s]")
