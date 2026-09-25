"""
Normalise every source file into data/norm/{split}_s{n}/part-XXXX.parquet (streamed, resumable).

Resumable by design: finished parts are skipped, so the job can run in time-boxed slices
(the laptop workspace kills long calls) and survives interruptions. A _SUCCESS marker is
written when a file is complete. pd.read_parquet(norm_path(split, n)) reads all parts.

Usage:
  python -m src.normalize                              # everything, no time limit
  python -m src.normalize --max-seconds 150            # one time-boxed slice; rerun until DONE
"""
import argparse
import time
from multiprocessing import Pool

import pandas as pd

from .config import NORM_DIR
from .data import READ_KW, source_path
from .preprocessing.normalize import normalize_record

CHUNK = 200_000


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    recs = [normalize_record(n, a) for n, a in zip(df["business_name"].values,
                                                    df["business_address"].values)]
    out = pd.DataFrame.from_records(recs)
    out.insert(0, "country", df["country"].str.strip().values)
    out.insert(0, "entity_id", df["entity_id"].values)
    return out


def norm_path(split: str, source: int):
    return NORM_DIR / f"{split}_s{source}"


def is_done(split: str, source: int) -> bool:
    return (norm_path(split, source) / "_SUCCESS").exists()


def run(split: str, source: int, pool, deadline: float, force: bool = False) -> bool:
    """Process missing parts until done (True) or the deadline passes (False)."""
    import shutil
    d = norm_path(split, source)
    if force and d.exists():
        shutil.rmtree(d, ignore_errors=True)
    elif not force and is_done(split, source):
        return True
    d.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem.split("-")[1]) for p in d.glob("part-*.parquet")}
    start = (max(done) + 1) if done else 0          # parts are written in order
    reader = pd.read_csv(source_path(split, source), chunksize=CHUNK,
                         skiprows=range(1, start * CHUNK + 1), **READ_KW)
    part = start
    for out in pool.imap(normalize_frame, reader):
        out.to_parquet(d / f"part-{part:04d}.parquet", index=False, compression="zstd")
        print(f"  {split} s{source}: part {part} ({(part + 1) * CHUNK:,} rows)", flush=True)
        part += 1
        if time.time() > deadline:
            return False
    (d / "_SUCCESS").write_text(str(part))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--force", action="store_true", help="Force re-normalization from scratch.")
    a = ap.parse_args()
    deadline = time.time() + a.max_seconds
    with Pool(a.workers) as pool:
        for sp in a.splits:
            for s in a.sources:
                if not run(sp, s, pool, deadline, force=a.force):
                    print("PAUSED (time budget) - rerun to continue", flush=True)
                    raise SystemExit(0)
                print(f"DONE {sp} s{s}", flush=True)
    print("ALL DONE", flush=True)
