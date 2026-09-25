"""
Safe, memory-aware loaders for the challenge TSV files.

Every file is read as plain strings:
  - sep="\t"                 addresses and id lists contain commas
  - dtype=str                ids and postal codes must not become numbers
  - keep_default_na=False    an empty name/address stays "", never NaN / "nan"
  - quoting=QUOTE_NONE       a stray double quote must not swallow tabs or rows
Load one file at a time and only the columns you need.
"""
import csv
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from .config import TRAIN_DIR, TEST_DIR, TRAIN_GROUND_TRUTH

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False,
               quoting=csv.QUOTE_NONE, encoding="utf-8")

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def source_path(split: str, source: int):
    """Path of <split>_source<n>.tsv for split in {'train','test'}, source in {1,2,3}."""
    if split not in ("train", "test") or source not in (1, 2, 3):
        raise ValueError(f"bad split/source: {split!r}/{source!r}")
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return base / f"{split}_source{source}.tsv"


def read_source(split: str, source: int, usecols: Optional[List[str]] = None,
                nrows: Optional[int] = None) -> pd.DataFrame:
    """Read one source file as strings. Pass usecols to save memory."""
    df = pd.read_csv(source_path(split, source), usecols=usecols, nrows=nrows, **READ_KW)
    missing = set(usecols or SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{split} source{source} missing columns {missing}")
    return df


def iter_source(split: str, source: int, chunksize: int = 500_000,
                usecols: Optional[List[str]] = None) -> Iterable[pd.DataFrame]:
    """Stream a source file in chunks (for normalising 5M-row files in 3-4 GB RAM)."""
    return pd.read_csv(source_path(split, source), usecols=usecols,
                       chunksize=chunksize, **READ_KW)


def read_ground_truth() -> pd.DataFrame:
    """train_ground_truth.tsv: source1_entity_id, matched_entity_ids ('' for singletons)."""
    return pd.read_csv(TRAIN_GROUND_TRUTH, **READ_KW)


def parse_id_list(value: str) -> List[str]:
    """'S2-1,S3-2' -> ['S2-1','S3-2']; '' -> []."""
    return [x.strip() for x in value.split(",") if x.strip()] if value else []


def gt_map(gt: Optional[pd.DataFrame] = None) -> Dict[str, Set[str]]:
    """{s1_id: set(matched ids)} for every train S1, singletons included."""
    gt = read_ground_truth() if gt is None else gt
    return {s1: set(parse_id_list(m)) for s1, m in
            zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values)}


def gt_pairs(gt: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Long form of the ground truth: one row per true (s1_id, match_id) pair."""
    gt = read_ground_truth() if gt is None else gt
    long = gt.assign(match_id=gt["matched_entity_ids"].str.split(",")).explode("match_id")
    long = long[long["match_id"].fillna("") != ""]
    return pd.DataFrame({"s1_id": long["source1_entity_id"].values,
                         "match_id": long["match_id"].str.strip().values})


def id_to_int(ids) -> "np.ndarray":
    """'S2-764573417' -> 2*10**11 + 764573417 (int64). Compact, fast joins on 10M+ ids."""
    import numpy as np
    s = pd.Series(ids, dtype="string")
    src = s.str[1].astype("int64").values
    if not np.isin(src, (1, 2, 3)).all():
        raise ValueError("unexpected id prefix (expected S1-/S2-/S3-)")
    return src * 10**11 + s.str[3:].astype("int64").values


def int_to_id(ints):
    """2*10**11 + 764573417 -> 'S2-764573417'."""
    import numpy as np
    if isinstance(ints, (int, np.integer)):
        return f"S{ints // 10**11}-{ints % 10**11}"
    return [f"S{i // 10**11}-{i % 10**11}" for i in ints]


def save_parquet_atomic(df: pd.DataFrame, path) -> None:
    """Write to <path>.tmp then rename, so an interrupted write never looks complete."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
