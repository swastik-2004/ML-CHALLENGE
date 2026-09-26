import numpy as np

from src.data import id_to_int
from src.predict_test import MATCH_HEADER, write_lists


def test_write_lists_every_s1_once_lf_no_duplicates(tmp_path):
    s1_ids = np.array(["S1-30", "S1-10", "S1-20"])          # file order must be kept
    s1_ints = id_to_int(s1_ids)
    ps = id_to_int(["S1-10", "S1-10", "S1-30", "S1-10"])
    pt = id_to_int(["S3-5", "S2-7", "S2-9", "S3-5"])          # duplicate pair (S1-10, S3-5)
    out = tmp_path / "m.tsv"
    n_empty = write_lists(out, MATCH_HEADER, s1_ids, s1_ints, ps, pt, chunk_rows=2)
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.decode().split("\n") == [
        "source1_entity_id\tmatched_entity_ids",
        "S1-30\tS2-9",
        "S1-10\tS2-7,S3-5",
        "S1-20\t",
        "",
    ]
    assert n_empty == 1
