"""Pair-feature tests on REAL train noise patterns. Run: python -m tests.test_pair_features"""
import math

import pandas as pd

from src.data import id_to_int
from src.features.pair_features import compute_features, add_context_features, to_pair_ids
from src.preprocessing.normalize import normalize_record

RAW = {  # id: (name, address, country)
    "S1-1": ("Pacific Suma LLC", "6500 N Shore Road, Belfair, WA", "US"),
    "S3-1": ("Suma Pacific LLC", "6500 N Shore Rd, Belfair, Washington", "US"),
    "S1-2": ("E+ Plains", "11237 Lanewood Circle, Dallas, TX", "US"),
    "S3-2": ("Deltazeta", "Dallas, Texas, #11237 Lanewood Cir", "US"),
    "S1-3": ("Complete Zeo Inc", "12032 Greenwich Road, Homer Township, OH", "US"),
    "S2-3": ("Inc Corpfilbdte Zeo", "2032 GREENWICH ROAD, HOMER TOWNSHIP, OH", "US"),
    "S1-4": ("Indriya Club Limited", "H.No - 83-231/A/11 S K Nagar, Hyderabad, Telangana", "India"),
    "S2-4": ("indriyaclub.com", "", "India"),
    "S2-5": ("Pacific Suma LLC", "900 Main St, Tacoma, WA", "US"),
}


def _rec():
    rows = []
    for i, (n, a, c) in RAW.items():
        r = normalize_record(n, a)
        r.update(entity_id=i, country=c)
        rows.append(r)
    return pd.DataFrame(rows).set_index("entity_id")


def _feats(pairs):
    p = pd.DataFrame(pairs, columns=["s1_id", "cand_id"])
    return compute_features(p, _rec())


def test_shuffled_name_same_address():
    f = _feats([("S1-1", "S3-1")]).iloc[0]
    assert f.name_key_eq == 1 and f.name_token_set == 1.0
    assert f.addr_key_eq == 1 and f.house_eq == 1 and f.state_agree == 1


def test_address_only_match_and_dropped_digit():
    f = _feats([("S1-2", "S3-2"), ("S1-3", "S2-3")])
    assert f.iloc[0].addr_key_eq == 1 and f.iloc[0].name_ratio < 0.5     # name is noise
    assert f.iloc[1].house_eq == 0 and f.iloc[1].house_suffix == 1       # 12032 vs 2032


def test_domain_and_missing_address_is_nan():
    f = _feats([("S1-4", "S2-4")]).iloc[0]
    assert f.name_compact_eq == 1 and f.cand_is_domain == 1
    assert math.isnan(f.addr_ratio) and math.isnan(f.house_eq) and f.addr_missing == 1


def test_same_name_different_place_conflicts():
    f = _feats([("S1-1", "S2-5")]).iloc[0]
    assert f.name_key_eq == 1 and f.house_eq == 0 and f.addr_token_set < 0.6


def test_context_features():
    p = pd.DataFrame([("S1-1", "S3-1"), ("S1-1", "S2-5")], columns=["s1_id", "cand_id"])
    df = add_context_features(pd.concat([p, compute_features(p, _rec())], axis=1))
    assert list(df.addr_token_set_rank) == [1.0, 2.0] and df.n_cands.tolist() == [2.0, 2.0]


def test_blocker_int_ids_are_converted():
    blk = pd.DataFrame({"s1_int": id_to_int(["S1-925783039"]), "tg_int": id_to_int(["S3-202863386"]),
                        "n_rules": [3], "best_priority": [1]})
    p = to_pair_ids(blk)
    assert p.s1_id.tolist() == ["S1-925783039"] and p.cand_id.tolist() == ["S3-202863386"]
    assert p.n_rules.tolist() == [3]          # blocker metadata passes through as a feature


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
