"""Unit tests for Candidate Blocker. Run: python -m tests.test_blocker (or pytest)."""
import numpy as np
import pandas as pd
from src.blocking.blocker import extract_blocking_tokens, index_rule, MultiPassBlocker
from src.data import id_to_int, int_to_id


def test_id_roundtrip():
    test_ids = ["S1-100", "S2-987654321", "S3-555"]
    ints = id_to_int(test_ids)
    recovered = int_to_id(ints)
    assert recovered == test_ids, f"Mismatch: {recovered} vs {test_ids}"


def test_extract_blocking_tokens():
    names = pd.Series(["Starbucks Coffee Inc", "The Cozy Grill", "A B C", ""])
    w1, w2, p3 = extract_blocking_tokens(names)
    assert w1[0] == "starbucks"
    assert w2[0] == "coffee"
    assert p3[0] == "sta"

    # 'The' is a stop word, so w1 should be 'cozy'
    assert w1[1] == "cozy"
    assert w2[1] == "grill"
    assert p3[1] == "coz"


def test_index_rule_matching():
    s1_vals = np.array(["apple", "banana", "cherry", ""])
    tg_vals = np.array(["apple", "apple", "banana", ""])
    both_country = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    n1 = len(s1_vals)

    pairs = index_rule(s1_vals, tg_vals, both_country, n1, max_s1=5, max_tg=5)
    # 'apple' in s1 (idx 0) matches 'apple' in tg (idx 0 and 1)
    # 'banana' in s1 (idx 1) matches 'banana' in tg (idx 2)
    # Empty string should NOT match
    assert len(pairs) == 3
    assert set(pairs["s1"].unique()) == {0, 1}


def test_crowding_limit():
    # If key appears more than max_s1 times, it must be excluded
    s1_vals = np.array(["common", "common", "common", "unique"])
    tg_vals = np.array(["common", "unique"])
    both_country = np.zeros(6, dtype=int)
    n1 = len(s1_vals)

    # max_s1 = 2 -> 'common' appears 3 times in s1, so it should be skipped
    pairs = index_rule(s1_vals, tg_vals, both_country, n1, max_s1=2, max_tg=5)
    assert len(pairs) == 1
    assert pairs.iloc[0]["s1"] == 3  # only 'unique' matched


def test_crowded_key_is_narrowed_not_dropped():
    # 'common' is shared by 3 S1 rows (> max_s1=2): without narrowing it is dropped entirely;
    # narrowed by a discriminator (e.g. a rare address token) the specific sub-keys still match
    s1_vals = np.array(["common", "common", "common"])
    tg_vals = np.array(["common", "common"])
    s1_disc = np.array(["glitterati", "makhmalabad", ""])
    tg_disc = np.array(["glitterati", "creekedge"])
    both_country = np.zeros(5, dtype=int)
    plain = index_rule(s1_vals, tg_vals, both_country, 3, max_s1=2, max_tg=5)
    assert len(plain) == 0
    narrowed = index_rule(s1_vals, tg_vals, both_country, 3, max_s1=2, max_tg=5,
                          narrow=[(s1_disc, tg_disc)])
    assert list(zip(narrowed.s1, narrowed.tg)) == [(0, 0)]   # only 'common|glitterati' matches
    assert narrowed.blk.tolist() == [1]


def test_soundex_and_rare_tokens():
    from src.blocking.blocker import soundex, rare_address_tokens
    assert soundex("shiva") == soundex("shiv") == "s100"
    assert soundex("") == ""
    addr = np.array(["12 main road glitterati pune", "4 main road glitterati pune",
                     "9 main road pune", "main road pune"], dtype=object)
    r1, r2 = rare_address_tokens(addr)
    assert r1[0] == "glitterati" and r1[1] == "glitterati"     # rarest token seen >= 2 times
    assert r2[0] == "glitterati main"
    assert r1[3] == "main"                                       # ties broken alphabetically


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
