import numpy as np

from src.features.global_context import candidate_context


def test_candidate_context_counts_and_ranks():
    # three S1s share record 5, one has record 7, two share record 9
    tg = np.array([5, 5, 5, 7, 9, 9], dtype=np.int64)
    n_rules = np.array([1, 3, 3, 2, 1, 1], dtype=np.int8)
    prio = np.array([1, 1, 1, 2, 3, 1], dtype=np.int8)   # lower priority number = stronger rule
    n_s1, rank = candidate_context(tg, n_rules, prio)
    assert n_s1.tolist() == [3, 3, 3, 1, 2, 2]
    # record 5: rows 1,2 tie (prio 1, 3 rules) -> rank 1; row 0 (prio 1, 1 rule) -> 3
    # record 9: row 5 (prio 1) beats row 4 (prio 3)
    assert rank.tolist() == [3, 1, 1, 1, 2, 1]


def test_candidate_context_empty():
    n_s1, rank = candidate_context(np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, np.int8))
    assert len(n_s1) == 0 and len(rank) == 0
