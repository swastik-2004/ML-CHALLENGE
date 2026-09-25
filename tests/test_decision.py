"""Decision-layer tests (no model needed). Run: python -m tests.test_decision"""
import pandas as pd

from src.models.decision import apply_decision, macro_f05, one_owner, select_expected_f


def _d(rows):
    return pd.DataFrame(rows, columns=["s1_int", "tg_int", "p", "y"])


def test_one_owner_keeps_best_s1_per_record():
    d = one_owner(_d([(1, 10, 0.9, 1), (2, 10, 0.4, 0), (2, 11, 0.8, 1)]))
    assert sorted(zip(d.s1_int, d.tg_int)) == [(1, 10), (2, 11)]


def test_expected_f_predicts_nothing_for_weak_candidates():
    sel = select_expected_f(_d([(1, 10, 0.05, 0), (1, 11, 0.04, 0)]), c=0.0, r=1.0)
    assert sel.empty                                     # likely singleton -> empty


def test_expected_f_takes_all_strong_candidates():
    sel = select_expected_f(_d([(1, 10, 0.97, 1), (1, 11, 0.95, 1), (1, 12, 0.10, 0)]), c=0.0, r=1.0)
    assert sorted(sel.tg_int) == [10, 11]


def test_apply_threshold_and_macro_score():
    sel = apply_decision(_d([(1, 10, 0.9, 1), (1, 11, 0.3, 0), (2, 12, 0.2, 0)]), {"method": "threshold", "t": 0.5})
    f, _ = macro_f05(sel, [1, 2, 3], [1, 0, 0])          # S1 3 has no candidates, no matches
    assert list(f) == [1.0, 1.0, 1.0]


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
