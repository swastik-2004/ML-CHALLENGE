"""Scorer tests. Run: python -m tests.test_evaluate  (or pytest)."""
import pandas as pd
from src.evaluate import compute_entity_f_beta, evaluate_macro_f05


def test_statement_example():
    # Problem statement: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    f = compute_entity_f_beta({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert round(f, 3) == 0.714, f


def test_singletons():
    assert compute_entity_f_beta(set(), set()) == 1.0
    assert compute_entity_f_beta({"S2-1"}, set()) == 0.0


def test_empty_prediction_on_matched_entity_is_zero():
    assert compute_entity_f_beta(set(), {"S2-1"}) == 0.0


def test_no_overlap_and_perfect():
    assert compute_entity_f_beta({"S2-9"}, {"S2-1"}) == 0.0
    assert compute_entity_f_beta({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0


def test_recall_only_half():
    # P=1, R=0.5 -> 1.25*0.5/(0.25+0.5)=0.8333
    assert round(compute_entity_f_beta({"S2-1"}, {"S2-1", "S3-2"}), 4) == 0.8333


def test_macro_average_and_missing_rows():
    gt = pd.DataFrame({"source1_entity_id": ["A", "B", "C"],
                       "matched_entity_ids": ["S2-1,S3-2", "", "S2-5"]})
    pred = pd.DataFrame({"source1_entity_id": ["A", "B"],          # C missing -> scored as empty -> 0
                         "matched_entity_ids": ["S2-1,S3-2", ""]})
    r = evaluate_macro_f05(pred, gt)
    assert abs(r["macro_f05"] - (1.0 + 1.0 + 0.0) / 3) < 1e-9, r
    assert r["total_evaluated_entities"] == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
