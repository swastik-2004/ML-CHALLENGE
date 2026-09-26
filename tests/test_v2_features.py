from src.features.pair_features import _near_number, name_skeleton


def test_name_skeleton_transliteration():
    assert name_skeleton("sky technology") == name_skeleton("skai teknoloji") == "sk tknlg"
    assert name_skeleton("shri kirti systems") == name_skeleton("kirti systems")
    assert name_skeleton("") == ""


def test_near_number_digit_damage():
    assert _near_number("8162", "162") and _near_number("40800", "4080") and _near_number("649", "644")
    assert not _near_number("123", "456") and not _near_number("12", "13")
