"""Normaliser regression tests built from REAL train true-match pairs. Run: python -m tests.test_normalize"""
from src.preprocessing.normalize import normalize_name as N, normalize_address as A


def test_shuffle_legal_and_case():
    k = N("Pacific Suma LLC")["name_key"]
    assert k == N("Suma Pacific LLC")["name_key"] == N("PACIFIC SUMA")["name_key"] == "pacific suma"


def test_ocr_accent_separator_bracket():
    k = N("General Printing Worldwide")["name_key"]
    assert N("6eneral Printing Wórldwide (Co)")["name_key"] == k
    assert N("GENERAL-PRINTING-WORLDWIDE")["name_key"] == k
    assert N("Indore We1lness Private [Limited]")["name_key"] == N("Indore Wellness Private Limited")["name_key"]
    assert N("Indore Wellness Private Límited")["name_key"] == "indore wellness"


def test_domains_and_dotted_legal():
    assert N("indriyaclub.com")["name_compact"] == N("Indriya Club Limited")["name_compact"]
    assert N("choicehorizonrate.com")["name_compact"] == N("Choice Horizon Rate Inc")["name_compact"]
    assert N("indriyaclub.com")["name_is_domain"] == 1
    assert N("Aurus Transition P.L.L.C.")["name_key"] == N("Aurus Transition PLLC")["name_key"]


def test_duplicates_and_scripts():
    assert N("roopika estate estate ltd")["name_key"] == N("Roopika Estate Ltd")["name_key"]
    g = N("એપેક્સ કન્સ્ટ્રક્શન્સ પ્રાઇવેટ લિમિટેડ")
    assert g["name_script"] == "gujarati" and "limited" in g["legal"], g
    assert N("E+ Plains")["name_key"] == N("E+ Pláins")["name_key"] == "e plains"


def test_addresses_reorder_state_abbrev():
    a = A("11237 Lanewood Circle, Dallas, TX")
    b = A("Dallas, Texas, #11237 Lanewood Cir")
    assert a["addr_key"] == b["addr_key"], (a, b)
    assert A("6500 N Shore Road, Belfair, WA")["addr_key"] == A("6500 N Shore Rd, Belfair, Washington")["addr_key"]
    assert A("102 X Society, Ahmedabad, Gujarat")["state"] == A("102 X Society, Ahmedabad, GJ")["state"] == "gj"


def test_house_numbers_and_postal():
    assert A("Madhya Pradesh, 0405, BLOCK-H NILGIRI APARTMENT")["numbers"] == "405"
    assert A("H.No - 83-231/A/11 S K Nagar, Hyderabad")["house_no"] == "83"
    assert A("17560 Ellis Road, Tahlequah, OK")["postal"] == ""          # house number, not a ZIP
    assert A("100 Main St, Charlotte, NC 28202")["postal"] == "28202"
    assert A("12 Rue de Rivoli, 75004 Paris")["postal"] == "75004"
    assert A("Andheri East, Mumbai 400069, Maharashtra")["postal"] == "400069"
    assert A("")["addr_empty"] == 1


def test_french_legal_and_landmarks():
    # French legal suffixes stripped so core names match
    assert N("Société Générale de Banque (Cie)")["name_key"] == N("Societe Generale de Banque")["name_key"]
    assert N("Etablissements Dupont SARL")["name_key"] == "dupont"
    # Indian landmark prepositions canonicalized
    assert A("Near SBI ATM, Station Road")["addr_key"] == A("Nr SBI ATM, Station Rd")["addr_key"]
    assert A("Behind Bus Stand, MG Road")["addr_key"] == A("Bhnd Bus Stand, MG Rd")["addr_key"]


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
