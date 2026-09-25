"""
Record normalisation v2 (P1).

Rules come from inspecting real train true-match pairs (25 Sep). The noise is systematic:
  names:     accent injection even in US/India ("Worldwide" -> "Wórldwide"), OCR digit swaps
             ("6eneral", "We1lness"), token shuffles/duplication, legal suffix added/dropped/
             bracketed/dotted ("[Limited]", "P.L.L.C."), domain forms ("indriyaclub.com"),
             native Indic scripts (Gujarati, Bengali, Devanagari ...).
  addresses: component reordering, state name vs code ("Texas"/"TX", "Gujarat"/"GJ"),
             '#', 'H.No', zero-padded or truncated house numbers, street-type abbreviations.

Hand-written normalisation lists only (no external lookup). Country is never hard-coded:
state/legal lists simply add knowledge; unknown countries fall through the generic path.
"""
import re
import unicodedata
from typing import Dict

try:
    from anyascii import anyascii   # ISC licence; transliterates Indic scripts to ASCII
except ImportError:                  # pragma: no cover
    anyascii = None

# ----------------------------------------------------------------------------- generic text
_COMBINING_LATIN = re.compile(r"[̀-ͯ]")
_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}")


class _CharMap(dict):
    """str.translate table: keep letters/digits/combining marks of ANY script, else -> space.
    (re's \\W would split Indic words at every vowel sign, so we classify by category.)"""
    def __missing__(self, cp):
        ch = chr(cp)
        v = ch if unicodedata.category(ch)[0] in "LNM" else " "
        self[cp] = v
        return v


_CHARMAP = _CharMap()


def strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return _COMBINING_LATIN.sub("", s).lower()


def base_clean(s: str) -> str:
    """accents off, lowercase, & -> and, punctuation -> space, collapse spaces."""
    if not s:
        return ""
    s = strip_accents_lower(s).replace("&", " and ")
    return " ".join(s.translate(_CHARMAP).split())


def script_of(s: str) -> str:
    """'latin' or the Unicode script word of the first non-Latin letter ('gujarati', ...)."""
    for ch in s:
        if ch.isalpha() and ord(ch) > 0x24F:
            n = unicodedata.name(ch, "")
            return n.split(" ")[0].lower() if n else "other"
    return "latin"


# ----------------------------------------------------------------------------- names
LEGAL = {
    # US / generic
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "pllc", "llp", "lp",
    "ltd", "limited", "plc", "pc", "pa", "esq",
    # India (incl. common transliterations of Indic spellings)
    "pvt", "private", "praivet", "praibhet", "prayvet",
    # France / EU
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scp", "selarl", "scop", "sca",
    "cie", "societe", "ste", "association", "ets", "etablissements",
    "gmbh", "ag", "bv", "nv",
}
DROP_WORDS = {"the", "and"}
_DOMAIN = re.compile(
    r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\."
    r"(?:co\.in|org\.in|net\.in|com|net|org|in|co|fr|biz|info|us|io)\s*/?\s*$", re.I)
_DOMAIN_LEGAL_TAIL = ("pvtltd", "privatelimited", "limited", "ltd", "llc", "inc", "corp", "co")
_OCR = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"}


def _ocr_fix(t: str) -> str:
    """Undo OCR digit swaps inside words: '6eneral'->'general', 'we1lness'->'wellness'.
    Only digits between letters, or a leading digit followed by 3 letters; len>=4."""
    if len(t) < 4 or t.isdigit() or t.isalpha():
        return t
    out = list(t)
    for i, ch in enumerate(t):
        if ch in _OCR:
            interior = 0 < i < len(t) - 1 and t[i - 1].isalpha() and t[i + 1].isalpha()
            leading = i == 0 and t[1:4].isalpha() and len(t[1:4]) == 3
            if interior or leading:
                out[i] = _OCR[ch]
    return "".join(out)


def normalize_name(raw: str) -> Dict[str, object]:
    s = raw or ""
    script = script_of(s)
    if script != "latin" and anyascii is not None:
        s = anyascii(s)
    is_domain = 0
    m = _DOMAIN.match(s)
    if m:
        s, is_domain = m.group(1).replace("-", " "), 1
    s = _DOTTED.sub(lambda x: x.group(0).replace(".", ""), s)
    toks = [_ocr_fix(t) for t in base_clean(s).split()]
    legal = sorted({t for t in toks if t in LEGAL})
    core = [t for t in toks if t not in LEGAL and t not in DROP_WORDS]
    if not core:                                   # name was only legal words: keep something
        core = [t for t in toks if t not in DROP_WORDS] or toks
    compact = "".join(core)
    if is_domain:                                  # 'roopikaestateltd.com' -> 'roopikaestate'
        for tail in _DOMAIN_LEGAL_TAIL:
            if compact.endswith(tail) and len(compact) > len(tail) + 2:
                compact = compact[: -len(tail)]
                break
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_key": " ".join(sorted(set(core))),   # order- and duplication-invariant
        "name_compact": compact,                   # matches domain forms / missing spaces
        "legal": " ".join(legal),
        "name_is_domain": is_domain,
        "name_script": script,
    }


# ----------------------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "ch", "andaman and nicobar islands": "an",
    # transliterated native-script spellings seen in the data (anyascii output)
    "gujrat": "gj", "krnatk": "ka", "mdhy prdes": "mp",
}
STATE_NAMES = {**US_STATES, **IN_STATES}
STATE_CODES = set(STATE_NAMES.values()) | {"ts", "or", "ct", "ut"}
ADDR_CANON = {
    "road": "rd", "street": "st", "str": "st", "drive": "dr", "circle": "cir", "lane": "ln",
    "court": "ct", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
    "place": "pl", "parkway": "pkwy", "highway": "hwy", "terrace": "ter", "square": "sq",
    "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "near": "nr", "opposite": "opp", "behind": "bhnd", "beside": "bsd", "adjacent": "adj",
    "township": "twp", "mount": "mt", "route": "rte", "sector": "sec",
    "chemin": "ch", "impasse": "imp", "allee": "allee", "allée": "allee", "quai": "quai",
}
STREET_TYPES = {"rd", "st", "dr", "cir", "ln", "ct", "ave", "blvd", "pl", "pkwy", "hwy", "ter",
                "sq", "way", "rue", "chemin", "allee", "impasse", "quai", "cr", "rte", "main", "ch", "imp"}
ADDR_DROP = {"no", "unit", "hno"}
_ADDR_LABEL = re.compile(
    r"\b(?:h\s*\.?\s*no|house\s*no|door\s*no|dor\s*no|flat\s*no|plot\s*no|shop\s*no)\b\.?")


def _postal_in_segment(toks) -> str:
    """6-digit PIN anywhere; 5-digit code only if it is not a house number:
    alone in its segment, next to a 2-letter state/region token, or '75008 paris' style."""
    for t in toks:
        if len(t) == 6 and t.isdigit() and t[0] != "0":
            return t
    for i, t in enumerate(toks):
        if len(t) == 5 and t.isdigit():
            others = [x for j, x in enumerate(toks) if j != i]
            if not others:
                return t
            if len(others) == 1 and len(others[0]) == 2 and others[0].isalpha():
                return t
            if i == 0 and not any(x in STREET_TYPES or x.isdigit() for x in others):
                return t
    return ""


def normalize_address(raw: str) -> Dict[str, str]:
    s = raw or ""
    if script_of(s) != "latin" and anyascii is not None:
        s = anyascii(s)
    s = _ADDR_LABEL.sub(" ", strip_accents_lower(s).replace("#", " "))
    all_toks, numbers, state, postal, house = [], [], "", "", ""
    for seg in s.split(","):
        clean = " ".join(seg.translate(_CHARMAP).split())
        if not clean:
            continue
        if clean in STATE_NAMES:
            clean = STATE_NAMES[clean]
        if clean in STATE_CODES and not state:
            state = clean
        toks = [ADDR_CANON.get(t, t) for t in clean.split() if t not in ADDR_DROP]
        if not postal:
            postal = _postal_in_segment(toks)
        toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks]
        nums = [t for t in toks if t.isdigit() and t != postal]
        if nums and not house and any(t in STREET_TYPES for t in toks):
            house = nums[0]
        numbers += nums
        all_toks += toks
    if not house and numbers:
        house = numbers[0]
    return {
        "addr_norm": " ".join(all_toks),
        "addr_key": " ".join(sorted(set(all_toks))),   # reorder-invariant
        "house_no": house,
        "numbers": " ".join(sorted(set(numbers))),
        "postal": postal,
        "state": state,
        "addr_empty": int(not all_toks),
    }


def normalize_record(name: str, address: str) -> Dict[str, object]:
    out = normalize_name(name)
    out.update(normalize_address(address))
    return out
