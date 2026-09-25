"""
Text cleaning and normalization utilities for entity names and addresses.
"""
import re
from typing import Optional


class TextCleaner:
    """Normalizes business names, handling abbreviations, punctuation, and casing."""

    NAME_EXPANSIONS = {
        r"\bcorp\b": "corporation",
        r"\binc\b": "incorporated",
        r"\bltd\b": "limited",
        r"\bpvt\b": "private",
        r"\bco\b": "company",
        r"\bllc\b": "limited liability company",
        r"\bdept\b": "department",
        r"\bmfg\b": "manufacturing",
        r"\bintl\b": "international",
        r"\bassoc\b": "association",
        r"&": " and ",
    }

    @classmethod
    def clean_name(cls, text: Optional[str]) -> str:
        """Clean and normalize a business name string."""
        if not text or not isinstance(text, str):
            return ""

        text = text.lower().strip()
        for pattern, replacement in cls.NAME_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)

        # Remove special characters except alphanumeric and whitespace
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


class AddressNormalizer:
    """Normalizes addresses across different countries (US, India, France)."""

    ADDRESS_EXPANSIONS = {
        r"\bst\b": "street",
        r"\brd\b": "road",
        r"\bave\b": "avenue",
        r"\bblvd\b": "boulevard",
        r"\bln\b": "lane",
        r"\bdr\b": "drive",
        r"\bhwy\b": "highway",
        r"\bste\b": "suite",
        r"\bapt\b": "apartment",
        r"\bfl\b": "floor",
        r"\bpl\b": "place",
        r"\bpkway\b|\bpkwy\b": "parkway",
        r"\bnear\b": "near",
        r"\bopp\b": "opposite",
    }

    @classmethod
    def clean_address(cls, text: Optional[str]) -> str:
        """Clean and standardize an address string."""
        if not text or not isinstance(text, str):
            return ""

        text = text.lower().strip()
        for pattern, replacement in cls.ADDRESS_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_postal_code(text: Optional[str], country: str) -> Optional[str]:
        """Extract postal code according to country format."""
        if not text or not isinstance(text, str):
            return None

        country = (country or "").upper().strip()
        if country == "INDIA":
            # 6-digit PIN code
            match = re.search(r"\b([1-9][0-9]{5})\b", text)
            if match:
                return match.group(1)
        elif country in ("US", "FRANCE"):
            # 5-digit ZIP / Code Postal
            match = re.search(r"\b([0-9]{5})\b", text)
            if match:
                return match.group(1)

        return None
