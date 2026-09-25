"""
Pairwise similarity feature engineering for candidate entity records.
"""
from typing import Dict, Any
import numpy as np
from ..preprocessing.cleaner import TextCleaner, AddressNormalizer


def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token-level Jaccard similarity."""
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 and s1 != "" else 0.0
    ngrams1 = set(s1[i:i + n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i + n] for i in range(len(s2) - n + 1))
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return float(intersection / union) if union > 0 else 0.0


def quick_levenshtein_ratio(s1: str, s2: str) -> float:
    """Fast estimate of string similarity ratio based on prefix and token overlap."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # Try rapidfuzz if available, else fallback to token + character overlap
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(s1, s2) / 100.0
    except ImportError:
        # Fallback approximation
        prefix_len = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        max_l = max(len(s1), len(s2))
        return (prefix_len / max_l + jaccard_similarity(s1, s2)) / 2.0


class PairFeatureExtractor:
    """Extracts informative feature vectors for a pair of business records."""

    @classmethod
    def extract_features(cls, r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, float]:
        """Compute pairwise similarity metrics between record 1 and record 2."""
        # Clean texts
        name1 = TextCleaner.clean_name(r1.get("business_name", ""))
        name2 = TextCleaner.clean_name(r2.get("business_name", ""))
        addr1 = AddressNormalizer.clean_address(r1.get("business_address", ""))
        addr2 = AddressNormalizer.clean_address(r2.get("business_address", ""))
        c1 = str(r1.get("country", "")).upper().strip()
        c2 = str(r2.get("country", "")).upper().strip()

        # Country match
        same_country = 1.0 if c1 and c1 == c2 else 0.0

        # Name metrics
        name_exact = 1.0 if name1 and name1 == name2 else 0.0
        name_jaccard = jaccard_similarity(name1, name2)
        name_ngram = char_ngram_jaccard(name1, name2, n=3)
        name_ratio = quick_levenshtein_ratio(name1, name2)
        name_len_diff = abs(len(name1) - len(name2))

        # Address metrics
        addr_exact = 1.0 if addr1 and addr1 == addr2 else 0.0
        addr_jaccard = jaccard_similarity(addr1, addr2)
        addr_ngram = char_ngram_jaccard(addr1, addr2, n=3)
        addr_ratio = quick_levenshtein_ratio(addr1, addr2)

        # Postal code check
        p1 = AddressNormalizer.extract_postal_code(r1.get("business_address", ""), c1)
        p2 = AddressNormalizer.extract_postal_code(r2.get("business_address", ""), c2)
        postal_match = 1.0 if p1 and p2 and p1 == p2 else 0.0
        postal_mismatch = 1.0 if p1 and p2 and p1 != p2 else 0.0

        return {
            "same_country": same_country,
            "name_exact": name_exact,
            "name_jaccard": name_jaccard,
            "name_ngram_jaccard": name_ngram,
            "name_ratio": name_ratio,
            "name_len_diff": float(name_len_diff),
            "addr_exact": addr_exact,
            "addr_jaccard": addr_jaccard,
            "addr_ngram_jaccard": addr_ngram,
            "addr_ratio": addr_ratio,
            "postal_match": postal_match,
            "postal_mismatch": postal_mismatch,
        }
