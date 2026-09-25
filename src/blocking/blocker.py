"""
Blocking strategies to prune the comparison space from O(N*M) to high-recall candidate sets.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from ..preprocessing.cleaner import TextCleaner


class RuleBlocker:
    """
    Multi-pass rule-based blocker that groups records by country,
    first significant name token, and 3-character prefixes.
    """

    def __init__(self, max_candidates_per_entity: int = 50):
        self.max_candidates = max_candidates_per_entity
        self.index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    def _extract_blocking_keys(self, name: str, country: str) -> Set[Tuple[str, str]]:
        """Generate blocking keys for a record."""
        keys = set()
        clean = TextCleaner.clean_name(name)
        country_norm = str(country).upper().strip()

        tokens = [t for t in clean.split() if len(t) > 2]
        if tokens:
            # Key 1: First word + country
            keys.add((country_norm, f"w1_{tokens[0]}"))
            # Key 2: First 3 characters of first word + country
            keys.add((country_norm, f"p3_{tokens[0][:3]}"))
            # Key 3: If multiple tokens, second word as well
            if len(tokens) > 1 and len(tokens[1]) > 2:
                keys.add((country_norm, f"w2_{tokens[1]}"))
        elif clean:
            keys.add((country_norm, f"raw_{clean[:4]}"))

        return keys

    def fit_targets(self, target_df: pd.DataFrame):
        """Build blocking inverted index for Source 2 and Source 3 records."""
        self.index.clear()
        for _, row in target_df.iterrows():
            eid = row["entity_id"]
            name = str(row.get("business_name", ""))
            country = str(row.get("country", ""))

            keys = self._extract_blocking_keys(name, country)
            for k in keys:
                self.index[k].append(eid)

    def block_entity(self, name: str, country: str) -> List[str]:
        """Retrieve candidate IDs for a single Source 1 entity."""
        keys = self._extract_blocking_keys(name, country)
        candidates = set()
        for k in keys:
            for cand_id in self.index.get(k, []):
                candidates.add(cand_id)
                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                break
        return list(candidates)


def generate_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_candidates_per_entity: int = 50
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and target sources.
    Returns DataFrame matching candidate_pairs.tsv schema.
    """
    blocker = RuleBlocker(max_candidates_per_entity=max_candidates_per_entity)
    blocker.fit_targets(target_df)

    records = []
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        cands = blocker.block_entity(row.get("business_name", ""), row.get("country", ""))
        records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands)
        })

    return pd.DataFrame(records)
