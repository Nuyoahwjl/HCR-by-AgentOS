import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import json
from typing import List, Dict
from config.settings import Config


class MedicalOntology:
    """Lightweight medical synonym dictionary for query expansion.

    Loads synonyms from a JSON file and expands queries with
    synonymous medical terms to improve retrieval recall.
    """

    def __init__(self, synonym_path: str = None):
        """
        Args:
            synonym_path: Path to the JSON synonym dictionary.
                          If None, uses Config.SYNONYM_PATH.
        """
        if synonym_path is None:
            synonym_path = os.path.join(project_root, Config.SYNONYM_PATH.lstrip("/"))
        self.synonyms = self._load_synonyms(synonym_path)
        # Build reverse index: term -> list of synonyms
        self.reverse_index = self._build_reverse_index()

    def _load_synonyms(self, path: str) -> Dict[str, List[str]]:
        """Load synonym dictionary from JSON file."""
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_reverse_index(self) -> Dict[str, str]:
        """Build reverse index: synonym -> canonical term."""
        reverse = {}
        for canonical, syns in self.synonyms.items():
            for syn in syns:
                reverse[syn] = canonical
            reverse[canonical] = canonical
        return reverse

    def expand_query(self, query: str) -> List[str]:
        """Expand a query string with synonyms.

        Looks for known terms in the query and returns expansions
        with synonymous terms replaced/augmented.

        Args:
            query: The original query string.

        Returns:
            List of expanded query strings (original first, then expansions).
        """
        expansions = []
        for canonical, syns in self.synonyms.items():
            # Check if canonical or any synonym appears in the query
            all_terms = [canonical] + syns
            found = False
            for term in all_terms:
                if term in query:
                    found = True
                    break

            if found:
                # Generate expansion by replacing with each synonym
                for syn in syns:
                    if syn not in query:
                        expanded = query
                        for term in all_terms:
                            if term in expanded:
                                expanded = expanded.replace(term, syn, 1)
                                break
                        if expanded != query:
                            expansions.append(expanded)

        return expansions

    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a single term.

        Args:
            term: The medical term.

        Returns:
            List of synonyms (including the term itself).
        """
        # Direct lookup
        if term in self.synonyms:
            return [term] + self.synonyms[term]

        # Reverse lookup
        if term in self.reverse_index:
            canonical = self.reverse_index[term]
            return [canonical] + self.synonyms.get(canonical, [])

        return [term]

    def get_canonical(self, term: str) -> str:
        """Get the canonical form of a medical term.

        Args:
            term: Any term (synonym or canonical).

        Returns:
            The canonical term, or the input itself if not found.
        """
        if term in self.synonyms:
            return term
        return self.reverse_index.get(term, term)
