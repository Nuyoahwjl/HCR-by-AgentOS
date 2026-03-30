import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from typing import List, Dict, Any
import json
from config.settings import Config


class QueryDecomposer:
    """Decomposes a user profile into multiple retrieval sub-queries.

    Each sub-query targets a different facet of the user's health profile:
    - symptom: symptom-related retrieval
    - history: past medical history retrieval
    - demographic: age/gender guideline retrieval
    - risk: combined risk factor retrieval

    Optionally expands queries with medical synonyms via MedicalOntology.
    """

    def __init__(self, ontology=None):
        """
        Args:
            ontology: Optional MedicalOntology instance for synonym expansion.
        """
        self.ontology = ontology

    def decompose(self, user_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Decompose user profile into sub-queries.

        Args:
            user_info: Dict with keys: id, gender, age, height, weight,
                       medical_history, symptoms.

        Returns:
            List of sub-query dicts with keys: facet, query.
        """
        sub_queries = []

        # 1. Symptom query
        symptoms = user_info.get("symptoms", "").strip()
        if symptoms and symptoms != "无":
            symptom_query = f"症状:{symptoms}"
            sub_queries.append({"facet": "symptom", "query": symptom_query})

            # Expand with synonyms if ontology available
            if self.ontology:
                expanded = self.ontology.expand_query(symptoms)
                for exp_symptom in expanded[:3]:  # top 3 expansions
                    sub_queries.append({
                        "facet": "symptom_expanded",
                        "query": f"症状:{exp_symptom}"
                    })

        # 2. History query
        history = user_info.get("medical_history", "").strip()
        if history and history != "无":
            history_query = f"既往病史:{history}"
            sub_queries.append({"facet": "history", "query": history_query})

            if self.ontology:
                expanded = self.ontology.expand_query(history)
                for exp_term in expanded[:2]:
                    sub_queries.append({
                        "facet": "history_expanded",
                        "query": f"既往病史:{exp_term}"
                    })

        # 3. Demographic query
        age = user_info.get("age", "")
        gender = user_info.get("gender", "")
        gender_cn = "男" if gender == "male" else ("女" if gender == "female" else gender)
        if age and gender:
            demo_query = f"{gender_cn},{age}岁"
            sub_queries.append({"facet": "demographic", "query": demo_query})

        # 4. Risk combination query (multiple risk factors combined)
        risk_factors = []
        try:
            age_int = int(age)
            if age_int >= 50:
                risk_factors.append(f"{age_int}岁")
            if age_int >= 40 and gender == "male":
                risk_factors.append("中年男性")
        except (ValueError, TypeError):
            pass

        if history and history != "无":
            risk_factors.append(history)
        if symptoms and symptoms != "无":
            risk_factors.append(symptoms)

        if len(risk_factors) >= 2:
            risk_query = ",".join(risk_factors)
            sub_queries.append({"facet": "risk_combination", "query": risk_query})

        # 5. Always include a full profile query as fallback
        full_parts = []
        if gender_cn:
            full_parts.append(gender_cn)
        if age:
            full_parts.append(f"{age}岁")
        height = user_info.get("height", "")
        weight = user_info.get("weight", "")
        if height:
            full_parts.append(f"身高{height}")
        if weight:
            full_parts.append(f"体重{weight}")
        if history and history != "无":
            full_parts.append(history)
        if symptoms and symptoms != "无":
            full_parts.append(symptoms)

        if full_parts:
            sub_queries.append({"facet": "full_profile", "query": ",".join(full_parts)})

        return sub_queries

    def merge_sub_query_results(
        self,
        all_results: Dict[str, List[str]],
        max_total: int = 10
    ) -> List[str]:
        """Merge results from multiple sub-queries with deduplication.

        Args:
            all_results: Dict mapping facet name to list of result strings.
            max_total: Maximum total results to return.

        Returns:
            Deduplicated list of result strings.
        """
        seen = set()
        merged = []

        # Priority order: risk_combination > symptom > history > demographic > others
        priority_order = [
            "risk_combination", "symptom", "symptom_expanded",
            "history", "history_expanded", "demographic",
            "full_profile"
        ]

        for facet in priority_order:
            if facet in all_results:
                for result in all_results[facet]:
                    # Simple dedup by first 50 chars
                    key = result[:50] if len(result) > 50 else result
                    if key not in seen:
                        seen.add(key)
                        merged.append(result)
                        if len(merged) >= max_total:
                            return merged

        return merged
