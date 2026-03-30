"""Main evaluator that runs all metric categories."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.retrieval_eval import evaluate_retrieval_batch
from eval.recommendation_eval import evaluate_recommendations_batch, load_guidelines
from eval.safety_eval import evaluate_safety_batch
from typing import List, Dict, Any
import json
import time


class Evaluator:
    """Unified evaluation framework for the HCR system.

    Evaluates across four dimensions:
    - Retrieval quality: Recall@K, MRR, nDCG@K
    - Recommendation quality: Accuracy, Coverage, Diversity, Precision
    - Safety: Hallucination rate, Omission rate
    - Performance: Latency
    """

    def __init__(self, guidelines_path: str = None):
        if guidelines_path:
            self.guidelines = load_guidelines(guidelines_path)
        else:
            self.guidelines = {}

    def evaluate_retrieval(self, test_cases: List[dict]) -> dict:
        """Evaluate retrieval quality.

        Args:
            test_cases: List of dicts with 'retrieved' and 'relevant' keys.
        """
        return evaluate_retrieval_batch(test_cases, k_values=[1, 3, 5])

    def evaluate_recommendations(self, test_cases: List[dict]) -> dict:
        """Evaluate recommendation quality.

        Args:
            test_cases: List of dicts with 'profile' and 'recommendation_text'.
        """
        return evaluate_recommendations_batch(test_cases, self.guidelines)

    def evaluate_safety(self, test_cases: List[dict]) -> dict:
        """Evaluate safety metrics.

        Args:
            test_cases: List of dicts with 'profile', 'recommendation_text', 'evidence', 'critical_items'.
        """
        return evaluate_safety_batch(test_cases)

    def evaluate_full(
        self,
        retrieval_cases: List[dict] = None,
        recommendation_cases: List[dict] = None,
        safety_cases: List[dict] = None,
        latency: float = None
    ) -> dict:
        """Run full evaluation suite.

        Returns:
            Dict with all metric categories.
        """
        results = {}

        if retrieval_cases:
            results["retrieval"] = self.evaluate_retrieval(retrieval_cases)

        if recommendation_cases:
            results["recommendation"] = self.evaluate_recommendations(recommendation_cases)

        if safety_cases:
            results["safety"] = self.evaluate_safety(safety_cases)

        if latency is not None:
            results["performance"] = {"latency_seconds": latency}

        return results

    @staticmethod
    def format_results(results: dict) -> str:
        """Format evaluation results as readable string."""
        lines = ["=== Evaluation Results ===\n"]

        for category, metrics in results.items():
            lines.append(f"【{category.upper()}】")
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"  {name}: {value:.4f}")
                    else:
                        lines.append(f"  {name}: {value}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_results(results: dict, path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
