from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Citation:
    """A single citation linking a recommendation to its evidence source.

    Attributes:
        recommendation: The recommended checkup item.
        source: Where the evidence came from (e.g., "相似病例 #671203", "临床指南-心血管").
        confidence: Confidence score for this citation.
        query_facet: Which query facet produced this evidence.
    """
    recommendation: str
    source: str
    confidence: float = 0.0
    query_facet: str = ""

    def format(self) -> str:
        return f"[{self.recommendation}] 依据: {self.source} (置信度: {self.confidence:.2f})"


class CitationTracker:
    """Tracks evidence citations for recommendations.

    Every recommendation must be grounded in retrieved evidence or clinical guidelines.
    This class manages the mapping between recommendations and their sources.
    """

    def __init__(self):
        self.citations: List[Citation] = []

    def add_citation(
        self,
        recommendation: str,
        source: str,
        confidence: float = 0.0,
        query_facet: str = ""
    ):
        """Add a citation linking a recommendation to evidence."""
        self.citations.append(Citation(
            recommendation=recommendation,
            source=source,
            confidence=confidence,
            query_facet=query_facet
        ))

    def add_citations_from_retrieval(
        self,
        recommendations: List[str],
        retrieval_results: Dict[str, List[str]],
        default_confidence: float = 0.7
    ):
        """Auto-link recommendations to retrieval results.

        For each recommendation, finds matching retrieval results and creates citations.

        Args:
            recommendations: List of recommended checkup items.
            retrieval_results: Dict mapping facet -> list of result strings.
            default_confidence: Default confidence for auto-linked citations.
        """
        for rec in recommendations:
            matched = False
            for facet, results in retrieval_results.items():
                for result in results:
                    # Simple keyword matching
                    if rec in result or any(kw in result for kw in rec.split("、")):
                        self.add_citation(
                            recommendation=rec,
                            source=result[:80],  # Truncate long sources
                            confidence=default_confidence,
                            query_facet=facet
                        )
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                self.add_citation(
                    recommendation=rec,
                    source="临床指南/标准推荐",
                    confidence=0.5,
                    query_facet="guideline"
                )

    def get_references_section(self) -> str:
        """Format all citations into a references section string."""
        if not self.citations:
            return ""

        lines = ["### 推荐依据\n"]
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"{i}. {cite.format()}")

        return "\n".join(lines)

    def get_ungrounded_recommendations(self, threshold: float = 0.3) -> List[str]:
        """Find recommendations with low-confidence or no evidence.

        Args:
            threshold: Minimum confidence to consider grounded.

        Returns:
            List of recommendation strings with weak evidence.
        """
        return [
            c.recommendation for c in self.citations
            if c.confidence < threshold
        ]

    def clear(self):
        """Clear all citations."""
        self.citations = []
