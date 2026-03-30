import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from typing import List, Tuple


class Rerank:
    """Cross-encoder reranker using sentence-transformers.

    Uses a cross-encoder model to rerank candidate passages given a query.
    Default model: BAAI/bge-reranker-v2-m3 (multilingual, strong on Chinese medical text).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_dir: str = None,
        **kwargs
    ):
        from sentence_transformers.cross_encoder import CrossEncoder
        self.model_name = model_name
        self.ranker = CrossEncoder(
            model_name=model_name,
            cache_folder=cache_dir,
            **kwargs
        )

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[dict]:
        """Rerank passages given a query.

        Args:
            query: The query string.
            passages: List of passage strings to rerank.
            top_k: Number of top results to return. If None, return all.

        Returns:
            List of dicts with keys: corpus_id, score, text, sorted by descending score.
        """
        if not passages:
            return []

        # Build sentence pairs
        sentence_pairs = [[query, passage] for passage in passages]

        # Score all pairs
        scores = self.ranker.predict(sentence_pairs)

        # Build results with index, score, text
        results = []
        for idx, score in enumerate(scores):
            results.append({
                'corpus_id': idx,
                'score': float(score),
                'text': passages[idx]
            })

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
