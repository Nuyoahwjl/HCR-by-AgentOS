import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from typing import List, Tuple
from agentos.rag.data import BaseData


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[int, float]]],
    top_k: int = 10,
    k: int = 60
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        results_list: List of ranked results, each as [(index, score), ...].
        top_k: Number of final results to return.
        k: RRF constant (default 60, per Cormack et al.).

    Returns:
        Fused list of (index, rrf_score) tuples sorted descending.
    """
    fused_scores = {}
    for results in results_list:
        for rank, (idx, _score) in enumerate(results):
            if idx not in fused_scores:
                fused_scores[idx] = 0.0
            fused_scores[idx] += 1.0 / (k + rank + 1)

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def weighted_fusion(
    results_list: List[List[Tuple[int, float]]],
    weights: List[float],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists using weighted score normalization.

    Each result list's scores are min-max normalized, then combined with weights.

    Args:
        results_list: List of ranked results, each as [(index, score), ...].
        weights: Weight for each result list (should sum to 1.0).
        top_k: Number of final results to return.

    Returns:
        Fused list of (index, weighted_score) tuples sorted descending.
    """
    fused_scores = {}

    for results, weight in zip(results_list, weights):
        if not results:
            continue

        # Min-max normalize scores
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s != min_s else 1.0

        for idx, score in results:
            normalized = (score - min_s) / range_s
            if idx not in fused_scores:
                fused_scores[idx] = 0.0
            fused_scores[idx] += weight * normalized

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class HybridRetriever:
    """Hybrid retriever combining BM25 lexical search and dense vector search.

    Supports two fusion strategies:
    - RRF (Reciprocal Rank Fusion): rank-based, no score normalization needed
    - Weighted fusion: score-based, min-max normalization + weighted sum

    Pipeline: BM25 + Dense (parallel) -> Fusion -> [Reranker] -> results
    """

    def __init__(
        self,
        chroma_db,
        bm25_retriever,
        fusion_method: str = "rrf",
        alpha: float = 0.6,
        rrf_k: int = 60
    ):
        """
        Args:
            chroma_db: ChromaDB instance for dense vector retrieval.
            bm25_retriever: BM25Retriever instance for lexical retrieval.
            fusion_method: "rrf" for Reciprocal Rank Fusion, "weighted" for weighted fusion.
            alpha: Weight for dense retrieval in weighted fusion (0-1). BM25 gets 1-alpha.
            rrf_k: RRF constant (only used for rrf method).
        """
        self.chroma_db = chroma_db
        self.bm25 = bm25_retriever
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        reranker=None,
        rerank_top_k: int = 5
    ) -> List[BaseData]:
        """Run hybrid retrieval pipeline.

        Args:
            query: Query string.
            top_k: Final number of results to return.
            dense_top_k: Number of candidates from dense retrieval.
            bm25_top_k: Number of candidates from BM25 retrieval.
            reranker: Optional Rerank instance for second-stage reranking.
            rerank_top_k: Number of results after reranking.

        Returns:
            List of BaseData results.
        """
        # Stage 1: Parallel retrieval
        dense_results = self._dense_retrieve(query, dense_top_k)
        bm25_results = self._bm25_retrieve(query, bm25_top_k)

        # Stage 2: Fusion
        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(
                [dense_results, bm25_results],
                top_k=top_k,
                k=self.rrf_k
            )
        else:
            fused = weighted_fusion(
                [dense_results, bm25_results],
                weights=[self.alpha, 1.0 - self.alpha],
                top_k=top_k
            )

        # Map fused indices back to documents
        all_documents = self._collect_all_documents(query, dense_top_k, bm25_top_k)
        fused_docs = []
        fused_doc_indices = set()
        for idx, score in fused:
            if idx < len(all_documents) and idx not in fused_doc_indices:
                doc = all_documents[idx]
                doc.add_metadata("fusion_score", score)
                fused_docs.append(doc)
                fused_doc_indices.add(idx)

        # Stage 3: Optional reranking
        if reranker and fused_docs:
            passages = [doc.get_content() for doc in fused_docs]
            reranked = reranker.rerank(query, passages)
            reranked_docs = []
            for item in reranked[:rerank_top_k]:
                corpus_id = item['corpus_id']
                if corpus_id < len(fused_docs):
                    reranked_docs.append(fused_docs[corpus_id])
            return reranked_docs

        return fused_docs[:top_k]

    def _dense_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Retrieve using dense vector search. Returns indexed results with scores."""
        raw_results = self.chroma_db.collection.query(query_texts=query, n_results=top_k)
        results = []
        for i in range(len(raw_results['ids'][0])):
            # ChromaDB returns distances; convert to similarity score (1 - distance)
            distance = raw_results['distances'][0][i] if 'distances' in raw_results else 0.0
            score = 1.0 / (1.0 + distance)
            results.append((i, score))
        return results

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Retrieve using BM25. Returns indexed results with scores."""
        return self.bm25.query(query, top_k)

    def _collect_all_documents(
        self,
        query: str,
        dense_top_k: int,
        bm25_top_k: int
    ) -> List[BaseData]:
        """Collect all unique documents from both retrieval paths, indexed for fusion."""
        seen_contents = set()
        all_docs = []

        # Dense results
        dense_raw = self.chroma_db.collection.query(query_texts=query, n_results=dense_top_k)
        for i in range(len(dense_raw['documents'][0])):
            content = dense_raw['documents'][0][i]
            metadata = dense_raw['metadatas'][0][i] if dense_raw['metadatas'] else {}
            if content not in seen_contents:
                seen_contents.add(content)
                all_docs.append(BaseData(content, dict(metadata)))

        # BM25 results
        for doc_text, score in self.bm25.query_with_scores(query, bm25_top_k):
            if doc_text not in seen_contents:
                seen_contents.add(doc_text)
                all_docs.append(BaseData(doc_text, {"bm25_score": score}))

        return all_docs
