import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from typing import List, Tuple
import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 lexical retriever for Chinese medical text.

    Builds an inverted index over a corpus of documents using jieba tokenization
    and BM25Okapi scoring.
    """

    def __init__(self, corpus: List[str] = None):
        self.corpus = corpus or []
        self.tokenized_corpus = []
        self.bm25 = None
        if corpus:
            self.build_index(corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba."""
        return list(jieba.cut_for_search(text))

    def build_index(self, corpus: List[str]):
        """Build BM25 index from a list of document strings."""
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Query the BM25 index.

        Args:
            query_text: The query string.
            top_k: Number of top results to return.

        Returns:
            List of (corpus_index, score) tuples sorted by descending score.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return scored_indices[:top_k]

    def query_with_scores(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Query and return document texts with scores.

        Args:
            query_text: The query string.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, score) tuples.
        """
        results = self.query(query_text, top_k)
        return [(self.corpus[idx], score) for idx, score in results]

    def add_documents(self, documents: List[str]):
        """Add documents to the existing index (rebuilds the index)."""
        self.corpus.extend(documents)
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def __len__(self):
        return len(self.corpus)
