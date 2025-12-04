"""
Hybrid Search Service - Combines Dense (Vector) and Sparse (BM25) retrieval
Optimized based on benchmark results: 30% dense + 70% sparse achieves best NDCG
"""
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from src.core.config import settings
from src.core.logging import logger


class FusionMethod(str, Enum):
    """Fusion methods for combining search results"""
    WEIGHTED = "weighted"
    RRF = "rrf"  # Reciprocal Rank Fusion


@dataclass
class SearchResult:
    """Individual search result"""
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "dense" or "sparse"


@dataclass
class HybridResult:
    """Combined hybrid search result"""
    doc_id: str
    content: str
    final_score: float
    dense_score: Optional[float]
    sparse_score: Optional[float]
    metadata: Dict[str, Any]


class BM25Scorer:
    """
    BM25 (Best Matching 25) sparse retrieval scorer
    Implements the Okapi BM25 ranking function
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 scorer

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.75 typical)
            epsilon: Floor for IDF computation
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Document statistics (populated during indexing)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.term_doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {doc_id -> term_freq}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on whitespace/punctuation"""
        import re
        # Convert to lowercase, remove punctuation, split
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Index documents for BM25 scoring

        Args:
            documents: List of (doc_id, content) tuples
        """
        self.doc_lengths = {}
        self.term_doc_freqs = {}
        self.inverted_index = {}

        total_length = 0
        self.doc_count = len(documents)

        for doc_id, content in documents:
            tokens = self._tokenize(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Count term frequencies in this document
            term_freqs: Dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1

            # Update inverted index and document frequencies
            seen_terms = set()
            for token, freq in term_freqs.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = freq

                if token not in seen_terms:
                    self.term_doc_freqs[token] = self.term_doc_freqs.get(token, 0) + 1
                    seen_terms.add(token)

        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        logger.info(f"BM25: Indexed {self.doc_count} documents, {len(self.inverted_index)} unique terms")

    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency for a term"""
        doc_freq = self.term_doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0

        # IDF formula with smoothing
        idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return max(idf, self.epsilon)

    def score(self, query: str, doc_id: str) -> float:
        """
        Calculate BM25 score for a query-document pair

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            BM25 score
        """
        query_tokens = self._tokenize(query)
        doc_length = self.doc_lengths.get(doc_id, 0)

        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            term_freq = self.inverted_index[term].get(doc_id, 0)
            if term_freq == 0:
                continue

            idf = self._idf(term)

            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search documents using BM25

        Args:
            query: Query string
            top_k: Number of results to return
            doc_ids: Optional list of doc_ids to search within

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        candidates = doc_ids if doc_ids else list(self.doc_lengths.keys())

        scores = []
        for doc_id in candidates:
            score = self.score(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


class HybridSearchService:
    """
    Hybrid search combining dense (vector) and sparse (BM25) retrieval
    Optimized configuration based on benchmark results
    """

    def __init__(
        self,
        vector_store=None,
        dense_weight: float = None,
        sparse_weight: float = None,
        fusion_method: str = None,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid search service

        Args:
            vector_store: Vector store service for dense retrieval
            dense_weight: Weight for dense retrieval (default from config)
            sparse_weight: Weight for sparse retrieval (default from config)
            fusion_method: Fusion method (default from config)
            rrf_k: RRF constant for reciprocal rank fusion
        """
        self.vector_store = vector_store
        self.bm25_scorer = BM25Scorer()

        # Use config defaults if not specified
        self.dense_weight = dense_weight if dense_weight is not None else settings.HYBRID_DENSE_WEIGHT
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.HYBRID_SPARSE_WEIGHT
        self.fusion_method = FusionMethod(fusion_method or settings.HYBRID_FUSION_METHOD)
        self.rrf_k = rrf_k

        # Document content cache for BM25
        self.doc_content_cache: Dict[str, str] = {}

        logger.info(
            f"HybridSearch initialized: dense_weight={self.dense_weight}, "
            f"sparse_weight={self.sparse_weight}, fusion={self.fusion_method.value}"
        )

    def index_for_bm25(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 sparse retrieval

        Args:
            documents: List of documents with 'id' and 'content' fields
        """
        doc_tuples = []
        for doc in documents:
            doc_id = doc.get("id") or self._generate_doc_id(doc.get("content", ""))
            content = doc.get("content", "")
            doc_tuples.append((doc_id, content))
            self.doc_content_cache[doc_id] = content

        self.bm25_scorer.index_documents(doc_tuples)

    def _generate_doc_id(self, content: str) -> str:
        """Generate a document ID from content hash"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining dense and sparse retrieval

        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            dense_weight: Override dense weight for this query
            sparse_weight: Override sparse weight for this query

        Returns:
            List of hybrid search results
        """
        d_weight = dense_weight if dense_weight is not None else self.dense_weight
        s_weight = sparse_weight if sparse_weight is not None else self.sparse_weight

        logger.info(f"Hybrid search: query='{query[:50]}...', dense={d_weight}, sparse={s_weight}")

        # Get more candidates for fusion
        candidate_k = min(top_k * 5, 100)

        dense_results = []
        sparse_results = []

        # Dense (vector) retrieval
        if d_weight > 0 and self.vector_store:
            try:
                dense_raw = await self.vector_store.hybrid_search(
                    query=query,
                    top_k=candidate_k,
                    filter_dict=filter_dict,
                    score_threshold=0.0  # Get all results for fusion
                )
                dense_results = [
                    SearchResult(
                        doc_id=self._generate_doc_id(text),
                        content=text,
                        score=score,
                        metadata=meta,
                        source="dense"
                    )
                    for text, score, meta in dense_raw
                ]
                logger.info(f"Dense search returned {len(dense_results)} results")
            except Exception as e:
                logger.warning(f"Dense search failed: {e}")

        # Sparse (BM25) retrieval
        if s_weight > 0 and self.bm25_scorer.doc_count > 0:
            try:
                sparse_raw = self.bm25_scorer.search(query, top_k=candidate_k)
                sparse_results = [
                    SearchResult(
                        doc_id=doc_id,
                        content=self.doc_content_cache.get(doc_id, ""),
                        score=score,
                        metadata={},
                        source="sparse"
                    )
                    for doc_id, score in sparse_raw
                ]
                logger.info(f"Sparse search returned {len(sparse_results)} results")
            except Exception as e:
                logger.warning(f"Sparse search failed: {e}")

        # Fuse results
        if self.fusion_method == FusionMethod.RRF:
            fused = self._fuse_rrf(dense_results, sparse_results, d_weight, s_weight)
        else:
            fused = self._fuse_weighted(dense_results, sparse_results, d_weight, s_weight)

        # Return top_k results
        return fused[:top_k]

    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results

        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        score_range = max_score - min_score

        if score_range == 0:
            return results

        for r in results:
            r.score = (r.score - min_score) / score_range

        return results

    def _fuse_weighted(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[HybridResult]:
        """
        Fuse results using weighted combination

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores

        Returns:
            Fused and ranked results
        """
        # Normalize scores
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # Build score maps
        doc_scores: Dict[str, Dict[str, Any]] = {}

        for r in dense_results:
            if r.doc_id not in doc_scores:
                doc_scores[r.doc_id] = {
                    "content": r.content,
                    "metadata": r.metadata,
                    "dense_score": 0.0,
                    "sparse_score": 0.0
                }
            doc_scores[r.doc_id]["dense_score"] = r.score
            doc_scores[r.doc_id]["metadata"].update(r.metadata)

        for r in sparse_results:
            if r.doc_id not in doc_scores:
                doc_scores[r.doc_id] = {
                    "content": r.content,
                    "metadata": r.metadata,
                    "dense_score": 0.0,
                    "sparse_score": 0.0
                }
            doc_scores[r.doc_id]["sparse_score"] = r.score
            if not doc_scores[r.doc_id]["content"]:
                doc_scores[r.doc_id]["content"] = r.content

        # Compute final scores
        results = []
        for doc_id, scores in doc_scores.items():
            final_score = (
                dense_weight * scores["dense_score"] +
                sparse_weight * scores["sparse_score"]
            )
            results.append(HybridResult(
                doc_id=doc_id,
                content=scores["content"],
                final_score=final_score,
                dense_score=scores["dense_score"] if scores["dense_score"] > 0 else None,
                sparse_score=scores["sparse_score"] if scores["sparse_score"] > 0 else None,
                metadata=scores["metadata"]
            ))

        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results

    def _fuse_rrf(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[HybridResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF)

        RRF score = Σ weight / (k + rank)

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results

        Returns:
            Fused and ranked results
        """
        doc_scores: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for rank, r in enumerate(dense_results):
            if r.doc_id not in doc_scores:
                doc_scores[r.doc_id] = {
                    "content": r.content,
                    "metadata": r.metadata,
                    "rrf_score": 0.0,
                    "dense_score": r.score,
                    "sparse_score": None
                }
            rrf_contribution = dense_weight / (self.rrf_k + rank + 1)
            doc_scores[r.doc_id]["rrf_score"] += rrf_contribution

        # Process sparse results
        for rank, r in enumerate(sparse_results):
            if r.doc_id not in doc_scores:
                doc_scores[r.doc_id] = {
                    "content": r.content,
                    "metadata": r.metadata,
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "sparse_score": r.score
                }
            else:
                doc_scores[r.doc_id]["sparse_score"] = r.score

            rrf_contribution = sparse_weight / (self.rrf_k + rank + 1)
            doc_scores[r.doc_id]["rrf_score"] += rrf_contribution

            if not doc_scores[r.doc_id]["content"]:
                doc_scores[r.doc_id]["content"] = r.content

        # Create results
        results = []
        for doc_id, scores in doc_scores.items():
            results.append(HybridResult(
                doc_id=doc_id,
                content=scores["content"],
                final_score=scores["rrf_score"],
                dense_score=scores["dense_score"],
                sparse_score=scores["sparse_score"],
                metadata=scores["metadata"]
            ))

        # Sort by RRF score
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results
