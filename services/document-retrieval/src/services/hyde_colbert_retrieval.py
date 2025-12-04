"""
HyDE-ColBERT Retrieval Service

Combines HyDE hypothetical document generation with ColBERT late interaction
for improved retrieval performance.

Features:
- Generate hypothetical answer documents using Ollama
- Encode with ColBERT token-level embeddings
- Score fusion strategies (weighted_average, rrf, average_all)
- Integration with existing RAG retrieval pipeline
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.config import settings
from src.services.hyde_generator import HyDEGenerator, get_hyde_generator
from src.services.colbert_encoder import ColBERTEncoder, get_colbert_encoder
from src.services.colbert_index import ColBERTIndex, ColBERTIndexManager, get_colbert_index_manager

logger = logging.getLogger(__name__)


class HyDEColBERTRetrieval:
    """
    HyDE-ColBERT retrieval system.

    Combines:
    - HyDE hypothetical document generation
    - ColBERT token-level embeddings
    - MaxSim late interaction scoring
    - Score fusion strategies
    """

    def __init__(
        self,
        hyde_generator: Optional[HyDEGenerator] = None,
        colbert_encoder: Optional[ColBERTEncoder] = None,
        index_manager: Optional[ColBERTIndexManager] = None,
        fusion_weight: float = 0.2,
        fusion_strategy: str = "weighted_average",
        normalize_scores: bool = True,
    ):
        """
        Initialize HyDE-ColBERT retrieval.

        Args:
            hyde_generator: HyDE generator instance
            colbert_encoder: ColBERT encoder instance
            index_manager: ColBERT index manager
            fusion_weight: Weight for HyDE scores (0-1) vs query scores
                          Default 0.2 = 20% HyDE, 80% query
            fusion_strategy: Score fusion strategy
            normalize_scores: Whether to normalize scores before fusion
        """
        self._hyde_generator = hyde_generator
        self._colbert_encoder = colbert_encoder
        self._index_manager = index_manager
        self.fusion_weight = fusion_weight
        self.fusion_strategy = fusion_strategy
        self.normalize_scores = normalize_scores

        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "hyde_generation_time_ms": 0.0,
            "colbert_encoding_time_ms": 0.0,
            "scoring_time_ms": 0.0,
        }

    @property
    def hyde_generator(self) -> HyDEGenerator:
        """Lazy load HyDE generator."""
        if self._hyde_generator is None:
            # Note: This returns a coroutine, need to handle async
            raise RuntimeError("HyDE generator not initialized. Call initialize() first.")
        return self._hyde_generator

    @property
    def colbert_encoder(self) -> ColBERTEncoder:
        """Lazy load ColBERT encoder."""
        if self._colbert_encoder is None:
            self._colbert_encoder = get_colbert_encoder()
        return self._colbert_encoder

    @property
    def index_manager(self) -> ColBERTIndexManager:
        """Lazy load index manager."""
        if self._index_manager is None:
            self._index_manager = get_colbert_index_manager()
        return self._index_manager

    async def initialize(self) -> None:
        """Initialize the retrieval service."""
        if self._hyde_generator is None:
            self._hyde_generator = await get_hyde_generator()
        if self._colbert_encoder is None:
            self._colbert_encoder = get_colbert_encoder()
        if self._index_manager is None:
            self._index_manager = get_colbert_index_manager()

        logger.info("HyDE-ColBERT retrieval service initialized")

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    def _reciprocal_rank_fusion(
        self,
        score_lists: List[np.ndarray],
        k: int = 60,
    ) -> np.ndarray:
        """
        Reciprocal Rank Fusion for combining multiple rankings.

        RRF(d) = sum(1 / (k + rank_i(d)))

        Args:
            score_lists: List of score arrays for different queries
            k: RRF constant (default 60 from original paper)

        Returns:
            Fused scores
        """
        n_docs = len(score_lists[0])
        rrf_scores = np.zeros(n_docs)

        for scores in score_lists:
            ranks = np.argsort(np.argsort(-scores)) + 1
            rrf_scores += 1.0 / (k + ranks)

        return rrf_scores

    def _fuse_scores(
        self,
        query_scores: np.ndarray,
        hyde_scores_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Fuse query and HyDE scores using configured strategy.

        Args:
            query_scores: Scores from original query
            hyde_scores_list: List of scores from each hypothetical

        Returns:
            Fused scores
        """
        hyde_scores = np.mean(hyde_scores_list, axis=0)

        if self.fusion_strategy == "weighted_average":
            if self.normalize_scores:
                query_scores_norm = self._normalize_scores(query_scores)
                hyde_scores_norm = self._normalize_scores(hyde_scores)
                fused = (
                    self.fusion_weight * hyde_scores_norm +
                    (1 - self.fusion_weight) * query_scores_norm
                )
            else:
                fused = (
                    self.fusion_weight * hyde_scores +
                    (1 - self.fusion_weight) * query_scores
                )

        elif self.fusion_strategy == "average_all":
            all_scores = [query_scores] + hyde_scores_list
            fused = np.mean(all_scores, axis=0)

        elif self.fusion_strategy == "average_hyde_only":
            fused = hyde_scores

        elif self.fusion_strategy == "max":
            all_scores = np.vstack([query_scores] + hyde_scores_list)
            fused = np.max(all_scores, axis=0)

        elif self.fusion_strategy == "rrf":
            fused = self._reciprocal_rank_fusion([query_scores] + hyde_scores_list)

        else:
            logger.warning(f"Unknown fusion strategy: {self.fusion_strategy}, using weighted_average")
            fused = (
                self.fusion_weight * hyde_scores +
                (1 - self.fusion_weight) * query_scores
            )

        return fused

    async def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        domain: str = "general",
        n_hypotheticals: Optional[int] = None,
        fusion_strategy: Optional[str] = None,
        fusion_weight: Optional[float] = None,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE-ColBERT.

        Args:
            query: Search query
            collection_name: Name of the collection to search
            top_k: Number of documents to retrieve
            domain: Domain for HyDE generation
            n_hypotheticals: Number of hypotheticals to generate
            fusion_strategy: Override default fusion strategy
            fusion_weight: Override default fusion weight
            return_scores: Whether to include scores in results

        Returns:
            List of retrieved documents with scores
        """
        import time

        self.stats["total_queries"] += 1
        start_time = time.time()

        # Get index
        index = self.index_manager.get_index(collection_name)
        if not index.is_loaded():
            raise ValueError(
                f"ColBERT index for '{collection_name}' not loaded. "
                "Call create_index() first."
            )

        # Generate hypotheticals
        hyde_start = time.time()
        result = await self.hyde_generator.generate_with_fusion(
            query=query,
            domain=domain,
            n_docs=n_hypotheticals or settings.HYDE_N_HYPOTHETICALS,
        )
        hypotheticals = result["hypotheticals"]
        self.stats["hyde_generation_time_ms"] += (time.time() - hyde_start) * 1000

        logger.info(f"Generated {len(hypotheticals)} hypotheticals for query: {query[:50]}...")

        # Encode query and hypotheticals
        encode_start = time.time()
        query_emb = self.colbert_encoder.encode_queries([query])[0]
        hyde_embeddings = self.colbert_encoder.encode_queries(hypotheticals)
        self.stats["colbert_encoding_time_ms"] += (time.time() - encode_start) * 1000

        # Compute scores
        score_start = time.time()

        # Query scores
        query_scores = self.colbert_encoder.batch_maxsim_score(
            query_emb,
            index.doc_embeddings,
        )

        # HyDE scores
        hyde_scores_list = []
        for hyde_emb in hyde_embeddings:
            scores = self.colbert_encoder.batch_maxsim_score(
                hyde_emb,
                index.doc_embeddings,
            )
            hyde_scores_list.append(scores)

        self.stats["scoring_time_ms"] += (time.time() - score_start) * 1000

        # Fuse scores
        strategy = fusion_strategy or self.fusion_strategy
        weight = fusion_weight if fusion_weight is not None else self.fusion_weight

        # Temporarily override for this query
        original_strategy = self.fusion_strategy
        original_weight = self.fusion_weight
        self.fusion_strategy = strategy
        self.fusion_weight = weight

        fused_scores = self._fuse_scores(query_scores, hyde_scores_list)

        self.fusion_strategy = original_strategy
        self.fusion_weight = original_weight

        # Get top-k
        top_indices = np.argsort(fused_scores)[::-1][:top_k]
        hyde_scores_avg = np.mean(hyde_scores_list, axis=0)

        # Format results
        results = []
        for idx in top_indices:
            result = {
                "id": index.doc_ids[idx],
                "content": index.doc_texts[idx],
                "metadata": index.doc_metadata[idx],
                "source": "hyde_colbert",
            }
            if return_scores:
                result["score"] = float(fused_scores[idx])
                result["query_score"] = float(query_scores[idx])
                result["hyde_score"] = float(hyde_scores_avg[idx])

            results.append(result)

        self.stats["successful_retrievals"] += 1

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"HyDE-ColBERT retrieval: {len(results)} documents in {total_time:.1f}ms "
            f"(strategy={strategy}, weight={weight})"
        )

        return results

    async def retrieve_with_diagnostics(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        domain: str = "general",
        n_hypotheticals: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents with detailed diagnostics.

        Args:
            query: Search query
            collection_name: Name of the collection
            top_k: Number of documents to retrieve
            domain: Domain for HyDE generation
            n_hypotheticals: Number of hypotheticals to generate

        Returns:
            Dictionary with results and diagnostics
        """
        import time
        from scipy.stats import pearsonr

        start_time = time.time()

        # Get index
        index = self.index_manager.get_index(collection_name)
        if not index.is_loaded():
            raise ValueError(f"ColBERT index for '{collection_name}' not loaded.")

        # Generate hypotheticals
        result = await self.hyde_generator.generate_with_fusion(
            query=query,
            domain=domain,
            n_docs=n_hypotheticals or settings.HYDE_N_HYPOTHETICALS,
        )
        hypotheticals = result["hypotheticals"]

        # Encode
        query_emb = self.colbert_encoder.encode_queries([query])[0]
        hyde_embeddings = self.colbert_encoder.encode_queries(hypotheticals)

        # Compute scores
        query_scores = self.colbert_encoder.batch_maxsim_score(
            query_emb,
            index.doc_embeddings,
        )

        hyde_scores_list = []
        for hyde_emb in hyde_embeddings:
            scores = self.colbert_encoder.batch_maxsim_score(
                hyde_emb,
                index.doc_embeddings,
            )
            hyde_scores_list.append(scores)

        hyde_scores = np.mean(hyde_scores_list, axis=0)

        # Fuse scores
        fused_scores = self._fuse_scores(query_scores, hyde_scores_list)

        # Get top-k results
        top_indices = np.argsort(fused_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": index.doc_ids[idx],
                "content": index.doc_texts[idx],
                "metadata": index.doc_metadata[idx],
                "score": float(fused_scores[idx]),
                "query_score": float(query_scores[idx]),
                "hyde_score": float(hyde_scores[idx]),
            })

        # Compute diagnostics
        correlation, p_value = pearsonr(query_scores, hyde_scores)

        query_only_top = set(np.argsort(query_scores)[::-1][:top_k])
        hyde_only_top = set(np.argsort(hyde_scores)[::-1][:top_k])
        fused_top = set(top_indices)

        total_time = (time.time() - start_time) * 1000

        diagnostics = {
            "query_score_stats": {
                "mean": float(query_scores.mean()),
                "std": float(query_scores.std()),
                "min": float(query_scores.min()),
                "max": float(query_scores.max()),
            },
            "hyde_score_stats": {
                "mean": float(hyde_scores.mean()),
                "std": float(hyde_scores.std()),
                "min": float(hyde_scores.min()),
                "max": float(hyde_scores.max()),
            },
            "fused_score_stats": {
                "mean": float(fused_scores.mean()),
                "std": float(fused_scores.std()),
                "min": float(fused_scores.min()),
                "max": float(fused_scores.max()),
            },
            "score_correlation": float(correlation),
            "correlation_p_value": float(p_value),
            "top_k_overlaps": {
                "query_hyde": len(query_only_top & hyde_only_top),
                "fused_query": len(fused_top & query_only_top),
                "fused_hyde": len(fused_top & hyde_only_top),
            },
            "hypotheticals_generated": len(hypotheticals),
            "hypotheticals": hypotheticals,
            "fusion_strategy": self.fusion_strategy,
            "fusion_weight": self.fusion_weight,
            "processing_time_ms": total_time,
        }

        return {
            "results": results,
            "diagnostics": diagnostics,
        }

    async def compare_retrieval_methods(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        domain: str = "general",
    ) -> Dict[str, Any]:
        """
        Compare different retrieval methods.

        Returns results from:
        - Query-only ColBERT
        - HyDE-only ColBERT
        - Fused HyDE-ColBERT

        Args:
            query: Search query
            collection_name: Collection name
            top_k: Number of results
            domain: Domain for HyDE

        Returns:
            Comparison results
        """
        # Get index
        index = self.index_manager.get_index(collection_name)
        if not index.is_loaded():
            raise ValueError(f"ColBERT index for '{collection_name}' not loaded.")

        # Generate hypotheticals
        result = await self.hyde_generator.generate_with_fusion(
            query=query,
            domain=domain,
        )
        hypotheticals = result["hypotheticals"]

        # Encode
        query_emb = self.colbert_encoder.encode_queries([query])[0]
        hyde_embeddings = self.colbert_encoder.encode_queries(hypotheticals)

        # Query-only scores
        query_scores = self.colbert_encoder.batch_maxsim_score(
            query_emb,
            index.doc_embeddings,
        )

        # HyDE scores
        hyde_scores_list = []
        for hyde_emb in hyde_embeddings:
            scores = self.colbert_encoder.batch_maxsim_score(
                hyde_emb,
                index.doc_embeddings,
            )
            hyde_scores_list.append(scores)
        hyde_scores = np.mean(hyde_scores_list, axis=0)

        # Fused scores
        fused_scores = self._fuse_scores(query_scores, hyde_scores_list)

        def get_top_results(scores: np.ndarray, method: str) -> List[Dict]:
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    "rank": i + 1,
                    "id": index.doc_ids[idx],
                    "content": index.doc_texts[idx][:200] + "...",
                    "score": float(scores[idx]),
                    "method": method,
                }
                for i, idx in enumerate(top_idx)
            ]

        return {
            "query": query,
            "domain": domain,
            "hypotheticals": hypotheticals,
            "methods": {
                "query_only": get_top_results(query_scores, "query_only"),
                "hyde_only": get_top_results(hyde_scores, "hyde_only"),
                "fused": get_top_results(fused_scores, "fused"),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return dict(self.stats)

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all components.

        Returns:
            Dictionary with health status of each component
        """
        status = {
            "hyde_generator": False,
            "colbert_encoder": False,
            "index_manager": False,
        }

        try:
            if self._hyde_generator:
                status["hyde_generator"] = await self._hyde_generator.health_check()
        except Exception as e:
            logger.error(f"HyDE generator health check failed: {e}")

        try:
            if self._colbert_encoder:
                status["colbert_encoder"] = True
        except Exception as e:
            logger.error(f"ColBERT encoder health check failed: {e}")

        try:
            if self._index_manager:
                status["index_manager"] = True
        except Exception as e:
            logger.error(f"Index manager health check failed: {e}")

        return status


# Singleton instance
_hyde_colbert_retrieval: Optional[HyDEColBERTRetrieval] = None


async def get_hyde_colbert_retrieval() -> HyDEColBERTRetrieval:
    """Get or create the HyDE-ColBERT retrieval singleton."""
    global _hyde_colbert_retrieval
    if _hyde_colbert_retrieval is None:
        _hyde_colbert_retrieval = HyDEColBERTRetrieval()
        await _hyde_colbert_retrieval.initialize()
    return _hyde_colbert_retrieval
