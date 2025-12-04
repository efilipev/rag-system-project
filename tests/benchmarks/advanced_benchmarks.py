"""
Additional Advanced Benchmarks for RAG Systems

1. Multi-Representation Indexing (Parent-Child, Summaries)
2. Late Chunking Benchmark
3. Context Length Impact
4. Hybrid Search (Dense + Sparse)
5. Index Type Comparison (HNSW, IVF, Flat)
6. Matryoshka Embeddings (Multi-resolution)
"""
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexingStrategy(str, Enum):
    """Indexing strategies"""
    STANDARD = "standard"  # Chunk → Embed → Index
    PARENT_CHILD = "parent_child"  # Store summaries, retrieve full docs
    LATE_CHUNKING = "late_chunking"  # Embed full doc, then chunk
    MULTI_REPRESENTATION = "multi_rep"  # Multiple embeddings per doc
    HIERARCHICAL = "hierarchical"  # RAPTOR-style tree


class IndexType(str, Enum):
    """Vector index types"""
    FLAT = "flat"  # Exact search
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    IVF = "ivf"  # Inverted File Index
    ANNOY = "annoy"  # Approximate Nearest Neighbors
    SCANN = "scann"  # Scalable Nearest Neighbors


@dataclass
class BenchmarkResult:
    """Generic benchmark result"""
    test_name: str
    configuration: Dict[str, Any]
    ndcg_at_10: float
    recall_at_10: float
    latency_ms: float
    storage_mb: float
    indexing_time_s: float
    metadata: Dict[str, Any]


class AdvancedBenchmarks:
    """
    Additional advanced benchmarks for RAG systems
    """

    def __init__(
        self,
        retriever,
        results_dir: str = "./advanced_benchmark_results"
    ):
        """Initialize advanced benchmarks"""
        self.retriever = retriever
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def benchmark_multi_representation_indexing(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict
    ) -> pd.DataFrame:
        """
        Benchmark multi-representation indexing strategies

        Strategies:
        1. Standard: Chunk docs → Embed chunks → Index
        2. Parent-Child: Embed summaries → Store full docs → Retrieve full
        3. Late Chunking: Embed full doc → Chunk embeddings → Index
        4. Multi-Rep: Multiple embeddings per doc (summary, detail, keywords)

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments

        Returns:
            Comparison results
        """
        logger.info("Benchmarking multi-representation indexing...")

        strategies = [
            IndexingStrategy.STANDARD,
            IndexingStrategy.PARENT_CHILD,
            IndexingStrategy.LATE_CHUNKING,
            IndexingStrategy.MULTI_REPRESENTATION
        ]

        results = []

        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy.value}")

            result = await self._test_indexing_strategy(
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                strategy=strategy
            )

            results.append(result)

        df = pd.DataFrame([vars(r) for r in results])
        df.to_csv(self.results_dir / "multi_rep_indexing.csv", index=False)

        return df

    async def benchmark_hybrid_search(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict
    ) -> pd.DataFrame:
        """
        Benchmark hybrid search (dense + sparse)

        Combinations:
        - Dense only (vector search)
        - Sparse only (BM25)
        - Hybrid: Dense + Sparse with different fusion weights
        - Hybrid: Dense + Sparse with RRF
        - Triple: Dense + Sparse + Learned fusion

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments

        Returns:
            Comparison results
        """
        logger.info("Benchmarking hybrid search...")

        configurations = [
            {"dense_weight": 1.0, "sparse_weight": 0.0, "name": "Dense Only"},
            {"dense_weight": 0.0, "sparse_weight": 1.0, "name": "Sparse Only (BM25)"},
            {"dense_weight": 0.7, "sparse_weight": 0.3, "name": "Hybrid 70/30"},
            {"dense_weight": 0.5, "sparse_weight": 0.5, "name": "Hybrid 50/50"},
            {"dense_weight": 0.3, "sparse_weight": 0.7, "name": "Hybrid 30/70"},
            {"fusion": "rrf", "name": "Hybrid RRF"},
        ]

        results = []

        for config in configurations:
            logger.info(f"Testing: {config['name']}")

            # Execute hybrid search
            metrics = await self.retriever.hybrid_search(
                queries=queries,
                qrels=qrels,
                dense_weight=config.get("dense_weight", 0.5),
                sparse_weight=config.get("sparse_weight", 0.5),
                fusion_method=config.get("fusion", "weighted")
            )

            results.append({
                "configuration": config["name"],
                "dense_weight": config.get("dense_weight", 0.5),
                "sparse_weight": config.get("sparse_weight", 0.5),
                "fusion_method": config.get("fusion", "weighted"),
                "ndcg_at_10": metrics["NDCG@10"],
                "recall_at_10": metrics["Recall@10"],
                "map_at_10": metrics["MAP@10"],
            })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "hybrid_search.csv", index=False)

        return df

    async def benchmark_index_types(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        num_vectors: int = 100000
    ) -> pd.DataFrame:
        """
        Compare different vector index types

        Index Types:
        - Flat: Exact search (baseline)
        - HNSW: Fast approximate search
        - IVF: Inverted file index
        - Annoy: Tree-based approximate search

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            num_vectors: Number of vectors for scaling test

        Returns:
            Comparison results
        """
        logger.info("Benchmarking index types...")

        index_configs = {
            IndexType.FLAT: {"exact": True},
            IndexType.HNSW: {"M": 16, "ef_construction": 200, "ef_search": 100},
            IndexType.IVF: {"n_lists": 100, "n_probe": 10},
        }

        results = []

        for index_type, params in index_configs.items():
            logger.info(f"Testing index type: {index_type.value}")

            # Create index
            index = await self.retriever.create_index(
                index_type=index_type,
                **params
            )

            # Measure indexing time
            import time
            start_time = time.time()

            await index.add_documents(corpus)

            indexing_time = time.time() - start_time

            # Measure search quality and latency
            start_time = time.time()

            metrics = await index.search(queries=queries, qrels=qrels)

            avg_latency = (time.time() - start_time) / len(queries) * 1000  # ms

            # Memory usage
            memory_mb = index.get_memory_usage() / (1024 * 1024)

            results.append({
                "index_type": index_type.value,
                "parameters": str(params),
                "ndcg_at_10": metrics["NDCG@10"],
                "recall_at_10": metrics["Recall@10"],
                "indexing_time_s": indexing_time,
                "search_latency_ms": avg_latency,
                "memory_mb": memory_mb,
                "exact_search": index_type == IndexType.FLAT
            })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "index_types.csv", index=False)

        return df

    async def benchmark_context_length(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict
    ) -> pd.DataFrame:
        """
        Test impact of context length on retrieval quality

        Tests:
        - How much surrounding context helps
        - Title + snippet vs full document
        - Different context window sizes

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments

        Returns:
            Results DataFrame
        """
        logger.info("Benchmarking context length impact...")

        context_configs = [
            {"type": "snippet", "tokens": 128, "name": "Snippet Only"},
            {"type": "chunk", "tokens": 512, "name": "Single Chunk"},
            {"type": "chunk_with_context", "tokens": 512, "context_tokens": 128, "name": "Chunk + Context"},
            {"type": "full_doc", "tokens": 2048, "name": "Full Document"},
        ]

        results = []

        for config in context_configs:
            logger.info(f"Testing: {config['name']}")

            # Prepare documents with specified context
            processed_corpus = self._prepare_corpus_with_context(
                corpus=corpus,
                config=config
            )

            # Evaluate
            metrics = await self.retriever.evaluate_corpus(
                corpus=processed_corpus,
                queries=queries,
                qrels=qrels
            )

            results.append({
                "configuration": config["name"],
                "context_type": config["type"],
                "tokens": config["tokens"],
                "context_tokens": config.get("context_tokens", 0),
                "ndcg_at_10": metrics["NDCG@10"],
                "recall_at_10": metrics["Recall@10"],
            })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "context_length.csv", index=False)

        return df

    async def benchmark_matryoshka_embeddings(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        base_model: str = "text-embedding-3-large"
    ) -> pd.DataFrame:
        """
        Benchmark Matryoshka embeddings (multi-resolution)

        Matryoshka embeddings allow using different dimensions from same model:
        - 3072-dim: Full quality
        - 1536-dim: 2x compression
        - 768-dim: 4x compression
        - 256-dim: 12x compression

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            base_model: Base embedding model

        Returns:
            Results DataFrame
        """
        logger.info("Benchmarking Matryoshka embeddings...")

        # Different dimension truncations
        dimensions = [3072, 1536, 768, 512, 256, 128]

        results = []

        for dim in dimensions:
            logger.info(f"Testing dimension: {dim}")

            # Use truncated embeddings
            metrics = await self.retriever.evaluate_with_dimension(
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                model=base_model,
                output_dimension=dim
            )

            # Calculate storage
            storage_mb = self._calculate_storage(
                num_vectors=len(corpus),
                embedding_dim=dim
            )

            compression_ratio = 3072 / dim

            results.append({
                "dimension": dim,
                "compression_ratio": compression_ratio,
                "storage_mb": storage_mb,
                "relative_storage": dim / 3072,
                "ndcg_at_10": metrics["NDCG@10"],
                "recall_at_10": metrics["Recall@10"],
            })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "matryoshka_embeddings.csv", index=False)

        return df

    async def _test_indexing_strategy(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        strategy: IndexingStrategy
    ) -> BenchmarkResult:
        """Test specific indexing strategy"""
        import time

        start_time = time.time()

        if strategy == IndexingStrategy.STANDARD:
            # Standard chunking and indexing
            processed_docs = self._standard_chunking(corpus)

        elif strategy == IndexingStrategy.PARENT_CHILD:
            # Index summaries, retrieve full docs
            processed_docs = self._parent_child_indexing(corpus)

        elif strategy == IndexingStrategy.LATE_CHUNKING:
            # Embed full doc, then chunk
            processed_docs = self._late_chunking(corpus)

        elif strategy == IndexingStrategy.MULTI_REPRESENTATION:
            # Multiple representations per doc
            processed_docs = self._multi_representation(corpus)

        indexing_time = time.time() - start_time

        # Evaluate
        start_time = time.time()
        metrics = await self.retriever.evaluate_corpus(
            corpus=processed_docs,
            queries=queries,
            qrels=qrels
        )
        latency = (time.time() - start_time) / len(queries) * 1000

        return BenchmarkResult(
            test_name="multi_representation_indexing",
            configuration={"strategy": strategy.value},
            ndcg_at_10=metrics["NDCG@10"],
            recall_at_10=metrics["Recall@10"],
            latency_ms=latency,
            storage_mb=0,  # Calculate based on strategy
            indexing_time_s=indexing_time,
            metadata={"strategy": strategy.value}
        )

    def _standard_chunking(self, corpus: Dict) -> Dict:
        """Standard chunking strategy"""
        # Placeholder
        return corpus

    def _parent_child_indexing(self, corpus: Dict) -> Dict:
        """Parent-child indexing"""
        # Placeholder - would generate summaries and link to full docs
        return corpus

    def _late_chunking(self, corpus: Dict) -> Dict:
        """Late chunking strategy"""
        # Placeholder - embed full doc, then split embedding
        return corpus

    def _multi_representation(self, corpus: Dict) -> Dict:
        """Multi-representation indexing"""
        # Placeholder - create multiple embeddings per doc
        return corpus

    def _prepare_corpus_with_context(
        self,
        corpus: Dict,
        config: Dict
    ) -> Dict:
        """Prepare corpus with specified context"""
        # Placeholder
        return corpus

    def _calculate_storage(
        self,
        num_vectors: int,
        embedding_dim: int,
        bytes_per_float: int = 4
    ) -> float:
        """Calculate storage in MB"""
        total_bytes = num_vectors * embedding_dim * bytes_per_float
        return total_bytes / (1024 * 1024)

    def plot_hybrid_search_comparison(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """Plot hybrid search results"""
        plt.figure(figsize=(12, 6))

        x = np.arange(len(results_df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(
            x - width/2,
            results_df["ndcg_at_10"] * 100,
            width,
            label="nDCG@10",
            alpha=0.8
        )

        ax.bar(
            x + width/2,
            results_df["recall_at_10"] * 100,
            width,
            label="Recall@10",
            alpha=0.8
        )

        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title(
            "Hybrid Search: Dense vs Sparse vs Combined",
            fontsize=16,
            fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["configuration"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def plot_index_type_tradeoffs(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """Plot index type tradeoffs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Quality vs Latency
        ax1.scatter(
            results_df["search_latency_ms"],
            results_df["ndcg_at_10"] * 100,
            s=200,
            alpha=0.6
        )

        for idx, row in results_df.iterrows():
            ax1.annotate(
                row["index_type"],
                (row["search_latency_ms"], row["ndcg_at_10"] * 100),
                xytext=(5, 5),
                textcoords="offset points"
            )

        ax1.set_xlabel("Search Latency (ms)", fontsize=12)
        ax1.set_ylabel("nDCG@10 (%)", fontsize=12)
        ax1.set_title("Quality vs Latency", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Indexing Time vs Memory
        ax2.scatter(
            results_df["indexing_time_s"],
            results_df["memory_mb"],
            s=200,
            alpha=0.6,
            color="green"
        )

        for idx, row in results_df.iterrows():
            ax2.annotate(
                row["index_type"],
                (row["indexing_time_s"], row["memory_mb"]),
                xytext=(5, 5),
                textcoords="offset points"
            )

        ax2.set_xlabel("Indexing Time (s)", fontsize=12)
        ax2.set_ylabel("Memory (MB)", fontsize=12)
        ax2.set_title("Indexing Cost", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()
