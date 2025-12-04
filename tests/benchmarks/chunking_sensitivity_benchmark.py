"""
Chunking Sensitivity Benchmark
Tests how different chunk sizes and strategies affect retrieval quality

Based on research showing context-aware embeddings are less sensitive to chunk size
"""
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""
    chunk_size: int  # In tokens
    chunk_overlap: int  # In tokens
    strategy: str  # "fixed", "sentence", "semantic", "paragraph"
    min_chunk_size: int = 50
    max_chunk_size: int = 2000


@dataclass
class ChunkingResult:
    """Result from chunking sensitivity test"""
    chunk_size: int
    chunk_overlap: int
    strategy: str
    embedding_model: str
    ndcg_at_10: float
    recall_at_10: float
    num_chunks: int
    avg_chunk_tokens: float
    variance: float  # Performance variance across different sizes
    storage_mb: float


class ChunkingSensitivityBenchmark:
    """
    Test how chunking affects retrieval quality

    Tests:
    1. Chunk size sensitivity (64, 128, 256, 512, 1024 tokens)
    2. Chunk overlap impact (0%, 10%, 25%, 50%)
    3. Chunking strategies (fixed, sentence, semantic, paragraph)
    4. Context-aware vs standard embeddings
    """

    # Standard chunk sizes to test (in tokens)
    CHUNK_SIZES = [64, 128, 256, 512, 1024, 2048]

    # Overlap percentages to test
    OVERLAP_PERCENTAGES = [0, 10, 25, 50]

    def __init__(
        self,
        retriever,
        embedding_models: List[str] = None,
        results_dir: str = "./chunking_results"
    ):
        """
        Initialize benchmark

        Args:
            retriever: Document retriever
            embedding_models: List of embedding models to compare
            results_dir: Directory to save results
        """
        self.retriever = retriever

        if embedding_models is None:
            embedding_models = [
                "voyage-3-large",
                "voyage-context-3",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "all-MiniLM-L6-v2"
            ]

        self.embedding_models = embedding_models
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def test_chunk_size_sensitivity(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        overlap_percentage: int = 10
    ) -> pd.DataFrame:
        """
        Test how chunk size affects retrieval quality

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            overlap_percentage: Chunk overlap percentage

        Returns:
            DataFrame with results for each chunk size
        """
        logger.info("Testing chunk size sensitivity...")

        results = []

        for embedding_model in self.embedding_models:
            logger.info(f"Testing model: {embedding_model}")

            for chunk_size in self.CHUNK_SIZES:
                overlap = int(chunk_size * overlap_percentage / 100)

                logger.info(f"  Chunk size: {chunk_size}, overlap: {overlap}")

                # Chunk documents
                chunked_corpus = await self._chunk_corpus(
                    corpus=corpus,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    strategy="fixed"
                )

                # Index with current embedding model
                await self.retriever.index_documents(
                    documents=chunked_corpus,
                    embedding_model=embedding_model
                )

                # Evaluate retrieval
                metrics = await self.retriever.evaluate(
                    queries=queries,
                    qrels=qrels,
                    k_values=[10]
                )

                # Calculate storage
                storage_mb = self._calculate_storage(
                    num_chunks=len(chunked_corpus),
                    embedding_dim=self._get_embedding_dim(embedding_model)
                )

                results.append(ChunkingResult(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    strategy="fixed",
                    embedding_model=embedding_model,
                    ndcg_at_10=metrics["NDCG@10"],
                    recall_at_10=metrics["Recall@10"],
                    num_chunks=len(chunked_corpus),
                    avg_chunk_tokens=chunk_size,
                    variance=0.0,  # Will calculate later
                    storage_mb=storage_mb
                ))

        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in results])

        # Calculate variance for each model
        for model in self.embedding_models:
            model_df = df[df["embedding_model"] == model]
            variance = model_df["ndcg_at_10"].var()
            df.loc[df["embedding_model"] == model, "variance"] = variance

        # Save results
        df.to_csv(self.results_dir / "chunk_size_sensitivity.csv", index=False)

        return df

    async def test_overlap_sensitivity(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        chunk_size: int = 512
    ) -> pd.DataFrame:
        """
        Test how chunk overlap affects retrieval quality

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            chunk_size: Fixed chunk size to use

        Returns:
            DataFrame with results for each overlap percentage
        """
        logger.info("Testing chunk overlap sensitivity...")

        results = []

        for embedding_model in self.embedding_models:
            for overlap_pct in self.OVERLAP_PERCENTAGES:
                overlap = int(chunk_size * overlap_pct / 100)

                logger.info(
                    f"Model: {embedding_model}, "
                    f"Overlap: {overlap_pct}% ({overlap} tokens)"
                )

                # Chunk documents
                chunked_corpus = await self._chunk_corpus(
                    corpus=corpus,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    strategy="fixed"
                )

                # Index and evaluate
                await self.retriever.index_documents(
                    documents=chunked_corpus,
                    embedding_model=embedding_model
                )

                metrics = await self.retriever.evaluate(
                    queries=queries,
                    qrels=qrels,
                    k_values=[10]
                )

                storage_mb = self._calculate_storage(
                    num_chunks=len(chunked_corpus),
                    embedding_dim=self._get_embedding_dim(embedding_model)
                )

                results.append({
                    "embedding_model": embedding_model,
                    "chunk_size": chunk_size,
                    "overlap_pct": overlap_pct,
                    "overlap_tokens": overlap,
                    "ndcg_at_10": metrics["NDCG@10"],
                    "recall_at_10": metrics["Recall@10"],
                    "num_chunks": len(chunked_corpus),
                    "storage_mb": storage_mb,
                    "redundancy_factor": 1 + (overlap_pct / 100)
                })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "overlap_sensitivity.csv", index=False)

        return df

    async def test_chunking_strategies(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        target_chunk_size: int = 512
    ) -> pd.DataFrame:
        """
        Compare different chunking strategies

        Strategies:
        - Fixed: Fixed token count
        - Sentence: Sentence boundaries
        - Semantic: Semantic similarity breaks
        - Paragraph: Paragraph boundaries
        - Sliding: Sliding window

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            target_chunk_size: Target chunk size

        Returns:
            DataFrame comparing strategies
        """
        logger.info("Testing chunking strategies...")

        strategies = ["fixed", "sentence", "semantic", "paragraph", "sliding"]
        results = []

        for embedding_model in self.embedding_models:
            for strategy in strategies:
                logger.info(f"Model: {embedding_model}, Strategy: {strategy}")

                # Chunk with strategy
                chunked_corpus, metadata = await self._chunk_corpus_with_strategy(
                    corpus=corpus,
                    target_size=target_chunk_size,
                    strategy=strategy
                )

                # Index and evaluate
                await self.retriever.index_documents(
                    documents=chunked_corpus,
                    embedding_model=embedding_model
                )

                metrics = await self.retriever.evaluate(
                    queries=queries,
                    qrels=qrels,
                    k_values=[10]
                )

                results.append({
                    "embedding_model": embedding_model,
                    "strategy": strategy,
                    "target_chunk_size": target_chunk_size,
                    "avg_chunk_size": metadata["avg_size"],
                    "min_chunk_size": metadata["min_size"],
                    "max_chunk_size": metadata["max_size"],
                    "std_chunk_size": metadata["std_size"],
                    "num_chunks": len(chunked_corpus),
                    "ndcg_at_10": metrics["NDCG@10"],
                    "recall_at_10": metrics["Recall@10"],
                })

        df = pd.DataFrame(results)
        df.to_csv(self.results_dir / "strategy_comparison.csv", index=False)

        return df

    async def test_context_aware_vs_standard(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict
    ) -> pd.DataFrame:
        """
        Compare context-aware vs standard embeddings across chunk sizes

        Context-aware models (like voyage-context-3) should be less sensitive
        to chunk size variations

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments

        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing context-aware vs standard embeddings...")

        # Define model pairs
        model_pairs = {
            "voyage-3-large": "standard",
            "voyage-context-3": "context_aware",
            "text-embedding-3-large": "standard",
            "text-embedding-3-small": "standard",
        }

        results = []

        for chunk_size in self.CHUNK_SIZES:
            for model_name, model_type in model_pairs.items():
                logger.info(f"Chunk size: {chunk_size}, Model: {model_name}")

                # Chunk documents
                chunked_corpus = await self._chunk_corpus(
                    corpus=corpus,
                    chunk_size=chunk_size,
                    overlap=int(chunk_size * 0.1),
                    strategy="fixed"
                )

                # Index and evaluate
                await self.retriever.index_documents(
                    documents=chunked_corpus,
                    embedding_model=model_name
                )

                metrics = await self.retriever.evaluate(
                    queries=queries,
                    qrels=qrels,
                    k_values=[10]
                )

                results.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "chunk_size": chunk_size,
                    "ndcg_at_10": metrics["NDCG@10"],
                    "recall_at_10": metrics["Recall@10"],
                })

        df = pd.DataFrame(results)

        # Calculate variance for each model
        for model_name in model_pairs.keys():
            model_df = df[df["model_name"] == model_name]
            variance = model_df["ndcg_at_10"].var()
            df.loc[df["model_name"] == model_name, "variance"] = variance

        df.to_csv(self.results_dir / "context_aware_comparison.csv", index=False)

        return df

    async def _chunk_corpus(
        self,
        corpus: Dict[str, Dict],
        chunk_size: int,
        overlap: int,
        strategy: str
    ) -> List[Dict]:
        """Chunk corpus with specified parameters"""
        # Placeholder - implement actual chunking
        # In production, use langchain or custom chunking logic
        chunked_docs = []

        for doc_id, doc in corpus.items():
            text = doc.get("text", "")

            # Simple token-based chunking (placeholder)
            tokens = text.split()

            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]

                if len(chunk_tokens) < chunk_size // 2:
                    continue  # Skip very small chunks

                chunked_docs.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "text": " ".join(chunk_tokens),
                    "parent_doc_id": doc_id,
                    "chunk_index": i // (chunk_size - overlap)
                })

        return chunked_docs

    async def _chunk_corpus_with_strategy(
        self,
        corpus: Dict[str, Dict],
        target_size: int,
        strategy: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Chunk corpus using specific strategy"""
        # Placeholder - implement different strategies
        chunked_docs = await self._chunk_corpus(
            corpus=corpus,
            chunk_size=target_size,
            overlap=int(target_size * 0.1),
            strategy=strategy
        )

        # Calculate metadata
        chunk_sizes = [len(doc["text"].split()) for doc in chunked_docs]

        metadata = {
            "avg_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_size": max(chunk_sizes) if chunk_sizes else 0,
            "std_size": pd.Series(chunk_sizes).std() if chunk_sizes else 0,
        }

        return chunked_docs, metadata

    def _calculate_storage(self, num_chunks: int, embedding_dim: int) -> float:
        """Calculate storage in MB"""
        # 4 bytes per float32
        bytes_per_embedding = embedding_dim * 4
        total_bytes = num_chunks * bytes_per_embedding
        return total_bytes / (1024 * 1024)  # Convert to MB

    def _get_embedding_dim(self, model_name: str) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            "voyage-3-large": 1024,
            "voyage-context-3": 1024,
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "all-MiniLM-L6-v2": 384,
        }
        return dimensions.get(model_name, 384)

    def plot_chunk_size_sensitivity(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot chunk size sensitivity (like Image 1)

        Args:
            results_df: Results from test_chunk_size_sensitivity
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 7))

        for model in results_df["embedding_model"].unique():
            model_data = results_df[results_df["embedding_model"] == model]

            plt.plot(
                model_data["chunk_size"],
                model_data["ndcg_at_10"] * 100,  # Convert to percentage
                marker="o",
                label=model,
                linewidth=2,
                markersize=8
            )

        plt.xlabel("Chunk Size (Tokens)", fontsize=14)
        plt.ylabel("Retrieval Quality (NDCG@10)", fontsize=14)
        plt.title(
            "Chunking Sensitivity: Impact of Chunk Size on Retrieval Quality",
            fontsize=16,
            fontweight="bold"
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def plot_variance_comparison(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot variance comparison between models

        Args:
            results_df: Results DataFrame
            save_path: Path to save plot
        """
        # Calculate variance for each model
        variance_data = results_df.groupby("embedding_model").agg({
            "ndcg_at_10": ["mean", "var", "std"]
        }).round(4)

        variance_data.columns = ["Mean nDCG@10", "Variance", "Std Dev"]
        variance_data = variance_data.sort_values("Variance")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Variance plot
        variance_data["Variance"].plot(
            kind="barh",
            ax=ax1,
            color="steelblue"
        )
        ax1.set_xlabel("Variance", fontsize=12)
        ax1.set_title("Chunk Size Sensitivity (Lower is Better)", fontsize=14)
        ax1.grid(True, alpha=0.3, axis="x")

        # Mean performance plot
        variance_data["Mean nDCG@10"].plot(
            kind="barh",
            ax=ax2,
            color="green"
        )
        ax2.set_xlabel("Mean nDCG@10", fontsize=12)
        ax2.set_title("Average Performance", fontsize=14)
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def generate_report(self, all_results: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive chunking sensitivity report

        Args:
            all_results: Dictionary of all test results

        Returns:
            Report dictionary
        """
        report = {
            "summary": {},
            "best_chunk_sizes": {},
            "least_sensitive_model": None,
            "optimal_overlap": None,
            "best_strategy": None,
            "recommendations": []
        }

        # Find best chunk size for each model
        if "chunk_size" in all_results:
            chunk_df = all_results["chunk_size"]

            for model in chunk_df["embedding_model"].unique():
                model_data = chunk_df[chunk_df["embedding_model"] == model]
                best_row = model_data.nlargest(1, "ndcg_at_10").iloc[0]

                report["best_chunk_sizes"][model] = {
                    "chunk_size": int(best_row["chunk_size"]),
                    "ndcg_at_10": float(best_row["ndcg_at_10"]),
                    "variance": float(best_row["variance"])
                }

            # Find least sensitive model (lowest variance)
            variance_by_model = chunk_df.groupby("embedding_model")["ndcg_at_10"].var()
            report["least_sensitive_model"] = variance_by_model.idxmin()

        # Optimal overlap
        if "overlap" in all_results:
            overlap_df = all_results["overlap"]
            best_overlap = overlap_df.nlargest(1, "ndcg_at_10").iloc[0]
            report["optimal_overlap"] = int(best_overlap["overlap_pct"])

        # Best strategy
        if "strategy" in all_results:
            strategy_df = all_results["strategy"]
            best_strategy = strategy_df.nlargest(1, "ndcg_at_10").iloc[0]
            report["best_strategy"] = best_strategy["strategy"]

        # Recommendations
        if report["least_sensitive_model"]:
            report["recommendations"].append(
                f"Use {report['least_sensitive_model']} for most stable performance across chunk sizes"
            )

        return report
