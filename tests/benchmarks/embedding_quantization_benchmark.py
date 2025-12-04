"""
Embedding Quantization Benchmark
Tests quality vs storage cost tradeoffs for different quantization methods

Quantization Methods:
- Float32 (baseline, 1x storage)
- Float16 (2x compression)
- Int8 (4x compression)
- Binary (32x compression for 1024-dim embeddings)
- Product Quantization (10-100x compression)
"""
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Quantization types"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    BINARY = "binary"
    PRODUCT_QUANTIZATION = "pq"


@dataclass
class QuantizationResult:
    """Result from quantization test"""
    embedding_model: str
    quantization_type: QuantizationType
    compression_ratio: float  # e.g., 4.0 for 4x compression
    storage_mb: float
    relative_storage: float  # Relative to float32 baseline
    ndcg_at_10: float
    recall_at_10: float
    map_at_10: float
    latency_ms: float  # Search latency
    quality_loss_pct: float  # % loss compared to float32


class EmbeddingQuantizationBenchmark:
    """
    Benchmark different embedding quantization methods

    Tests:
    1. Float32 (baseline)
    2. Float16 (half precision)
    3. Int8 (8-bit quantization)
    4. Binary (1-bit quantization)
    5. Product Quantization (PQ)
    """

    def __init__(
        self,
        retriever,
        embedding_models: List[str] = None,
        num_documents: int = 100000,
        results_dir: str = "./quantization_results"
    ):
        """
        Initialize benchmark

        Args:
            retriever: Document retriever
            embedding_models: Models to test
            num_documents: Number of documents for storage calculation
            results_dir: Directory to save results
        """
        self.retriever = retriever
        self.num_documents = num_documents

        if embedding_models is None:
            embedding_models = [
                "voyage-context-3",  # 1024-dim
                "cohere-v4",  # Context-aware
                "text-embedding-3-large",  # 3072-dim
                "text-embedding-3-small",  # 1536-dim
            ]

        self.embedding_models = embedding_models
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def test_all_quantization_methods(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict
    ) -> pd.DataFrame:
        """
        Test all quantization methods across all models

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments

        Returns:
            DataFrame with comprehensive results
        """
        logger.info("Testing all quantization methods...")

        results = []

        for embedding_model in self.embedding_models:
            logger.info(f"Testing model: {embedding_model}")

            embedding_dim = self._get_embedding_dim(embedding_model)

            # Get baseline (float32) performance
            baseline_metrics = await self._test_quantization(
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                embedding_model=embedding_model,
                quantization_type=QuantizationType.FLOAT32
            )

            baseline_ndcg = baseline_metrics["ndcg_at_10"]

            # Test each quantization method
            for quant_type in QuantizationType:
                logger.info(f"  Quantization: {quant_type.value}")

                metrics = await self._test_quantization(
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    embedding_model=embedding_model,
                    quantization_type=quant_type
                )

                # Calculate compression and storage
                compression_ratio = self._get_compression_ratio(
                    quant_type,
                    embedding_dim
                )

                storage_mb = self._calculate_storage(
                    num_vectors=self.num_documents,
                    embedding_dim=embedding_dim,
                    quantization_type=quant_type
                )

                relative_storage = 1.0 / compression_ratio

                # Quality loss
                quality_loss_pct = (
                    (baseline_ndcg - metrics["ndcg_at_10"]) / baseline_ndcg * 100
                    if baseline_ndcg > 0 else 0
                )

                results.append(QuantizationResult(
                    embedding_model=embedding_model,
                    quantization_type=quant_type,
                    compression_ratio=compression_ratio,
                    storage_mb=storage_mb,
                    relative_storage=relative_storage,
                    ndcg_at_10=metrics["ndcg_at_10"],
                    recall_at_10=metrics["recall_at_10"],
                    map_at_10=metrics["map_at_10"],
                    latency_ms=metrics["latency_ms"],
                    quality_loss_pct=quality_loss_pct
                ))

        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in results])

        # Save results
        df.to_csv(self.results_dir / "quantization_results.csv", index=False)

        return df

    async def _test_quantization(
        self,
        corpus: Dict[str, Dict],
        queries: Dict[str, str],
        qrels: Dict,
        embedding_model: str,
        quantization_type: QuantizationType
    ) -> Dict[str, float]:
        """
        Test specific quantization method

        Args:
            corpus: Document corpus
            queries: Test queries
            qrels: Relevance judgments
            embedding_model: Embedding model
            quantization_type: Quantization type

        Returns:
            Evaluation metrics
        """
        # Generate embeddings
        embeddings = await self.retriever.generate_embeddings(
            documents=corpus,
            model=embedding_model
        )

        # Quantize embeddings
        quantized_embeddings = self._quantize_embeddings(
            embeddings=embeddings,
            quantization_type=quantization_type
        )

        # Index quantized embeddings
        await self.retriever.index_embeddings(
            embeddings=quantized_embeddings,
            quantization_type=quantization_type
        )

        # Evaluate retrieval
        import time
        start_time = time.time()

        metrics = await self.retriever.evaluate(
            queries=queries,
            qrels=qrels,
            k_values=[10]
        )

        latency_ms = (time.time() - start_time) / len(queries) * 1000

        metrics["latency_ms"] = latency_ms

        return {
            "ndcg_at_10": metrics.get("NDCG@10", 0.0),
            "recall_at_10": metrics.get("Recall@10", 0.0),
            "map_at_10": metrics.get("MAP@10", 0.0),
            "latency_ms": latency_ms
        }

    def _quantize_embeddings(
        self,
        embeddings: np.ndarray,
        quantization_type: QuantizationType
    ) -> np.ndarray:
        """
        Quantize embeddings

        Args:
            embeddings: Original embeddings (float32)
            quantization_type: Quantization method

        Returns:
            Quantized embeddings
        """
        if quantization_type == QuantizationType.FLOAT32:
            # No quantization
            return embeddings

        elif quantization_type == QuantizationType.FLOAT16:
            # Convert to float16
            return embeddings.astype(np.float16)

        elif quantization_type == QuantizationType.INT8:
            # Scale to int8 range
            min_val = embeddings.min()
            max_val = embeddings.max()

            # Normalize to [0, 255]
            normalized = (embeddings - min_val) / (max_val - min_val) * 255
            quantized = normalized.astype(np.int8)

            return quantized

        elif quantization_type == QuantizationType.BINARY:
            # Convert to binary (sign bit)
            binary = (embeddings > 0).astype(np.uint8)
            return binary

        elif quantization_type == QuantizationType.PRODUCT_QUANTIZATION:
            # Simplified PQ (would use faiss in production)
            # Divide embeddings into subvectors and quantize each
            return self._product_quantize(embeddings, n_subvectors=8, n_bits=8)

        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

    def _product_quantize(
        self,
        embeddings: np.ndarray,
        n_subvectors: int = 8,
        n_bits: int = 8
    ) -> np.ndarray:
        """
        Product Quantization

        Args:
            embeddings: Original embeddings
            n_subvectors: Number of subvectors
            n_bits: Bits per subvector

        Returns:
            Quantized embeddings
        """
        # Placeholder - in production use faiss.IndexPQ
        # For now, just do basic quantization
        return self._quantize_embeddings(embeddings, QuantizationType.INT8)

    def _get_compression_ratio(
        self,
        quantization_type: QuantizationType,
        embedding_dim: int
    ) -> float:
        """
        Get compression ratio for quantization type

        Args:
            quantization_type: Quantization type
            embedding_dim: Embedding dimension

        Returns:
            Compression ratio (e.g., 4.0 for 4x compression)
        """
        if quantization_type == QuantizationType.FLOAT32:
            return 1.0
        elif quantization_type == QuantizationType.FLOAT16:
            return 2.0
        elif quantization_type == QuantizationType.INT8:
            return 4.0
        elif quantization_type == QuantizationType.BINARY:
            # 32x for float32, but depends on dim
            return 32.0
        elif quantization_type == QuantizationType.PRODUCT_QUANTIZATION:
            # Typical PQ gives 8-32x compression
            return 16.0
        else:
            return 1.0

    def _calculate_storage(
        self,
        num_vectors: int,
        embedding_dim: int,
        quantization_type: QuantizationType
    ) -> float:
        """
        Calculate storage in MB

        Args:
            num_vectors: Number of vectors
            embedding_dim: Embedding dimension
            quantization_type: Quantization type

        Returns:
            Storage in MB
        """
        bytes_per_dimension = {
            QuantizationType.FLOAT32: 4,
            QuantizationType.FLOAT16: 2,
            QuantizationType.INT8: 1,
            QuantizationType.BINARY: 1/8,  # 1 bit per dimension
            QuantizationType.PRODUCT_QUANTIZATION: 0.25,  # Approximate
        }

        bytes_per_dim = bytes_per_dimension.get(quantization_type, 4)
        total_bytes = num_vectors * embedding_dim * bytes_per_dim

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _get_embedding_dim(self, model_name: str) -> int:
        """Get embedding dimension for model"""
        dimensions = {
            "voyage-context-3": 1024,
            "cohere-v4": 1024,
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "all-MiniLM-L6-v2": 384,
        }
        return dimensions.get(model_name, 384)

    def plot_quality_vs_storage(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot quality vs storage cost (like Image 2)

        Args:
            results_df: Results DataFrame
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define markers for quantization types
        markers = {
            QuantizationType.FLOAT32: "o",
            QuantizationType.FLOAT16: "s",
            QuantizationType.INT8: "^",
            QuantizationType.BINARY: "D",
            QuantizationType.PRODUCT_QUANTIZATION: "v",
        }

        colors = plt.cm.Set1(np.linspace(0, 1, len(results_df["embedding_model"].unique())))

        for i, model in enumerate(results_df["embedding_model"].unique()):
            model_data = results_df[results_df["embedding_model"] == model]

            for quant_type in QuantizationType:
                quant_data = model_data[model_data["quantization_type"] == quant_type]

                if len(quant_data) == 0:
                    continue

                ax.scatter(
                    quant_data["relative_storage"],
                    quant_data["ndcg_at_10"] * 100,  # Convert to percentage
                    marker=markers[quant_type],
                    s=200,
                    alpha=0.7,
                    color=colors[i],
                    label=f"{model}" if quant_type == QuantizationType.FLOAT32 else ""
                )

                # Annotate with chunk size
                for _, row in quant_data.iterrows():
                    ax.annotate(
                        f"({int(row['compression_ratio'])}x)",
                        (row["relative_storage"], row["ndcg_at_10"] * 100),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.7
                    )

        # Add grid lines
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')

        # Labels and title
        ax.set_xlabel("Relative Storage Costs", fontsize=14)
        ax.set_ylabel("Retrieval Quality (NDCG@10)", fontsize=14)
        ax.set_title(
            "Quality vs Storage Tradeoff: Embedding Quantization",
            fontsize=16,
            fontweight="bold"
        )

        # Create custom legend for quantization types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='float32'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=10, label='float16 (2x)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=10, label='int8 (4x)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                   markersize=10, label='binary (32x)'),
        ]

        ax.legend(
            handles=legend_elements,
            title="Embedding Quantization",
            loc='lower right',
            fontsize=10
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def plot_pareto_frontier(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot Pareto frontier of quality vs storage

        Args:
            results_df: Results DataFrame
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        for model in results_df["embedding_model"].unique():
            model_data = results_df[results_df["embedding_model"] == model]

            # Sort by storage
            model_data = model_data.sort_values("relative_storage")

            # Plot line connecting points
            ax.plot(
                model_data["relative_storage"],
                model_data["ndcg_at_10"] * 100,
                marker="o",
                markersize=8,
                linewidth=2,
                label=model,
                alpha=0.7
            )

        ax.set_xscale('log')
        ax.set_xlabel("Relative Storage Cost (log scale)", fontsize=14)
        ax.set_ylabel("Retrieval Quality (NDCG@10)", fontsize=14)
        ax.set_title(
            "Pareto Frontier: Quality vs Storage",
            fontsize=16,
            fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def plot_quality_degradation(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot quality degradation by quantization type

        Args:
            results_df: Results DataFrame
            save_path: Path to save plot
        """
        # Pivot for heatmap
        pivot_df = results_df.pivot(
            index="embedding_model",
            columns="quantization_type",
            values="quality_loss_pct"
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn_r",  # Red for high loss, green for low loss
            cbar_kws={"label": "Quality Loss (%)"},
            vmin=0,
            vmax=20
        )

        plt.title(
            "Quality Degradation by Quantization Method",
            fontsize=16,
            fontweight="bold"
        )
        plt.xlabel("Quantization Type", fontsize=12)
        plt.ylabel("Embedding Model", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved: {save_path}")

        plt.show()

    def generate_recommendations(
        self,
        results_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate recommendations based on results

        Args:
            results_df: Results DataFrame

        Returns:
            Recommendations dictionary
        """
        recommendations = {
            "best_overall": {},
            "best_compression": {},
            "quality_threshold": {},
            "use_cases": {}
        }

        # Find best overall (highest quality per storage unit)
        results_df["efficiency"] = (
            results_df["ndcg_at_10"] / results_df["relative_storage"]
        )

        best_overall = results_df.nlargest(1, "efficiency").iloc[0]
        recommendations["best_overall"] = {
            "model": best_overall["embedding_model"],
            "quantization": best_overall["quantization_type"],
            "ndcg": float(best_overall["ndcg_at_10"]),
            "compression": float(best_overall["compression_ratio"])
        }

        # Best for high compression (>10x)
        high_compression = results_df[results_df["compression_ratio"] >= 10]
        if len(high_compression) > 0:
            best_compressed = high_compression.nlargest(1, "ndcg_at_10").iloc[0]
            recommendations["best_compression"] = {
                "model": best_compressed["embedding_model"],
                "quantization": best_compressed["quantization_type"],
                "ndcg": float(best_compressed["ndcg_at_10"]),
                "compression": float(best_compressed["compression_ratio"]),
                "quality_loss": float(best_compressed["quality_loss_pct"])
            }

        # Use case recommendations
        recommendations["use_cases"] = {
            "high_quality": "Use float32 or float16 for maximum quality",
            "balanced": "Use int8 for 4x compression with <5% quality loss",
            "extreme_compression": "Use binary for 32x compression (10-15% quality loss)",
            "cost_sensitive": "Use int8 or Product Quantization for best cost/quality"
        }

        return recommendations
