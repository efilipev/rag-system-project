"""
Test Embedding Models on BEIR Datasets

This script evaluates all embedding models on BEIR benchmark datasets
to find the best performing model across different retrieval tasks.

Usage:
    python tests/benchmarks/test_embeddings_on_beir.py --quick  # Quick test (3 small datasets)
    python tests/benchmarks/test_embeddings_on_beir.py --full   # Full BEIR benchmark
    python tests/benchmarks/test_embeddings_on_beir.py --models MiniLM BGE  # Specific models
"""
import asyncio
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.embedding_model_comparison_benchmark import (
    EMBEDDING_MODELS,
    EmbeddingModelConfig
)
from tests.benchmarks.beir_evaluation import BEIRBenchmark, BEIRConfig
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BEIRRetriever:
    """Simple retriever for BEIR evaluation using sentence-transformers"""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self._corpus_embeddings = None
        self._corpus_ids = None

    async def retrieve(
        self,
        query: str,
        corpus: Dict[str, Dict[str, str]],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, float]:
        """
        Retrieve top-k documents for query

        Returns:
            Dict mapping doc_id to similarity score
        """
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Index corpus if not already done
        if self._corpus_embeddings is None or self._corpus_ids is None:
            self._index_corpus(corpus)

        # Compute similarities
        similarities = np.dot(self._corpus_embeddings, query_embedding)

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        results = {}
        for idx in top_k_indices:
            doc_id = self._corpus_ids[idx]
            score = float(similarities[idx])
            results[doc_id] = score

        return results

    def _index_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """Index the corpus for retrieval"""
        logger.info(f"Indexing {len(corpus)} documents...")

        self._corpus_ids = list(corpus.keys())

        # Create document texts (title + text)
        texts = []
        for doc_id in self._corpus_ids:
            doc = corpus[doc_id]
            text = doc.get("title", "") + " " + doc.get("text", "")
            texts.append(text)

        # Encode all documents
        self._corpus_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Normalize for cosine similarity
        norms = np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True)
        self._corpus_embeddings = self._corpus_embeddings / norms

        logger.info(f"Corpus indexed: {self._corpus_embeddings.shape}")


class EmbeddingBEIRBenchmark:
    """Benchmark embeddings on BEIR datasets"""

    def __init__(
        self,
        models: List[EmbeddingModelConfig] = None,
        data_dir: str = "./beir_data",
        results_dir: str = "./beir_embedding_results"
    ):
        self.models = models or EMBEDDING_MODELS
        self.beir_benchmark = BEIRBenchmark(
            data_dir=data_dir,
            results_dir=results_dir
        )
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_single_model(
        self,
        model_config: EmbeddingModelConfig,
        datasets: List[str],
        k_values: List[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate a single embedding model on BEIR datasets

        Args:
            model_config: Model configuration
            datasets: List of BEIR dataset names
            k_values: K values for metrics

        Returns:
            DataFrame with results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_config.name}")
        logger.info(f"Model ID: {model_config.model_id}")
        logger.info(f"{'='*80}\n")

        k_values = k_values or [1, 3, 5, 10, 20, 100]

        try:
            # Load model
            logger.info(f"Loading model: {model_config.model_id}")
            model = SentenceTransformer(model_config.model_id)
            logger.info(f"Model loaded successfully")

            # Create retriever
            retriever = BEIRRetriever(model)

            # Evaluate on each dataset
            all_results = []

            for dataset_name in datasets:
                logger.info(f"\n--- Testing on {dataset_name} ---")

                try:
                    config = BEIRConfig(
                        dataset_name=dataset_name,
                        split="test",
                        k_values=k_values
                    )

                    result = await self.beir_benchmark.evaluate_technique(
                        technique_name=model_config.name,
                        retriever=retriever,
                        config=config,
                        parameters={"model_id": model_config.model_id}
                    )

                    # Convert to dict and add model info
                    result_dict = {
                        "model_name": model_config.name,
                        "model_id": model_config.model_id,
                        "dimension": model_config.dimension,
                        "dataset": result.dataset,
                        "num_queries": result.num_queries,
                        "num_documents": result.num_documents,
                        "execution_time": result.execution_time,
                    }

                    # Add metrics (filter out non-scalar values)
                    try:
                        if hasattr(result.metrics, 'items'):
                            for key, value in result.metrics.items():
                                if isinstance(value, (int, float, str, bool)) or value is None:
                                    result_dict[key] = value
                        else:
                            logger.warning(f"result.metrics is not a dict: {type(result.metrics)}")
                    except Exception as e:
                        logger.error(f"Error processing metrics: {e}, type: {type(result.metrics)}")
                        raise

                    all_results.append(result_dict)

                    ndcg_10 = result.metrics.get('NDCG@10', 0) if hasattr(result.metrics, 'get') else 0
                    logger.info(
                        f"✓ {dataset_name}: "
                        f"NDCG@10={ndcg_10:.4f}, "
                        f"Time={result.execution_time:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"✗ Failed on {dataset_name}: {e}", exc_info=True)
                    continue

            # Create DataFrame
            df = pd.DataFrame(all_results)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"{model_config.name.replace(' ', '_')}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"\n✓ Results saved: {output_file}")

            return df

        except Exception as e:
            logger.error(f"✗ Model evaluation failed: {e}", exc_info=True)
            return pd.DataFrame()

    async def evaluate_all_models(
        self,
        datasets: List[str],
        k_values: List[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate all embedding models

        Args:
            datasets: List of BEIR dataset names
            k_values: K values for metrics

        Returns:
            Combined DataFrame with all results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BEIR EMBEDDING BENCHMARK")
        logger.info(f"{'='*80}")
        logger.info(f"Models: {len(self.models)}")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"K values: {k_values or [1, 3, 5, 10, 20, 100]}")
        logger.info(f"{'='*80}\n")

        all_results = []

        for i, model_config in enumerate(self.models, 1):
            logger.info(f"\n[{i}/{len(self.models)}] Processing {model_config.name}")

            df = await self.evaluate_single_model(
                model_config=model_config,
                datasets=datasets,
                k_values=k_values
            )

            if not df.empty:
                all_results.append(df)

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)

            # Save combined results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"all_models_comparison_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            logger.info(f"\n✓ All results saved: {output_file}")

            # Generate summary
            self._generate_summary(combined_df)

            return combined_df
        else:
            logger.error("No results collected!")
            return pd.DataFrame()

    def _generate_summary(self, df: pd.DataFrame):
        """Generate and print summary statistics"""
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY REPORT")
        logger.info(f"{'='*80}\n")

        # Overall best by NDCG@10
        if 'NDCG@10' in df.columns:
            avg_ndcg = df.groupby('model_name')['NDCG@10'].mean().sort_values(ascending=False)

            logger.info("🏆 TOP 5 MODELS BY AVERAGE NDCG@10:\n")
            for i, (model, score) in enumerate(avg_ndcg.head(5).items(), 1):
                logger.info(f"  {i}. {model}: {score:.4f}")

            logger.info("\n📊 PERFORMANCE BY DATASET:\n")
            for dataset in df['dataset'].unique():
                dataset_df = df[df['dataset'] == dataset]
                best = dataset_df.nlargest(1, 'NDCG@10').iloc[0]
                logger.info(
                    f"  {dataset}:"
                    f"\n    Best: {best['model_name']} (NDCG@10: {best['NDCG@10']:.4f})"
                )

        # Speed analysis
        if 'execution_time' in df.columns:
            logger.info("\n⚡ FASTEST MODELS (avg time per dataset):\n")
            avg_time = df.groupby('model_name')['execution_time'].mean().sort_values()
            for i, (model, time_s) in enumerate(avg_time.head(5).items(), 1):
                logger.info(f"  {i}. {model}: {time_s:.2f}s")

        logger.info(f"\n{'='*80}\n")


# CLI Interface

async def run_quick_benchmark():
    """Run quick benchmark on small datasets"""
    logger.info("Running QUICK BEIR benchmark (3 small datasets)")

    # Use small, fast datasets
    datasets = ["nfcorpus", "scifact", "fiqa"]

    # Select 3 fast models for quick test
    test_models = [
        model for model in EMBEDDING_MODELS
        if model.name in [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "BGE-base-en-v1.5"
        ]
    ]

    benchmark = EmbeddingBEIRBenchmark(models=test_models)
    results = await benchmark.evaluate_all_models(datasets=datasets)

    return results


async def run_full_benchmark():
    """Run full BEIR benchmark"""
    logger.info("Running FULL BEIR benchmark (7 datasets)")

    # Standard BEIR benchmark datasets
    datasets = [
        "msmarco",
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "scifact"
    ]

    benchmark = EmbeddingBEIRBenchmark(models=EMBEDDING_MODELS)
    results = await benchmark.evaluate_all_models(datasets=datasets)

    return results


async def run_custom_benchmark(model_names: List[str], datasets: List[str] = None):
    """Run benchmark with specific models"""
    logger.info(f"Running custom benchmark with specified models")

    # Filter models by name
    selected_models = [
        model for model in EMBEDDING_MODELS
        if any(name.lower() in model.name.lower() for name in model_names)
    ]

    if not selected_models:
        logger.error(f"No models matched: {model_names}")
        logger.info("Available models:")
        for model in EMBEDDING_MODELS:
            logger.info(f"  - {model.name}")
        sys.exit(1)

    logger.info(f"Testing {len(selected_models)} models")

    # Default to small datasets if not specified
    if not datasets:
        datasets = ["nfcorpus", "scifact", "fiqa"]

    benchmark = EmbeddingBEIRBenchmark(models=selected_models)
    results = await benchmark.evaluate_all_models(datasets=datasets)

    return results


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            📊 EMBEDDING MODELS - BEIR BENCHMARK 📊                          ║
║                                                                              ║
║  Evaluate embedding models on standard IR benchmark datasets                ║
║  Using BEIR (Benchmarking IR) framework                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test Embedding Models on BEIR Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/benchmarks/test_embeddings_on_beir.py --quick
  python tests/benchmarks/test_embeddings_on_beir.py --full
  python tests/benchmarks/test_embeddings_on_beir.py --models MiniLM BGE --datasets nfcorpus scifact
        """
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (3 models, 3 small datasets, ~10-15 min)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full benchmark (all models, 7 datasets, ~2-3 hours)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Test specific models (partial name matching)"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa",
            "fiqa", "scifact", "scidocs", "fever", "climate-fever"
        ],
        help="Specific BEIR datasets to use"
    )

    args = parser.parse_args()

    print_banner()

    try:
        if args.quick:
            await run_quick_benchmark()
        elif args.full:
            await run_full_benchmark()
        elif args.models:
            await run_custom_benchmark(args.models, args.datasets)
        else:
            # Default: quick test
            await run_quick_benchmark()

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
