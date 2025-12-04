"""
BEIR Benchmark Evaluation Framework
Evaluates different RAG techniques on standard IR benchmarks

BEIR Datasets:
- MS MARCO
- TREC-COVID
- NFCorpus
- Natural Questions
- HotpotQA
- FiQA
- SciFact
- And more...

Metrics:
- nDCG@k
- MAP@k
- Recall@k
- MRR@k
- Precision@k
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logger = logging.getLogger(__name__)


@dataclass
class BEIRConfig:
    """Configuration for BEIR evaluation"""
    dataset_name: str
    split: str = "test"
    data_path: Optional[str] = None
    k_values: List[int] = None

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 20, 100]


@dataclass
class EvaluationResult:
    """Result from evaluation"""
    dataset: str
    technique: str
    metrics: Dict[str, float]
    execution_time: float
    num_queries: int
    num_documents: int
    parameters: Dict[str, Any]
    timestamp: str


class RAGTechniqueWrapper:
    """
    Wrapper to adapt RAG techniques to BEIR interface
    """

    def __init__(self, retriever, technique_name: str, parameters: Dict = None):
        """
        Initialize wrapper

        Args:
            retriever: RAG retrieval component
            technique_name: Name of the technique
            parameters: Technique-specific parameters
        """
        self.retriever = retriever
        self.technique_name = technique_name
        self.parameters = parameters or {}

    async def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Search interface compatible with BEIR

        Args:
            corpus: Document corpus {doc_id: {title, text}}
            queries: Queries {query_id: query_text}
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            Results {query_id: {doc_id: score}}
        """
        results = {}

        for query_id, query_text in queries.items():
            # Execute retrieval with current technique
            doc_scores = await self.retriever.retrieve(
                query=query_text,
                corpus=corpus,
                top_k=top_k,
                **self.parameters
            )

            results[query_id] = doc_scores

        return results


class BEIRBenchmark:
    """
    BEIR Benchmark Evaluation
    Compare different RAG techniques on standard datasets
    """

    # Available BEIR datasets
    DATASETS = [
        "msmarco",  # MS MARCO passage ranking
        "trec-covid",  # COVID-19 related queries
        "nfcorpus",  # Nutrition facts
        "nq",  # Natural Questions
        "hotpotqa",  # Multi-hop questions
        "fiqa",  # Financial QA
        "scifact",  # Scientific fact verification
        "scidocs",  # Citation prediction
        "fever",  # Fact extraction and verification
        "climate-fever",  # Climate science claims
        "dbpedia-entity",  # Entity retrieval
        "webis-touche2020",  # Argument retrieval
        "quora",  # Duplicate question detection
    ]

    def __init__(
        self,
        data_dir: str = "./beir_data",
        results_dir: str = "./beir_results"
    ):
        """
        Initialize BEIR benchmark

        Args:
            data_dir: Directory to store BEIR datasets
            results_dir: Directory to store evaluation results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"BEIR Benchmark initialized: {self.data_dir}")

    async def download_dataset(self, dataset_name: str) -> str:
        """
        Download BEIR dataset

        Args:
            dataset_name: Name of dataset

        Returns:
            Path to downloaded dataset
        """
        logger.info(f"Downloading BEIR dataset: {dataset_name}")

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = str(self.data_dir / dataset_name)

        # Download and unzip
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            util.download_and_unzip,
            url,
            str(self.data_dir)
        )

        logger.info(f"Dataset downloaded: {data_path}")
        return data_path

    async def load_dataset(
        self,
        dataset_name: str,
        split: str = "test"
    ) -> tuple[Dict, Dict, Dict]:
        """
        Load BEIR dataset

        Args:
            dataset_name: Dataset name
            split: Data split (train/dev/test)

        Returns:
            Tuple of (corpus, queries, qrels)
        """
        # Download if not exists
        data_path = self.data_dir / dataset_name
        if not data_path.exists():
            data_path = await self.download_dataset(dataset_name)

        logger.info(f"Loading dataset: {dataset_name} ({split})")

        # Load using BEIR loader
        corpus, queries, qrels = GenericDataLoader(
            data_folder=str(data_path)
        ).load(split=split)

        logger.info(
            f"Loaded {len(corpus)} documents, "
            f"{len(queries)} queries, "
            f"{len(qrels)} relevance judgments"
        )

        return corpus, queries, qrels

    async def evaluate_technique(
        self,
        technique_name: str,
        retriever,
        config: BEIRConfig,
        parameters: Dict[str, Any] = None
    ) -> EvaluationResult:
        """
        Evaluate a RAG technique on BEIR dataset

        Args:
            technique_name: Name of technique being evaluated
            retriever: Retrieval component
            config: BEIR configuration
            parameters: Technique-specific parameters

        Returns:
            Evaluation results
        """
        import time

        start_time = time.time()

        logger.info(
            f"Evaluating '{technique_name}' on {config.dataset_name} "
            f"with params: {parameters}"
        )

        # Load dataset
        corpus, queries, qrels = await self.load_dataset(
            config.dataset_name,
            config.split
        )

        # Wrap technique for BEIR interface
        wrapper = RAGTechniqueWrapper(
            retriever=retriever,
            technique_name=technique_name,
            parameters=parameters
        )

        # Execute retrieval
        results = await wrapper.search(
            corpus=corpus,
            queries=queries,
            top_k=max(config.k_values),
        )

        # Evaluate using BEIR metrics
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=config.k_values
        )

        execution_time = time.time() - start_time

        # Flatten metrics - handle both tuple and dict returns from BEIR
        flat_metrics = {}
        if isinstance(metrics, tuple):
            # BEIR returns (ndcg, map, recall, precision) tuple
            # Extract all metrics with k values
            for metric_dict in metrics:
                if isinstance(metric_dict, dict):
                    for metric_name, k_values_dict in metric_dict.items():
                        if isinstance(k_values_dict, dict):
                            for k, value in k_values_dict.items():
                                flat_metrics[f"{metric_name}@{k}"] = value
                        else:
                            flat_metrics[metric_name] = k_values_dict
        elif isinstance(metrics, dict):
            for metric_name, k_values_dict in metrics.items():
                # Handle both dict and non-dict values
                if isinstance(k_values_dict, dict):
                    for k, value in k_values_dict.items():
                        flat_metrics[f"{metric_name}@{k}"] = value
                else:
                    # If it's a single value, use it directly
                    flat_metrics[metric_name] = k_values_dict
        else:
            logger.warning(f"Unexpected metrics type: {type(metrics)}")

        result = EvaluationResult(
            dataset=config.dataset_name,
            technique=technique_name,
            metrics=flat_metrics,
            execution_time=execution_time,
            num_queries=len(queries),
            num_documents=len(corpus),
            parameters=parameters or {},
            timestamp=datetime.now().isoformat()
        )

        logger.info(
            f"Evaluation complete: {technique_name} | "
            f"nDCG@10={flat_metrics.get('NDCG@10', 0):.4f} | "
            f"Time={execution_time:.2f}s"
        )

        return result

    async def compare_techniques(
        self,
        techniques: Dict[str, tuple],  # {name: (retriever, params)}
        datasets: List[str],
        split: str = "test",
        k_values: List[int] = None
    ) -> pd.DataFrame:
        """
        Compare multiple RAG techniques across datasets

        Args:
            techniques: Dictionary of techniques to compare
                Format: {technique_name: (retriever, parameters)}
            datasets: List of BEIR datasets to evaluate on
            split: Data split
            k_values: K values for metrics

        Returns:
            DataFrame with comparison results
        """
        logger.info(
            f"Comparing {len(techniques)} techniques "
            f"on {len(datasets)} datasets"
        )

        all_results = []

        for dataset_name in datasets:
            config = BEIRConfig(
                dataset_name=dataset_name,
                split=split,
                k_values=k_values
            )

            for technique_name, (retriever, parameters) in techniques.items():
                try:
                    result = await self.evaluate_technique(
                        technique_name=technique_name,
                        retriever=retriever,
                        config=config,
                        parameters=parameters
                    )

                    all_results.append(asdict(result))

                except Exception as e:
                    logger.error(
                        f"Evaluation failed: {technique_name} on {dataset_name}: {e}",
                        exc_info=True
                    )
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"comparison_{timestamp}.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Comparison results saved: {output_file}")

        return df

    def generate_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            Report dictionary
        """
        report = {
            "summary": {},
            "by_technique": {},
            "by_dataset": {},
            "best_techniques": {},
            "recommendations": []
        }

        # Summary statistics
        report["summary"] = {
            "num_techniques": results_df["technique"].nunique(),
            "num_datasets": results_df["dataset"].nunique(),
            "total_evaluations": len(results_df),
            "avg_execution_time": results_df["execution_time"].mean(),
        }

        # Performance by technique
        for technique in results_df["technique"].unique():
            technique_df = results_df[results_df["technique"] == technique]

            # Extract key metrics
            metric_cols = [c for c in technique_df.columns if "@" in c]
            avg_metrics = {}

            for col in metric_cols:
                if col in technique_df.columns:
                    avg_metrics[col] = technique_df[col].mean()

            report["by_technique"][technique] = {
                "avg_metrics": avg_metrics,
                "avg_time": technique_df["execution_time"].mean(),
                "num_datasets": len(technique_df)
            }

        # Performance by dataset
        for dataset in results_df["dataset"].unique():
            dataset_df = results_df[results_df["dataset"] == dataset]

            report["by_dataset"][dataset] = {
                "num_techniques": len(dataset_df),
                "best_technique": dataset_df.nlargest(1, "NDCG@10")["technique"].values[0]
                if "NDCG@10" in dataset_df.columns else None,
            }

        # Find best techniques for different metrics
        key_metrics = ["NDCG@10", "MAP@10", "Recall@10", "MRR@10"]

        for metric in key_metrics:
            if metric in results_df.columns:
                best_row = results_df.nlargest(1, metric).iloc[0]
                report["best_techniques"][metric] = {
                    "technique": best_row["technique"],
                    "dataset": best_row["dataset"],
                    "score": best_row[metric]
                }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(results_df)

        return report

    def _generate_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []

        # Find overall best technique
        if "NDCG@10" in results_df.columns:
            avg_by_technique = results_df.groupby("technique")["NDCG@10"].mean()
            best_technique = avg_by_technique.idxmax()
            best_score = avg_by_technique.max()

            recommendations.append(
                f"Overall best technique: '{best_technique}' "
                f"(avg nDCG@10: {best_score:.4f})"
            )

        # Speed vs quality tradeoff
        if "execution_time" in results_df.columns and "NDCG@10" in results_df.columns:
            results_df["efficiency"] = results_df["NDCG@10"] / results_df["execution_time"]
            most_efficient = results_df.nlargest(1, "efficiency").iloc[0]

            recommendations.append(
                f"Most efficient: '{most_efficient['technique']}' "
                f"(nDCG@10: {most_efficient['NDCG@10']:.4f}, "
                f"time: {most_efficient['execution_time']:.2f}s)"
            )

        return recommendations

    async def run_full_benchmark(
        self,
        techniques: Dict[str, tuple],
        datasets: List[str] = None,
        quick_mode: bool = False
    ) -> tuple[pd.DataFrame, Dict]:
        """
        Run comprehensive benchmark

        Args:
            techniques: Techniques to evaluate
            datasets: Datasets to use (default: subset of BEIR)
            quick_mode: If True, use smaller datasets for faster eval

        Returns:
            Tuple of (results DataFrame, report dictionary)
        """
        if datasets is None:
            if quick_mode:
                # Smaller datasets for quick evaluation
                datasets = ["nfcorpus", "scifact", "fiqa"]
            else:
                # Standard benchmark suite
                datasets = [
                    "msmarco",
                    "trec-covid",
                    "nfcorpus",
                    "nq",
                    "hotpotqa",
                    "fiqa",
                    "scifact"
                ]

        # Run comparison
        results_df = await self.compare_techniques(
            techniques=techniques,
            datasets=datasets,
            k_values=[1, 3, 5, 10, 20, 100]
        )

        # Generate report
        report = self.generate_report(results_df)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark report saved: {report_file}")

        return results_df, report
