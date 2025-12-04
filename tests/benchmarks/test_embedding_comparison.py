"""
Test runner for Embedding Model Comparison Benchmark

Usage:
    pytest tests/benchmarks/test_embedding_comparison.py -v -s

Or run directly:
    python tests/benchmarks/test_embedding_comparison.py
"""
import pytest
import asyncio
import logging
from pathlib import Path
from typing import Dict, List

from tests.benchmarks.embedding_model_comparison_benchmark import (
    EmbeddingModelComparison,
    EMBEDDING_MODELS,
    EmbeddingModelConfig,
    EmbeddingProvider
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample Test Data
def create_sample_corpus() -> Dict[str, Dict]:
    """Create sample document corpus for testing"""
    return {
        # Technical/Code documents
        "doc1": {
            "text": "Python is a high-level programming language with dynamic typing and garbage collection. "
                   "It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "title": "Introduction to Python Programming"
        },
        "doc2": {
            "text": "REST APIs use HTTP methods like GET, POST, PUT, and DELETE to perform CRUD operations. "
                   "JSON is the most common data format for REST API communication.",
            "title": "RESTful API Design Principles"
        },
        "doc3": {
            "text": "Docker containers provide lightweight virtualization by packaging applications with their dependencies. "
                   "Kubernetes orchestrates container deployment, scaling, and management.",
            "title": "Container Orchestration with Kubernetes"
        },
        "doc4": {
            "text": "Neural networks consist of layers of interconnected nodes that process information. "
                   "Deep learning uses multiple hidden layers to learn hierarchical representations.",
            "title": "Deep Learning and Neural Networks"
        },

        # Academic/Research documents
        "doc5": {
            "text": "Quantum mechanics describes the behavior of matter and energy at atomic scales. "
                   "The uncertainty principle states that position and momentum cannot be simultaneously known with precision.",
            "title": "Fundamentals of Quantum Mechanics"
        },
        "doc6": {
            "text": "Climate change is driven by greenhouse gas emissions from human activities. "
                   "Rising global temperatures affect weather patterns, sea levels, and ecosystems.",
            "title": "Climate Change and Environmental Impact"
        },
        "doc7": {
            "text": "CRISPR-Cas9 is a gene editing technology that allows precise modifications to DNA. "
                   "It has applications in treating genetic diseases and agricultural improvement.",
            "title": "Gene Editing with CRISPR Technology"
        },
        "doc8": {
            "text": "The Big Bang theory explains the origin and evolution of the universe. "
                   "Cosmic microwave background radiation provides evidence for this cosmological model.",
            "title": "Cosmology and the Big Bang Theory"
        },

        # General Knowledge documents
        "doc9": {
            "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
                   "Chlorophyll in plant cells absorbs light energy to produce glucose and oxygen.",
            "title": "How Plants Make Food"
        },
        "doc10": {
            "text": "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century. "
                   "It marked advances in art, science, literature, and philosophy.",
            "title": "The Renaissance Period"
        },
        "doc11": {
            "text": "The water cycle describes how water moves through Earth's atmosphere, land, and oceans. "
                   "Evaporation, condensation, and precipitation are key stages of this cycle.",
            "title": "Understanding the Water Cycle"
        },
        "doc12": {
            "text": "Democracy is a system of government where power is vested in the people. "
                   "Citizens exercise their power through voting and elected representatives.",
            "title": "Principles of Democratic Governance"
        },

        # Additional technical documents
        "doc13": {
            "text": "SQL databases use structured schemas with tables, rows, and columns. "
                   "NoSQL databases offer flexible schemas and horizontal scalability for large datasets.",
            "title": "SQL vs NoSQL Databases"
        },
        "doc14": {
            "text": "Agile software development emphasizes iterative development and continuous feedback. "
                   "Scrum and Kanban are popular agile frameworks for project management.",
            "title": "Agile Development Methodologies"
        },
        "doc15": {
            "text": "Blockchain is a distributed ledger technology that records transactions across multiple nodes. "
                   "Cryptographic hashing ensures data integrity and immutability.",
            "title": "Blockchain Technology Overview"
        },
    }


def create_sample_queries() -> Dict[str, str]:
    """Create sample queries for testing"""
    return {
        # Technical queries
        "q1": "How does Python handle memory management?",
        "q2": "What are the main HTTP methods used in REST APIs?",
        "q3": "Explain container orchestration",
        "q4": "What is deep learning?",
        "q5": "Compare SQL and NoSQL databases",

        # Academic queries
        "q6": "What is the uncertainty principle in quantum mechanics?",
        "q7": "How does climate change affect the environment?",
        "q8": "What is CRISPR gene editing?",
        "q9": "Explain the Big Bang theory",

        # General queries
        "q10": "How do plants make food?",
        "q11": "What was the Renaissance?",
        "q12": "Describe the water cycle",
        "q13": "What is democracy?",

        # Additional technical queries
        "q14": "What is agile software development?",
        "q15": "How does blockchain work?",
    }


def create_sample_qrels() -> Dict[str, Dict[str, int]]:
    """Create sample relevance judgments (qrels)"""
    return {
        # Technical query relevance
        "q1": {"doc1": 3, "doc4": 1},  # High relevance to Python doc
        "q2": {"doc2": 3, "doc13": 1},  # High relevance to REST API doc
        "q3": {"doc3": 3, "doc15": 1},  # High relevance to Kubernetes doc
        "q4": {"doc4": 3, "doc1": 1},  # High relevance to deep learning doc
        "q5": {"doc13": 3, "doc2": 1},  # High relevance to database doc

        # Academic query relevance
        "q6": {"doc5": 3},  # Quantum mechanics
        "q7": {"doc6": 3},  # Climate change
        "q8": {"doc7": 3},  # CRISPR
        "q9": {"doc8": 3},  # Big Bang

        # General query relevance
        "q10": {"doc9": 3},  # Photosynthesis
        "q11": {"doc10": 3},  # Renaissance
        "q12": {"doc11": 3},  # Water cycle
        "q13": {"doc12": 3},  # Democracy

        # Additional technical relevance
        "q14": {"doc14": 3, "doc2": 1},  # Agile
        "q15": {"doc15": 3, "doc13": 1},  # Blockchain
    }


def create_domain_labels() -> Dict[str, str]:
    """Create domain labels for queries"""
    return {
        # Technical domain
        "q1": "technical",
        "q2": "technical",
        "q3": "technical",
        "q4": "technical",
        "q5": "technical",
        "q14": "technical",
        "q15": "technical",

        # Academic domain
        "q6": "academic",
        "q7": "academic",
        "q8": "academic",
        "q9": "academic",

        # General domain
        "q10": "general",
        "q11": "general",
        "q12": "general",
        "q13": "general",
    }


# Benchmark Tests

@pytest.mark.asyncio
async def test_embedding_comparison_small_scale():
    """Test embedding comparison with a small sample"""
    logger.info("Running small-scale embedding comparison test")

    # Create test data
    corpus = create_sample_corpus()
    queries = create_sample_queries()
    qrels = create_sample_qrels()
    domain_labels = create_domain_labels()

    # Select subset of models for quick test
    test_models = [
        model for model in EMBEDDING_MODELS
        if model.name in [
            "all-MiniLM-L6-v2",  # Current model
            "all-mpnet-base-v2",  # Better open source
            "GTE-large",  # High-quality open source
        ]
    ]

    # Run benchmark
    benchmark = EmbeddingModelComparison(
        models=test_models,
        results_dir="./test_results/embedding_comparison",
        test_sample_size=15,  # Use all 15 docs
        query_sample_size=15  # Use all 15 queries
    )

    results_df = await benchmark.run_comprehensive_benchmark(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        domain_labels=domain_labels
    )

    # Assertions
    assert len(results_df) == len(test_models), "Should have results for all models"
    assert "ndcg_at_10" in results_df.columns, "Should have NDCG metric"
    assert "overall_score" in results_df.columns, "Should have overall score"

    # Generate visualizations
    logger.info("Generating visualizations...")
    benchmark.generate_all_visualizations(results_df)

    logger.info("✓ Small-scale test completed successfully")

    return results_df


@pytest.mark.asyncio
@pytest.mark.slow
async def test_embedding_comparison_full():
    """Test embedding comparison with all models (slow)"""
    logger.info("Running full embedding comparison test")

    # Create test data
    corpus = create_sample_corpus()
    queries = create_sample_queries()
    qrels = create_sample_qrels()
    domain_labels = create_domain_labels()

    # Run benchmark with all models
    benchmark = EmbeddingModelComparison(
        models=EMBEDDING_MODELS,
        results_dir="./test_results/embedding_comparison_full",
        test_sample_size=15,
        query_sample_size=15
    )

    results_df = await benchmark.run_comprehensive_benchmark(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        domain_labels=domain_labels
    )

    # Assertions
    assert len(results_df) > 5, "Should have tested multiple models"

    # Generate visualizations
    benchmark.generate_all_visualizations(results_df)

    logger.info("✓ Full test completed successfully")

    return results_df


@pytest.mark.asyncio
async def test_open_source_models_only():
    """Test only open-source models (no API costs)"""
    logger.info("Testing open-source models only")

    # Filter open-source models
    open_source_models = [
        model for model in EMBEDDING_MODELS
        if model.is_open_source
    ]

    corpus = create_sample_corpus()
    queries = create_sample_queries()
    qrels = create_sample_qrels()
    domain_labels = create_domain_labels()

    benchmark = EmbeddingModelComparison(
        models=open_source_models,
        results_dir="./test_results/open_source_comparison",
        test_sample_size=15,
        query_sample_size=15
    )

    results_df = await benchmark.run_comprehensive_benchmark(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        domain_labels=domain_labels
    )

    # All should have zero cost
    assert (results_df["cost_per_1k_queries"] == 0).all(), "Open source models should have zero cost"

    logger.info("✓ Open-source test completed")

    return results_df


# Main execution
async def main():
    """Run benchmark directly"""
    print("="*80)
    print("EMBEDDING MODEL COMPARISON BENCHMARK")
    print("="*80)
    print()

    # Run small-scale test
    print("Running small-scale comparison...")
    results = await test_embedding_comparison_small_scale()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)
    print()

    # Print summary
    print("Top 3 Models by Overall Score:")
    top_3 = results.nlargest(3, "overall_score")

    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n{i}. {row['model_name']}")
        print(f"   Overall Score: {row['overall_score']:.2f}/100")
        print(f"   NDCG@10: {row['ndcg_at_10']:.4f}")
        print(f"   Speed: {row['avg_embedding_time_ms']:.2f}ms/doc")
        print(f"   Cost: ${row['cost_per_1k_queries']:.4f}/1k queries")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
