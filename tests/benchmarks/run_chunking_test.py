#!/usr/bin/env python
"""
Simplified Chunking Sensitivity Test
Tests chunk size impact on retrieval quality using open-source embedding models
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import embedding models
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_LANGCHAIN_HF = True
except ImportError:
    HAS_LANGCHAIN_HF = False
    logger.warning("langchain-huggingface not installed")


@dataclass
class ChunkingTestResult:
    """Result from chunking test"""
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    dimension: int
    ndcg_at_10: float
    recall_at_10: float
    map_at_10: float
    num_chunks: int
    avg_chunk_tokens: float
    embedding_time_ms: float
    search_time_ms: float
    storage_mb: float


# Open-source embedding models to test (from embedding_comparison_results)
EMBEDDING_MODELS = {
    "BGE-base-en-v1.5": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "provider": "huggingface"
    },
    "BGE-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "provider": "huggingface"
    },
    "GTE-base": {
        "model_id": "thenlper/gte-base",
        "dimension": 768,
        "provider": "huggingface"
    },
    "GTE-large": {
        "model_id": "thenlper/gte-large",
        "dimension": 1024,
        "provider": "huggingface"
    },
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "provider": "sentence-transformers"
    },
    "all-mpnet-base-v2": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "provider": "sentence-transformers"
    },
    "E5-large-v2": {
        "model_id": "intfloat/e5-large-v2",
        "dimension": 1024,
        "provider": "huggingface"
    },
}

# Chunk sizes to test
CHUNK_SIZES = [64, 128, 256, 512, 1024, 2048]


def create_sample_corpus() -> Dict[str, Dict]:
    """Create extended sample corpus for testing with longer documents"""
    # Extended documents about RAG systems with more content
    documents = {
        "doc1": {
            "text": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language models
            by incorporating external knowledge retrieval. The system first retrieves relevant documents from
            a knowledge base, then uses these documents as context for generating responses. This approach
            addresses the limitation of LLMs having static knowledge cutoffs and potential hallucination issues.
            RAG systems typically consist of several components: a document store, an embedding model for
            semantic search, a retriever that finds relevant passages, and a language model for generation.
            The quality of retrieval directly impacts the quality of generated responses.

            The architecture of RAG systems can be divided into two main phases: indexing and querying.
            During the indexing phase, documents are processed, chunked into smaller segments, and converted
            into vector embeddings using a pre-trained encoder model. These embeddings are stored in a vector
            database optimized for similarity search. The querying phase involves encoding the user's question
            using the same encoder, searching for similar document chunks, and passing the retrieved context
            along with the original question to a language model for answer generation.

            Several key factors affect RAG performance. Chunk size determines how much context is retrieved
            per document segment. Smaller chunks may miss important context, while larger chunks may introduce
            noise. The embedding model choice affects semantic understanding - models trained on diverse data
            generalize better. The number of retrieved chunks (top-k) balances recall with context window limits.
            Reranking retrieved documents can significantly improve precision by using more sophisticated models.

            Advanced RAG techniques include query expansion, where the original query is augmented with related
            terms or hypothetical answers to improve retrieval. Multi-hop retrieval chains multiple queries to
            answer complex questions. Iterative refinement allows the system to ask follow-up questions based
            on initial retrieval results. These techniques combined can achieve state-of-the-art performance
            on knowledge-intensive NLP benchmarks."""
        },
        "doc2": {
            "text": """Vector databases are essential for modern RAG systems. They store document embeddings
            and enable fast similarity search using algorithms like HNSW (Hierarchical Navigable Small World).
            Popular vector databases include Qdrant, Pinecone, Weaviate, and Chroma. Each offers different
            trade-offs between performance, scalability, and features. Qdrant provides excellent filtering
            capabilities, while Pinecone offers managed cloud solutions. The choice depends on specific
            requirements like query latency, update frequency, and infrastructure preferences.

            HNSW (Hierarchical Navigable Small World) is the most common algorithm for approximate nearest
            neighbor search in vector databases. It builds a multi-layer graph structure where higher layers
            contain fewer, more widely connected nodes for fast global navigation, while lower layers provide
            fine-grained local search. Parameters like ef_construction and M affect build time and search quality.

            Qdrant is an open-source vector database written in Rust, designed for production workloads. It
            supports payload filtering, allowing queries like "find similar documents where category equals X".
            The filtering is applied during the search, not as a post-processing step, making it efficient.
            Qdrant also supports quantization to reduce memory usage while maintaining search quality.

            Alternative vector databases include Milvus, which scales to billions of vectors across clusters,
            and Chroma, which is simpler but suitable for smaller deployments. Weaviate adds schema-based
            organization and GraphQL querying. Pinecone is a managed service that handles infrastructure
            automatically but requires sending data to external servers. Each choice involves trade-offs
            between cost, complexity, performance, and data privacy requirements."""
        },
        "doc3": {
            "text": """ColBERT (Contextualized Late Interaction over BERT) is an advanced retrieval model that
            uses late interaction for computing relevance scores. Unlike traditional bi-encoders that compress
            documents into single vectors, ColBERT maintains per-token representations. During retrieval,
            it computes MaxSim scores between query and document tokens, enabling fine-grained matching.
            This approach achieves better accuracy while remaining efficient through pre-computation of
            document embeddings and using approximate nearest neighbor search for initial candidate selection.

            The late interaction mechanism works by computing the maximum similarity between each query token
            and all document tokens, then summing these maximum similarities. This allows the model to capture
            fine-grained semantic matching that single-vector approaches might miss. For example, when matching
            "machine learning applications", each word can find its best match independently in the document.

            ColBERT v2 introduces several optimizations including compression of document embeddings using
            residual representations and centroids. This reduces storage requirements by 6-10x while maintaining
            most of the accuracy. The compression works by clustering token embeddings and storing only the
            difference (residual) from cluster centroids, which requires fewer bits.

            Training ColBERT requires contrastive learning with hard negatives - documents that are similar but
            not relevant. The model learns to rank relevant documents higher than these challenging distractors.
            Fine-tuning on domain-specific data can significantly improve performance for specialized applications
            like legal or medical document retrieval where terminology differs from general text."""
        },
        "doc4": {
            "text": """HyDE (Hypothetical Document Embeddings) is a zero-shot technique for improving retrieval.
            Given a query, HyDE first generates a hypothetical document that might answer the query using an LLM.
            This hypothetical document is then embedded and used for similarity search instead of the original query.
            The key insight is that documents are more similar to other documents than to queries, bridging
            the vocabulary gap between how users phrase questions and how information is written in documents.

            The HyDE approach addresses a fundamental asymmetry in semantic search: users ask questions using
            natural conversational language, while documents contain declarative information in a different style.
            For example, a user might ask "How do I fix a memory leak in Python?" while a relevant document
            discusses "Memory management techniques including garbage collection, reference counting, and common
            patterns for avoiding memory leaks in Python applications."

            Implementation of HyDE requires a language model capable of generating plausible answers. GPT-3.5,
            GPT-4, or open-source models like Llama can serve this purpose. The generated document doesn't need
            to be factually correct - it just needs to use similar vocabulary and structure as actual relevant
            documents. Multiple hypothetical documents can be generated and their embeddings averaged for robustness.

            Limitations of HyDE include increased latency from the LLM generation step and potential cost for
            API-based models. The technique works best when the LLM has good coverage of the target domain.
            For specialized domains with unique terminology, fine-tuning may be necessary. Despite limitations,
            HyDE often provides significant improvements over direct query embedding, especially for complex queries."""
        },
        "doc5": {
            "text": """Hybrid search combines dense retrieval (semantic search using embeddings) with sparse
            retrieval (keyword matching using BM25). This approach leverages the strengths of both methods:
            dense retrieval captures semantic meaning while sparse retrieval handles exact keyword matches.
            Research shows that hybrid search with appropriate fusion weights often outperforms either method
            alone. Common fusion strategies include weighted combination and Reciprocal Rank Fusion (RRF).

            BM25 is a probabilistic ranking function that extends TF-IDF with document length normalization
            and term frequency saturation. It excels at finding documents containing exact query terms, which
            is important for technical terminology, product names, or specific identifiers that embedding models
            might not distinguish well. BM25 requires no training and works immediately on any text corpus.

            The fusion weight between dense and sparse scores significantly affects hybrid search performance.
            Typical weights range from 0.3 to 0.7 for the dense component, with the optimal value depending
            on the query type and corpus characteristics. Some systems learn fusion weights automatically
            using a small set of labeled query-document pairs.

            Reciprocal Rank Fusion (RRF) is an alternative to weighted combination that doesn't require
            tuning weights. It computes final scores as the sum of reciprocal ranks from each retrieval method:
            score = sum(1 / (k + rank)) where k is typically 60. RRF is robust to score scale differences
            between methods and often performs comparably to optimally tuned weighted fusion without hyperparameters.

            Modern hybrid search implementations often use late fusion at the reranking stage. Initial
            retrieval with both BM25 and embeddings yields candidate documents, which are then rescored
            by a cross-encoder model that jointly processes the query and each document. This achieves
            higher precision while the initial retrievers ensure good recall."""
        },
    }
    return documents


def create_sample_queries() -> Dict[str, str]:
    """Create sample queries for testing"""
    return {
        "q1": "What is RAG and how does it work?",
        "q2": "Which vector databases are best for RAG systems?",
        "q3": "How does ColBERT improve retrieval accuracy?",
        "q4": "What is HyDE and how does it improve search?",
        "q5": "Why combine dense and sparse retrieval?",
    }


def create_sample_qrels() -> Dict[str, Dict[str, int]]:
    """Create relevance judgments (qrels) for evaluation"""
    return {
        "q1": {"doc1": 2, "doc2": 1},  # doc1 highly relevant, doc2 somewhat relevant
        "q2": {"doc2": 2, "doc1": 1},
        "q3": {"doc3": 2, "doc1": 1},
        "q4": {"doc4": 2, "doc3": 1},
        "q5": {"doc5": 2, "doc3": 1},
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple token-based chunking"""
    words = text.split()
    chunks = []

    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= chunk_size // 4:  # Keep chunks at least 25% of target size
            chunks.append(" ".join(chunk))

    return chunks


def calculate_ndcg(retrieved_scores: Dict[str, float], qrels: Dict[str, int], k: int = 10) -> float:
    """Calculate NDCG@k"""
    if not qrels:
        return 0.0

    # Get retrieved docs in order
    retrieved = sorted(retrieved_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # Calculate DCG
    dcg = 0.0
    for i, (doc_id, _) in enumerate(retrieved):
        rel = qrels.get(doc_id.split("_chunk_")[0], 0)  # Handle chunked doc IDs
        dcg += (2**rel - 1) / np.log2(i + 2)

    # Calculate IDCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall(retrieved_scores: Dict[str, float], qrels: Dict[str, int], k: int = 10) -> float:
    """Calculate Recall@k"""
    if not qrels:
        return 0.0

    retrieved = sorted(retrieved_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    retrieved_docs = set(doc_id.split("_chunk_")[0] for doc_id, _ in retrieved)

    relevant_docs = set(doc_id for doc_id, rel in qrels.items() if rel > 0)

    if not relevant_docs:
        return 0.0

    return len(retrieved_docs & relevant_docs) / len(relevant_docs)


def calculate_map(retrieved_scores: Dict[str, float], qrels: Dict[str, int], k: int = 10) -> float:
    """Calculate MAP@k"""
    if not qrels:
        return 0.0

    retrieved = sorted(retrieved_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    relevant_count = 0
    precision_sum = 0.0

    for i, (doc_id, _) in enumerate(retrieved):
        parent_doc = doc_id.split("_chunk_")[0]
        if qrels.get(parent_doc, 0) > 0:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    total_relevant = sum(1 for rel in qrels.values() if rel > 0)

    return precision_sum / total_relevant if total_relevant > 0 else 0.0


class SimpleEmbeddingModel:
    """Simple wrapper for embedding models"""

    def __init__(self, model_name: str, model_config: Dict):
        self.model_name = model_name
        self.model_id = model_config["model_id"]
        self.dimension = model_config["dimension"]
        self.provider = model_config["provider"]
        self.model = None

    def load(self):
        """Load the model"""
        logger.info(f"Loading model: {self.model_name} ({self.model_id})")

        # Always try sentence-transformers first as it can load most HuggingFace models
        if HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(self.model_id)
        elif HAS_LANGCHAIN_HF:
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_id,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise RuntimeError(f"Cannot load model {self.model_name}: no available provider")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts"""
        if self.model is None:
            self.load()

        if isinstance(self.model, SentenceTransformer):
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            return np.array(self.model.embed_documents(texts))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        if self.model is None:
            self.load()

        if isinstance(self.model, SentenceTransformer):
            return self.model.encode([query], convert_to_numpy=True)[0]
        else:
            return np.array(self.model.embed_query(query))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_chunking_test(
    model_names: List[str] = None,
    chunk_sizes: List[int] = None,
    overlap_pct: int = 10
) -> pd.DataFrame:
    """
    Run chunking sensitivity test

    Args:
        model_names: List of model names to test (default: all)
        chunk_sizes: List of chunk sizes to test (default: CHUNK_SIZES)
        overlap_pct: Overlap percentage (default: 10)

    Returns:
        DataFrame with results
    """
    if model_names is None:
        model_names = list(EMBEDDING_MODELS.keys())
    if chunk_sizes is None:
        chunk_sizes = CHUNK_SIZES

    # Create test data
    corpus = create_sample_corpus()
    queries = create_sample_queries()
    qrels = create_sample_qrels()

    results = []

    for model_name in model_names:
        if model_name not in EMBEDDING_MODELS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        model_config = EMBEDDING_MODELS[model_name]

        try:
            model = SimpleEmbeddingModel(model_name, model_config)
            model.load()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue

        for chunk_size in chunk_sizes:
            overlap = int(chunk_size * overlap_pct / 100)

            logger.info(f"Testing {model_name} with chunk_size={chunk_size}, overlap={overlap}")

            # Chunk documents
            chunked_docs = {}
            for doc_id, doc_data in corpus.items():
                chunks = chunk_text(doc_data["text"], chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    chunked_docs[f"{doc_id}_chunk_{i}"] = chunk

            # Embed documents
            start_time = time.time()
            doc_texts = list(chunked_docs.values())
            doc_embeddings = model.embed(doc_texts)
            embedding_time_ms = (time.time() - start_time) * 1000

            # Create doc_id to embedding mapping
            doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(chunked_docs.keys())}

            # Evaluate queries
            all_ndcg = []
            all_recall = []
            all_map = []
            total_search_time = 0

            for query_id, query_text in queries.items():
                # Embed query
                start_time = time.time()
                query_embedding = model.embed_query(query_text)

                # Calculate similarities
                scores = {}
                for doc_id, idx in doc_id_to_idx.items():
                    sim = cosine_similarity(query_embedding, doc_embeddings[idx])
                    scores[doc_id] = sim

                total_search_time += (time.time() - start_time) * 1000

                # Calculate metrics
                query_qrels = qrels.get(query_id, {})
                all_ndcg.append(calculate_ndcg(scores, query_qrels, k=10))
                all_recall.append(calculate_recall(scores, query_qrels, k=10))
                all_map.append(calculate_map(scores, query_qrels, k=10))

            # Calculate storage
            storage_mb = (len(chunked_docs) * model_config["dimension"] * 4) / (1024 * 1024)

            results.append(ChunkingTestResult(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                embedding_model=model_name,
                dimension=model_config["dimension"],
                ndcg_at_10=np.mean(all_ndcg),
                recall_at_10=np.mean(all_recall),
                map_at_10=np.mean(all_map),
                num_chunks=len(chunked_docs),
                avg_chunk_tokens=chunk_size,
                embedding_time_ms=embedding_time_ms,
                search_time_ms=total_search_time / len(queries),
                storage_mb=storage_mb
            ))

            logger.info(f"  NDCG@10: {np.mean(all_ndcg):.4f}, Recall@10: {np.mean(all_recall):.4f}")

    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])

    return df


def main():
    """Main function"""
    print("=" * 80)
    print("CHUNKING SENSITIVITY BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    # Run tests with all models from embedding comparison
    test_models = [
        "BGE-base-en-v1.5",
        "BGE-large-en-v1.5",
        "GTE-base",
        "GTE-large",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "E5-large-v2",
    ]

    test_chunk_sizes = [64, 128, 256, 512]

    print(f"\nTesting models: {test_models}")
    print(f"Chunk sizes: {test_chunk_sizes}")
    print("-" * 80)

    results_df = run_chunking_test(
        model_names=test_models,
        chunk_sizes=test_chunk_sizes,
        overlap_pct=10
    )

    # Save results
    results_dir = Path(__file__).parent.parent.parent / "test_results" / "chunking_new"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"chunking_sensitivity_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nResults saved to: {csv_path}")

    # Print best chunk size for each model
    print("\n" + "-" * 40)
    print("BEST CHUNK SIZE PER MODEL:")
    print("-" * 40)

    for model in results_df["embedding_model"].unique():
        model_df = results_df[results_df["embedding_model"] == model]
        best_row = model_df.loc[model_df["ndcg_at_10"].idxmax()]
        print(f"{model}: chunk_size={int(best_row['chunk_size'])}, NDCG@10={best_row['ndcg_at_10']:.4f}")

    return results_df


if __name__ == "__main__":
    results = main()
