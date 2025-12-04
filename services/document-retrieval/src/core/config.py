"""
Configuration settings for Document Retrieval Service
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Service Configuration
    SERVICE_NAME: str = "document-retrieval"
    LOG_LEVEL: str = "INFO"

    # RabbitMQ Configuration
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    RABBITMQ_QUEUE: str = "document-retrieval"
    RABBITMQ_EXCHANGE: str = "rag-exchange"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600

    # PostgreSQL Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str = "ragpassword"
    POSTGRES_DB: str = "ragdb"

    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "wikipedia"
    USE_QDRANT: bool = True

    # Chroma Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION: str = "documents"
    USE_CHROMA: bool = True

    # Model Configuration - Optimized based on benchmark results (Nov 2025)
    # BGE-base-en-v1.5: NDCG@10=0.991, 216ms/doc, best quality/speed balance
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    VECTOR_DIMENSION: int = 768  # BGE-base dimension

    # Retrieval Configuration - Optimized based on benchmark results
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    DEFAULT_SCORE_THRESHOLD: float = 0.5  # Lowered for better recall

    # Hybrid Search Configuration - Benchmark showed 30/70 dense/sparse optimal
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_DENSE_WEIGHT: float = 0.3  # Dense (vector) weight
    HYBRID_SPARSE_WEIGHT: float = 0.7  # Sparse (BM25) weight
    HYBRID_FUSION_METHOD: str = "weighted"  # "weighted" or "rrf"

    # Reranking Configuration
    ENABLE_RERANKING: bool = True
    RERANKING_TOP_K: int = 50  # Rerank top 50, return top 10

    # Chunking Configuration - Benchmark showed 1024 tokens optimal
    CHUNK_SIZE: int = 1024  # Optimal based on chunking benchmark
    CHUNK_OVERLAP: int = 102  # 10% overlap (1024 * 0.1)

    # Performance Configuration
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32

    # HyDE Configuration (using Ollama)
    # Optimized based on hyde-colbert-paper benchmark results (Phase 5)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    HYDE_N_HYPOTHETICALS: int = 3  # Optimal: generates 3 diverse hypotheticals
    HYDE_TEMPERATURES: list = [0.3, 0.5, 0.7]  # Diversity through temperature variation
    HYDE_MAX_TOKENS: int = 256
    HYDE_QUALITY_THRESHOLD: float = 0.7  # Phase 1 optimal: filters to ~40% high-quality hypotheticals

    # ColBERT Configuration
    # ColBERTv2 with adaptive pooling for 50-75% storage reduction
    COLBERT_CHECKPOINT: str = "colbert-ir/colbertv2.0"
    COLBERT_DOC_MAXLEN: int = 300  # General domain default
    COLBERT_QUERY_MAXLEN: int = 32
    COLBERT_USE_ADAPTIVE_POOLING: bool = True
    COLBERT_INDEX_PATH: str = "./data/colbert_index"

    # Fusion Configuration
    # Phase 5 optimized: 59.39% NDCG@10 on SciFact
    HYDE_COLBERT_FUSION_STRATEGY: str = "weighted_average"
    HYDE_COLBERT_FUSION_WEIGHT: float = 0.2  # Phase 5 optimal: 20% HyDE, 80% query
    HYDE_COLBERT_NORMALIZE_SCORES: bool = True  # Min-max normalization ENABLED


settings = Settings()
