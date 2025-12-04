"""
Configuration settings for Query Analysis Service
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
    SERVICE_NAME: str = "query-analysis"
    LOG_LEVEL: str = "INFO"
    SERVICE_PORT: int = 8101

    # RabbitMQ Configuration
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "raguser"
    RABBITMQ_PASSWORD: str = "ragpassword"
    RABBITMQ_VHOST: str = "/"
    RABBITMQ_QUEUE: str = "query-analysis"
    RABBITMQ_EXCHANGE: str = "rag-exchange"
    RABBITMQ_HISTORY_QUEUE: str = "query-history"
    RABBITMQ_ANALYTICS_QUEUE: str = "query-analytics"

    @property
    def RABBITMQ_URL(self) -> str:
        """Construct RabbitMQ URL from components"""
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}{self.RABBITMQ_VHOST}"

    # Redis Configuration
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600
    CACHE_ENABLED: bool = True

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Model Configuration - Using smaller model to reduce memory usage
    # sentence-transformers/all-MiniLM-L6-v2: Lighter model (~80MB vs ~400MB)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # MiniLM dimension
    SPACY_MODEL: str = "en_core_web_sm"  # For NER and keyword extraction (sm for lower memory)
    INTENT_MODEL_TYPE: str = "keyword"  # Options: 'zero-shot', 'sklearn', 'custom', 'keyword'
    INTENT_MODEL_PATH: Optional[str] = None  # Path to trained intent classifier

    # OpenAI Configuration (optional for advanced query analysis)
    OPENAI_API_KEY: Optional[str] = None

    # Performance Configuration
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    MAX_QUERY_LENGTH: int = 512

    # Query Analysis Configuration
    MIN_QUERY_LENGTH: int = 3
    MAX_KEYWORDS: int = 10
    ENABLE_ENTITY_EXTRACTION: bool = True
    ENABLE_INTENT_CLASSIFICATION: bool = True
    ENABLE_QUERY_EXPANSION: bool = True
    QUERY_EXPANSION_COUNT: int = 3  # Number of query variations to generate

    # LaTeX Parser Service
    LATEX_PARSER_URL: str = "http://latex-parser:8104"

    # Feature Flags
    ENABLE_CACHING: bool = True
    ENABLE_ADVANCED_NER: bool = True

    # Monitoring Configuration
    METRICS_ENABLED: bool = True
    TRACING_ENABLED: bool = False
    TRACE_SAMPLE_RATE: float = 0.1


settings = Settings()
