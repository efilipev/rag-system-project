"""
Configuration settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    SERVICE_NAME: str = "document-ranking"
    LOG_LEVEL: str = "INFO"
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"

    # Ranker Configuration
    # Options: 'cross-encoder', 'bm25'
    RANKER_TYPE: str = "cross-encoder"

    # Cross-Encoder Configuration (used if RANKER_TYPE='cross-encoder')
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_MAX_LENGTH: int = 512
    CROSS_ENCODER_BATCH_SIZE: int = 32
    CROSS_ENCODER_DEVICE: str = "cpu"  # 'cpu', 'cuda', or 'auto'

    # BM25 Configuration (used if RANKER_TYPE='bm25')
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # Ranking Limits
    MAX_DOCUMENTS_PER_REQUEST: int = 100
    DEFAULT_TOP_K: int = 10
    MAX_QUERY_LENGTH: int = 2000


settings = Settings()
