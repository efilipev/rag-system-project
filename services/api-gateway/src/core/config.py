"""
Configuration settings for API Gateway
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    SERVICE_NAME: str = "api-gateway"
    LOG_LEVEL: str = "INFO"

    # Service URLs (internal Docker network ports)
    QUERY_ANALYSIS_URL: str = "http://query-analysis:8101"
    DOCUMENT_RETRIEVAL_URL: str = "http://document-retrieval:8000"
    DOCUMENT_RANKING_URL: str = "http://document-ranking:8000"
    LATEX_PARSER_URL: str = "http://latex-parser:8000"
    LLM_GENERATION_URL: str = "http://llm-generation:8000"
    RESPONSE_FORMATTER_URL: str = "http://response-formatter:8000"

    # Cache
    REDIS_URL: str = "redis://redis:6379"
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600

    # Pipeline Configuration
    ENABLE_QUERY_ANALYSIS: bool = True
    ENABLE_RANKING: bool = True
    ENABLE_LATEX_PARSING: bool = True
    RETRIEVAL_TOP_K: int = 20
    RANKING_TOP_K: int = 10
    DEFAULT_OUTPUT_FORMAT: str = "markdown"

    # Timeouts (seconds)
    QUERY_ANALYSIS_TIMEOUT: float = 10.0
    DOCUMENT_RETRIEVAL_TIMEOUT: float = 30.0
    DOCUMENT_RANKING_TIMEOUT: float = 20.0
    LATEX_PARSER_TIMEOUT: float = 10.0
    LLM_GENERATION_TIMEOUT: float = 120.0
    RESPONSE_FORMATTER_TIMEOUT: float = 5.0

    # Request limits
    MAX_QUERY_LENGTH: int = 2000

    # Authentication
    ENABLE_AUTHENTICATION: bool = False  # Set to True to require auth
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"  # Change in production!
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = False  # Set to True to enable
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    # Security Headers
    ENABLE_SECURITY_HEADERS: bool = True
    ENABLE_HSTS: bool = False  # Set to True when using HTTPS in production

    # CORS Settings
    CORS_ALLOWED_ORIGINS: list = ["*"]  # In production, specify exact origins
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_MAX_AGE: int = 600  # Cache preflight requests for 10 minutes


settings = Settings()
