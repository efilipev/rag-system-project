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

    SERVICE_NAME: str = "latex-parser"
    LOG_LEVEL: str = "INFO"
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"

    # Parser Configuration
    PARSER_TYPE: str = "default"  # 'default', 'sympy', 'latex'

    # Parsing Limits
    MAX_LATEX_LENGTH: int = 10000
    MAX_BATCH_SIZE: int = 50

    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour


settings = Settings()
