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

    SERVICE_NAME: str = "response-formatter"
    LOG_LEVEL: str = "INFO"
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"

    # Formatter Configuration
    FORMATTER_TYPE: str = "jinja2"  # 'jinja2', 'default'

    # Formatting Limits
    MAX_CONTENT_LENGTH: int = 100000
    MAX_SOURCES: int = 50
    MAX_BATCH_SIZE: int = 20

    # Default Settings
    DEFAULT_OUTPUT_FORMAT: str = "markdown"
    INCLUDE_CITATIONS_DEFAULT: bool = True
    INCLUDE_METADATA_DEFAULT: bool = False


settings = Settings()
