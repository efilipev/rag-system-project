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

    SERVICE_NAME: str = "llm-generation"
    LOG_LEVEL: str = "INFO"
    RABBITMQ_URL: str = "amqp://raguser:ragpassword@localhost:5672/"
    REDIS_URL: str = "redis://localhost:6379"

    # LLM Provider Configuration
    # Options: 'openai', 'local', 'ollama', 'vllm', 'llamacpp'
    LLM_PROVIDER: str = "local"

    # OpenAI Configuration (used if LLM_PROVIDER='openai')
    OPENAI_API_KEY: str = ""

    # Local LLM Configuration (used if LLM_PROVIDER='local', 'ollama', 'vllm')
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434"
    LOCAL_LLM_API_TYPE: str = "ollama"  # 'ollama', 'vllm', 'llamacpp', 'openai-compatible'
    LOCAL_LLM_TIMEOUT: int = 300

    # Model Configuration
    DEFAULT_MODEL: str = "llama3"  # or 'gpt-3.5-turbo' for OpenAI
    MAX_TOKENS_DEFAULT: int = 1000
    TEMPERATURE_DEFAULT: float = 0.7

    # Generation Limits
    MAX_CONTEXT_DOCUMENTS: int = 10
    MAX_QUERY_LENGTH: int = 2000


settings = Settings()
