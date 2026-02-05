"""Configuration management with environment detection."""

import logging
import os
from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Core settings.
    telegram_bot_token: str
    tee_env: Environment = Environment.DEVELOPMENT

    # LLM settings (RedPill API).
    # Use phala/ prefix for TEE-protected inference with attestation support.
    # Models with verified attestation: phala/qwen-2.5-7b-instruct, phala/gemma-3-27b-it
    redpill_api_key: str
    llm_model: str = "phala/gemma-3-27b-it"
    llm_base_url: str = "https://api.redpill.ai/v1"

    # Game settings.
    game_timeout_seconds: int = 1800  # 30 minutes.
    message_update_interval: int = 10  # Seconds between countdown updates.

    # dstack settings.
    dstack_socket_path: str = "/var/run/dstack.sock"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def is_production(self) -> bool:
        return self.tee_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.tee_env == Environment.DEVELOPMENT

    @property
    def dstack_available(self) -> bool:
        """Check if dstack socket is available (only in production TEE)."""
        return os.path.exists(self.dstack_socket_path)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def setup_logging() -> None:
    """Configure logging based on environment."""
    settings = get_settings()
    level = logging.DEBUG if settings.is_development else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_str)

    # In production, suppress sensitive loggers.
    if settings.is_production:
        logging.getLogger("tee_totalled.llm").setLevel(logging.WARNING)
        logging.getLogger("tee_totalled.game").setLevel(logging.WARNING)
