"""Momex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError


@dataclass
class PostgresConfig:
    """PostgreSQL configuration.

    Attributes:
        url: PostgreSQL connection URL.
        pool_min: Minimum connections in pool.
        pool_max: Maximum connections in pool.
    """

    url: str = ""
    pool_min: int = 2
    pool_max: int = 10

    def __post_init__(self) -> None:
        env_url = os.getenv("MOMEX_POSTGRES_URL")
        if env_url and not self.url:
            self.url = env_url


@dataclass
class MomexConfig:
    """Momex configuration.

    Example:
        config = MomexConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-xxx",
        )
    """

    # Class-level default
    _default: "MomexConfig | None" = None

    # LLM (required)
    provider: str = ""  # openai, azure, anthropic, deepseek, qwen
    model: str = ""
    api_key: str = ""
    api_base: str = ""  # Required for Azure
    temperature: float = 0.0

    # Storage
    backend: Literal["sqlite", "postgres"] = "sqlite"
    storage_path: str = "./momex_data"  # SQLite only
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    def __post_init__(self) -> None:
        # Env var overrides
        if env := os.getenv("MOMEX_BACKEND"):
            if env in ("sqlite", "postgres"):
                self.backend = env  # type: ignore
        if env := os.getenv("MOMEX_STORAGE_PATH"):
            self.storage_path = env

    def validate(self) -> None:
        """Validate configuration."""
        # LLM env var fallbacks
        if not self.provider:
            self.provider = os.getenv("MOMEX_PROVIDER", "openai")
        if not self.model:
            self.model = os.getenv("MOMEX_MODEL", "")
        if not self.api_key:
            self.api_key = os.getenv("MOMEX_API_KEY", "")
        if not self.api_base:
            self.api_base = os.getenv("MOMEX_API_BASE", "")

        if not self.model:
            raise ConfigurationError(
                message="model is required",
                suggestion="Set model in MomexConfig or MOMEX_MODEL env var",
            )
        if not self.api_key:
            raise ConfigurationError(
                message="api_key is required",
                suggestion="Set api_key in MomexConfig or MOMEX_API_KEY env var",
            )
        if self.provider == "azure" and not self.api_base:
            raise ConfigurationError(
                message="api_base is required for Azure",
                suggestion="Set api_base in MomexConfig or MOMEX_API_BASE env var",
            )
        if self.backend == "postgres" and not self.postgres.url:
            raise ConfigurationError(
                message="postgres.url is required",
                suggestion="Set postgres url or MOMEX_POSTGRES_URL env var",
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MomexConfig:
        """Load from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise ConfigurationError(message=f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parse nested postgres config
        postgres = None
        if "postgres" in data:
            postgres = PostgresConfig(**data.pop("postgres"))

        config = cls(**{k: v for k, v in data.items() if v is not None})
        if postgres:
            config.postgres = postgres

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save to YAML file."""
        import yaml

        data: dict = {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "backend": self.backend,
        }

        if self.api_base:
            data["api_base"] = self.api_base

        if self.backend == "sqlite":
            data["storage_path"] = self.storage_path
        else:
            data["postgres"] = {
                "url": self.postgres.url,
                "pool_min": self.postgres.pool_min,
                "pool_max": self.postgres.pool_max,
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def set_default(cls, **kwargs) -> "MomexConfig":
        """Set global default configuration."""
        cls._default = cls(**kwargs)
        return cls._default

    @classmethod
    def get_default(cls) -> "MomexConfig":
        """Get global default configuration."""
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def clear_default(cls) -> None:
        """Clear global default configuration."""
        cls._default = None

    @property
    def is_postgres(self) -> bool:
        return self.backend == "postgres"

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"

    def get_llm_config(self):
        """Get TypeAgent LLMConfig."""
        from typeagent.llm import LLMConfig
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature,
        )

    def create_llm(self):
        """Create TypeAgent LLM instance."""
        from typeagent.llm import create_llm
        return create_llm(self.get_llm_config())
