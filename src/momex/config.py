"""Momex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError


class StorageBackend(StrEnum):
    """Storage backend types."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass
class StorageConfig:
    """SQLite storage configuration.

    Attributes:
        path: Path for SQLite databases (one per collection).

    Example YAML:
        storage:
          path: ./momex_data
    """

    path: str = "./momex_data"


@dataclass
class PostgresConfig:
    """PostgreSQL storage configuration.

    Attributes:
        url: PostgreSQL connection URL.
        pool_min: Minimum connections in pool.
        pool_max: Maximum connections in pool.
        schema: Database schema to use (for multi-tenant isolation).

    Example YAML:
        postgres:
          url: postgresql://user:pass@localhost:5432/momex
          pool_min: 2
          pool_max: 10
          schema: public
    """

    url: str = ""
    pool_min: int = 2
    pool_max: int = 10
    schema: str = "public"

    def __post_init__(self) -> None:
        # Environment variable override
        env_url = os.getenv("MOMEX_POSTGRES_URL")
        if env_url and not self.url:
            self.url = env_url


@dataclass
class MomexConfig:
    """Configuration for Momex memory system.

    LLM and embedding configuration is handled by TypeAgent via environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (for Azure)

    Momex needs to know where to store the data and which backend to use.

    Attributes:
        backend: Storage backend type ("sqlite" or "postgres").
        storage: SQLite storage configuration (used when backend="sqlite").
        postgres: PostgreSQL configuration (used when backend="postgres").
        db_name: Database filename for each collection (SQLite only).

    Example (Code):
        # SQLite (default)
        config = MomexConfig()

        # PostgreSQL
        config = MomexConfig(
            backend="postgres",
            postgres=PostgresConfig(url="postgresql://user:pass@localhost:5432/momex")
        )

    Example (YAML):
        # SQLite
        backend: sqlite
        storage:
          path: ./momex_data

        # PostgreSQL
        backend: postgres
        postgres:
          url: postgresql://user:pass@localhost:5432/momex
          pool_min: 2
          pool_max: 10
    """

    # Class-level default config
    _default: "MomexConfig | None" = None

    # Backend selection
    backend: Literal["sqlite", "postgres"] = "sqlite"

    # SQLite configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    db_name: str = "memory.db"

    # PostgreSQL configuration
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    def __post_init__(self) -> None:
        """Initialize configuration."""
        # Environment variable overrides
        env_backend = os.getenv("MOMEX_BACKEND")
        if env_backend and env_backend in ("sqlite", "postgres"):
            self.backend = env_backend  # type: ignore

        env_storage = os.getenv("MOMEX_STORAGE_PATH")
        if env_storage and self.storage.path == "./momex_data":
            self.storage = StorageConfig(path=env_storage)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if self.backend == "postgres" and not self.postgres.url:
            env_url = os.getenv("MOMEX_POSTGRES_URL")
            if env_url:
                self.postgres.url = env_url
            else:
                raise ConfigurationError(
                    message="PostgreSQL URL is required when using postgres backend",
                    suggestion="Set postgres.url in config or MOMEX_POSTGRES_URL environment variable",
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MomexConfig:
        """Create configuration from a YAML file.

        YAML format (SQLite):
            backend: sqlite
            storage:
              path: ./momex_data
            db_name: memory.db

        YAML format (PostgreSQL):
            backend: postgres
            postgres:
              url: postgresql://user:pass@localhost:5432/momex
              pool_min: 2
              pool_max: 10

        Args:
            path: Path to the YAML configuration file.

        Returns:
            MomexConfig instance.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise ConfigurationError(
                message=f"Config file not found: {path}",
                config_path=str(path),
                suggestion="Create the config file or check the path.",
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parse storage config (SQLite)
        storage = None
        if "storage" in data:
            storage_data = data.pop("storage")
            storage = StorageConfig(**storage_data)

        # Parse postgres config
        postgres = None
        if "postgres" in data:
            postgres_data = data.pop("postgres")
            postgres = PostgresConfig(**postgres_data)

        config = cls(**{k: v for k, v in data.items() if v is not None})

        if storage:
            config.storage = storage
        if postgres:
            config.postgres = postgres

        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to the output YAML file.
        """
        import yaml

        data: dict = {"backend": self.backend}

        if self.backend == "sqlite":
            data["storage"] = {"path": self.storage.path}
            data["db_name"] = self.db_name
        else:
            data["postgres"] = {
                "url": self.postgres.url,
                "pool_min": self.postgres.pool_min,
                "pool_max": self.postgres.pool_max,
                "schema": self.postgres.schema,
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def set_default(
        cls,
        storage_path: str | None = None,
        storage: StorageConfig | None = None,
        backend: Literal["sqlite", "postgres"] = "sqlite",
        postgres: PostgresConfig | None = None,
    ) -> "MomexConfig":
        """Set the global default configuration.

        Args:
            storage_path: Base directory for SQLite (shorthand for storage config).
            storage: Full SQLite storage configuration.
            backend: Storage backend type.
            postgres: PostgreSQL configuration.

        Returns:
            The created default MomexConfig instance.
        """
        if storage is None:
            storage = StorageConfig(path=storage_path or "./momex_data")
        if postgres is None:
            postgres = PostgresConfig()

        cls._default = cls(
            backend=backend,
            storage=storage,
            postgres=postgres,
        )
        return cls._default

    @classmethod
    def get_default(cls) -> "MomexConfig":
        """Get the global default configuration.

        If no default has been set, creates one with default values.

        Returns:
            The default MomexConfig instance.
        """
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def clear_default(cls) -> None:
        """Clear the global default configuration."""
        cls._default = None

    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL backend."""
        return self.backend == "postgres"

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite backend."""
        return self.backend == "sqlite"
