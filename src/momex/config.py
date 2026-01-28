"""Momex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError


@dataclass
class FactType:
    """A type of fact to extract from conversations."""

    name: str
    description: str


# Default fact types (from mem0)
DEFAULT_FACT_TYPES: list[FactType] = [
    FactType(
        name="Personal Preferences",
        description="Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.",
    ),
    FactType(
        name="Important Personal Details",
        description="Remember significant personal information like names, relationships, and important dates.",
    ),
    FactType(
        name="Plans and Intentions",
        description="Note upcoming events, trips, goals, and any plans the user has shared.",
    ),
    FactType(
        name="Activity and Service Preferences",
        description="Recall preferences for dining, travel, hobbies, and other services.",
    ),
    FactType(
        name="Health and Wellness Preferences",
        description="Keep a record of dietary restrictions, fitness routines, and other wellness-related information.",
    ),
    FactType(
        name="Professional Details",
        description="Remember job titles, work habits, career goals, and other professional information.",
    ),
    FactType(
        name="Miscellaneous Information",
        description="Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.",
    ),
]


@dataclass
class StorageConfig:
    """Storage backend configuration.

    Attributes:
        backend: Storage backend type ("sqlite" or "postgres").
        path: Path for SQLite database (used when backend="sqlite").
        connection_string: PostgreSQL connection string (used when backend="postgres").
        table_prefix: Table name prefix for PostgreSQL (default "momex").

    Example YAML:
        # SQLite (default)
        storage:
          backend: sqlite
          path: ./momex_data

        # PostgreSQL
        storage:
          backend: postgres
          connection_string: postgresql://user:pass@localhost/momex

        # Supabase
        storage:
          backend: postgres
          connection_string: postgresql://user:pass@db.xxx.supabase.co:5432/postgres
    """

    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str = "./momex_data"
    connection_string: str | None = None
    table_prefix: str = "momex"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend == "postgres" and not self.connection_string:
            raise ConfigurationError(
                message="connection_string is required for PostgreSQL backend",
                suggestion="Set storage.connection_string in your config.",
            )


@dataclass
class MomexConfig:
    """Configuration for Momex memory system.

    LLM configuration is handled by TypeAgent via environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (for Azure)

    Attributes:
        storage: Storage backend configuration.
        storage_path: (Deprecated) Use storage.path instead.
        db_name: Database filename (for SQLite).
        fact_types: List of fact types to extract from conversations.
        similarity_threshold: Minimum similarity score for memory matching (0.0-1.0).
        importance_weight: Weight for importance in search scoring (0.0-1.0).
        embedding_model: OpenAI embedding model name.
        embedding_dim: Embedding vector dimension (auto-detected from model).

    Example:
        # Simple SQLite config
        config = MomexConfig()

        # PostgreSQL config
        config = MomexConfig(
            storage=StorageConfig(
                backend="postgres",
                connection_string="postgresql://user:pass@localhost/momex"
            )
        )

        # From YAML
        config = MomexConfig.from_yaml("momex_config.yaml")
    """

    # Class-level default config
    _default: "MomexConfig | None" = None

    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Legacy field (deprecated, use storage.path)
    storage_path: str | None = None
    db_name: str = "memory.db"

    # LLM configuration
    fact_types: list[FactType] = field(default_factory=lambda: DEFAULT_FACT_TYPES.copy())

    # Search configuration
    similarity_threshold: float = 0.3  # Adjusted for text-embedding-3-small
    importance_weight: float = 0.3  # 30% importance, 70% similarity

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int | None = None  # Auto-detected from model

    def __post_init__(self) -> None:
        """Initialize configuration."""
        # Handle legacy storage_path
        if self.storage_path:
            self.storage = StorageConfig(backend="sqlite", path=self.storage_path)

        # Environment variable override
        env_storage = os.getenv("MOMEX_STORAGE_PATH")
        if env_storage and self.storage.path == "./momex_data":
            self.storage = StorageConfig(backend="sqlite", path=env_storage)

        # Auto-detect embedding dimension from model
        if self.embedding_dim is None:
            self.embedding_dim = self._get_embedding_dim(self.embedding_model)

    @staticmethod
    def _get_embedding_dim(model: str) -> int:
        """Get embedding dimension for a model."""
        dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return dims.get(model, 1536)

    def get_storage_path(self) -> str:
        """Get the storage path (for backward compatibility)."""
        return self.storage.path

    @classmethod
    def from_yaml(cls, path: str | Path) -> MomexConfig:
        """Create configuration from a YAML file.

        YAML format:
            # Storage backend configuration
            storage:
              backend: sqlite  # or postgres
              path: ./momex_data  # for sqlite
              connection_string: postgresql://...  # for postgres
              table_prefix: momex  # for postgres

            # Embedding model configuration
            embedding_model: text-embedding-3-small

            # Search configuration
            similarity_threshold: 0.3
            importance_weight: 0.3

            # Custom fact types
            fact_types:
              - name: Personal Preferences
                description: Keep track of likes and dislikes

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

        # Parse storage config
        storage = None
        if "storage" in data:
            storage_data = data.pop("storage")
            storage = StorageConfig(**storage_data)

        # Parse fact_types
        fact_types = None
        if "fact_types" in data:
            fact_types = [
                FactType(name=ft["name"], description=ft["description"])
                for ft in data["fact_types"]
            ]
            del data["fact_types"]

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        config = cls(**data)

        if storage:
            config.storage = storage
        if fact_types:
            config.fact_types = fact_types

        return config

    @classmethod
    def set_default(
        cls,
        storage_path: str | None = None,
        storage: StorageConfig | None = None,
        fact_types: list[FactType] | None = None,
        similarity_threshold: float = 0.3,
        embedding_model: str = "text-embedding-3-small",
    ) -> "MomexConfig":
        """Set the global default configuration.

        Args:
            storage_path: Base directory for SQLite (shorthand for storage config).
            storage: Full storage configuration.
            fact_types: List of fact types to extract.
            similarity_threshold: Minimum similarity score for memory matching.
            embedding_model: OpenAI embedding model name.

        Returns:
            The created default MomexConfig instance.
        """
        if storage is None:
            storage = StorageConfig(
                backend="sqlite",
                path=storage_path or "./momex_data",
            )

        cls._default = cls(
            storage=storage,
            fact_types=fact_types or DEFAULT_FACT_TYPES.copy(),
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
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
