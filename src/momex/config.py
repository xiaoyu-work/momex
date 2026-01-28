"""Momex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class StorageConfig:
    """Storage configuration.

    Attributes:
        path: Path for SQLite databases (one per collection).

    Example YAML:
        storage:
          path: ./momex_data
    """

    path: str = "./momex_data"


@dataclass
class MomexConfig:
    """Configuration for Momex memory system.

    LLM and embedding configuration is handled by TypeAgent via environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (for Azure)

    Momex only needs to know where to store the data.

    Attributes:
        storage: Storage configuration.
        db_name: Database filename for each collection.

    Example:
        # Simple config
        config = MomexConfig()

        # Custom storage path
        config = MomexConfig(
            storage=StorageConfig(path="./my_data")
        )

        # From YAML
        config = MomexConfig.from_yaml("momex_config.yaml")
    """

    # Class-level default config
    _default: "MomexConfig | None" = None

    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    db_name: str = "memory.db"

    def __post_init__(self) -> None:
        """Initialize configuration."""
        # Environment variable override
        env_storage = os.getenv("MOMEX_STORAGE_PATH")
        if env_storage and self.storage.path == "./momex_data":
            self.storage = StorageConfig(path=env_storage)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MomexConfig:
        """Create configuration from a YAML file.

        YAML format:
            storage:
              path: ./momex_data

            db_name: memory.db

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

        config = cls(**{k: v for k, v in data.items() if v is not None})

        if storage:
            config.storage = storage

        return config

    @classmethod
    def set_default(
        cls,
        storage_path: str | None = None,
        storage: StorageConfig | None = None,
    ) -> "MomexConfig":
        """Set the global default configuration.

        Args:
            storage_path: Base directory for SQLite (shorthand for storage config).
            storage: Full storage configuration.

        Returns:
            The created default MomexConfig instance.
        """
        if storage is None:
            storage = StorageConfig(path=storage_path or "./momex_data")

        cls._default = cls(storage=storage)
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
