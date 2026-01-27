"""Memex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
class MemexConfig:
    """Configuration for Memex memory system.

    LLM configuration is handled by TypeAgent via environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (for Azure)

    Attributes:
        storage_path: Base directory for storing memory databases.
        db_name: Database filename.
        fact_types: List of fact types to extract from conversations.
        similarity_threshold: Minimum similarity score for memory matching (0.0-1.0).
    """

    # Class-level default config
    _default: "MemexConfig | None" = None

    storage_path: str = "./memex_data"
    db_name: str = "memory.db"
    fact_types: list[FactType] = field(default_factory=lambda: DEFAULT_FACT_TYPES.copy())
    similarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Load defaults from environment variables if not set."""
        env_storage = os.getenv("MEMEX_STORAGE_PATH")
        if env_storage and self.storage_path == "./memex_data":
            self.storage_path = env_storage

    @classmethod
    def from_yaml(cls, path: str | Path) -> MemexConfig:
        """Create configuration from a YAML file.

        YAML format:
            storage_path: ./memex_data
            similarity_threshold: 0.5
            fact_types:
              - name: Personal Preferences
                description: Keep track of likes and dislikes
              - name: Work Information
                description: Job titles, projects, colleagues

        Args:
            path: Path to the YAML configuration file.

        Returns:
            MemexConfig instance.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parse fact_types if provided
        fact_types = None
        if "fact_types" in data:
            fact_types = [
                FactType(name=ft["name"], description=ft["description"])
                for ft in data["fact_types"]
            ]
            del data["fact_types"]

        config = cls(**{k: v for k, v in data.items() if v is not None})

        if fact_types:
            config.fact_types = fact_types

        return config

    @classmethod
    def set_default(
        cls,
        storage_path: str = "./memex_data",
        fact_types: list[FactType] | None = None,
        similarity_threshold: float = 0.5,
    ) -> "MemexConfig":
        """Set the global default configuration.

        This allows you to configure once and use everywhere:

            MemexConfig.set_default(storage_path="./my_data")
            alice = Memory(collection="user:alice")  # uses default config
            bob = Memory(collection="user:bob")      # uses same default config

        Args:
            storage_path: Base directory for storing memory databases.
            fact_types: List of fact types to extract.
            similarity_threshold: Minimum similarity score for memory matching.

        Returns:
            The created default MemexConfig instance.
        """
        cls._default = cls(
            storage_path=storage_path,
            fact_types=fact_types or DEFAULT_FACT_TYPES.copy(),
            similarity_threshold=similarity_threshold,
        )
        return cls._default

    @classmethod
    def get_default(cls) -> "MemexConfig":
        """Get the global default configuration.

        If no default has been set, creates one with default values.

        Returns:
            The default MemexConfig instance.
        """
        if cls._default is None:
            cls._default = cls()
        return cls._default

    @classmethod
    def clear_default(cls) -> None:
        """Clear the global default configuration."""
        cls._default = None
