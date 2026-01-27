"""Memex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LLMProvider = Literal["openai", "azure", "ollama", "anthropic"]


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

    Attributes:
        storage_path: Base directory for storing memory databases.
            Defaults to "./memex_data".
        llm_provider: LLM provider to use. One of "openai", "azure", "ollama", "anthropic".
            Defaults to "openai".
        llm_model: Model name to use. Defaults to environment variable or "gpt-4o".
        llm_api_key: API key for the LLM provider. Defaults to environment variable.
        llm_endpoint: Custom endpoint URL for LLM API. Optional.
        auto_extract: Whether to automatically extract knowledge from messages.
            Defaults to True.
        db_name: Database filename. Defaults to "memory.db".
        fact_types: List of fact types to extract. Defaults to DEFAULT_FACT_TYPES.
        similarity_threshold: Minimum similarity score for memory matching. Defaults to 0.5.
    """

    storage_path: str = "./memex_data"
    llm_provider: LLMProvider = "openai"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_endpoint: str | None = None
    auto_extract: bool = True
    db_name: str = "memory.db"
    fact_types: list[FactType] = field(default_factory=lambda: DEFAULT_FACT_TYPES.copy())
    similarity_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Load defaults from environment variables if not set."""
        if self.llm_model is None:
            self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        if self.llm_api_key is None:
            if self.llm_provider == "azure":
                self.llm_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            elif self.llm_provider == "anthropic":
                self.llm_api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                self.llm_api_key = os.getenv("OPENAI_API_KEY")

        if self.llm_endpoint is None:
            if self.llm_provider == "azure":
                self.llm_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            elif self.llm_provider == "ollama":
                self.llm_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")

        # Handle environment variable for storage path
        env_storage = os.getenv("MEMEX_STORAGE_PATH")
        if env_storage and self.storage_path == "./memex_data":
            self.storage_path = env_storage

    @classmethod
    def from_env(cls) -> MemexConfig:
        """Create configuration from environment variables.

        Environment variables:
            MEMEX_STORAGE_PATH: Base storage path
            MEMEX_LLM_PROVIDER: LLM provider (openai, azure, ollama, anthropic)
            OPENAI_MODEL: Model name
            OPENAI_API_KEY: OpenAI API key
            AZURE_OPENAI_API_KEY: Azure OpenAI API key
            AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint

        Returns:
            MemexConfig instance.
        """
        return cls(
            storage_path=os.getenv("MEMEX_STORAGE_PATH", "./memex_data"),
            llm_provider=os.getenv("MEMEX_LLM_PROVIDER", "openai"),  # type: ignore
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MemexConfig:
        """Create configuration from a YAML file.

        YAML format:
            storage_path: ./memex_data
            llm_provider: openai
            llm_model: gpt-4o
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
