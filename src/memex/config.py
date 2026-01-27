"""Memex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

LLMProvider = Literal["openai", "azure", "ollama", "anthropic"]


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
    """

    storage_path: str = "./memex_data"
    llm_provider: LLMProvider = "openai"
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_endpoint: str | None = None
    auto_extract: bool = True
    db_name: str = "memory.db"

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
