"""Memex configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

LLMProvider = Literal["openai", "azure", "ollama", "anthropic"]


@dataclass
class MemexConfig:
    """Configuration for Memex memory system.

    Attributes:
        storage_path: Base directory for storing memory databases.
            Defaults to "./memex_data".
        storage_path_template: Template for tenant-specific paths.
            Supports {org_id}, {user_id}, {agent_id} placeholders.
            Defaults to "{org_id}/{user_id}".
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
    storage_path_template: str = "{org_id}/{user_id}"
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

    def get_db_path(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        org_id: str | None = None,
    ) -> str:
        """Get the full database path for a tenant.

        Args:
            user_id: User identifier.
            agent_id: Agent identifier.
            org_id: Organization identifier.

        Returns:
            Full path to the SQLite database file.
        """
        # Build the tenant-specific subdirectory
        template_values = {
            "user_id": user_id or "default",
            "agent_id": agent_id or "default",
            "org_id": org_id or "default",
        }

        # Only include non-default segments in path
        path_parts = []
        if org_id:
            path_parts.append(org_id)
        if user_id:
            path_parts.append(user_id)
        if agent_id:
            path_parts.append(agent_id)

        if path_parts:
            tenant_path = os.path.join(*path_parts)
        else:
            tenant_path = "default"

        full_path = os.path.join(self.storage_path, tenant_path, self.db_name)

        # Ensure directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        return full_path

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
