"""Momex configuration module.

Provides structured configuration for LLM, embedding, and storage settings.
Supports code-based configuration, YAML files, and environment variables.

Environment variables use the MOMEX_ prefix with nested structure:
- LLM: MOMEX_LLM_PROVIDER, MOMEX_LLM_MODEL, MOMEX_LLM_API_KEY, MOMEX_LLM_API_BASE
- Embedding: MOMEX_EMBEDDING_PROVIDER, MOMEX_EMBEDDING_MODEL, MOMEX_EMBEDDING_API_KEY
- Storage: MOMEX_STORAGE_BACKEND, MOMEX_STORAGE_PATH, MOMEX_STORAGE_POSTGRES_URL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .exceptions import ConfigurationError


@dataclass
class LLMConfig:
    """LLM configuration.

    Attributes:
        provider: LLM provider (openai, azure, anthropic, deepseek, qwen).
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        api_key: API key for the provider.
        api_base: Base URL (required for Azure).
        temperature: Temperature for responses (default: 0.0).
    """

    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0

    def to_typeagent_config(self):
        """Convert to TypeAgent LLMConfig."""
        from typeagent.llm import LLMConfig as TALLMConfig

        return TALLMConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature,
        )

    def create_llm(self):
        """Create TypeAgent LLM instance."""
        from typeagent.llm import create_llm

        return create_llm(self.to_typeagent_config())


@dataclass
class EmbeddingConfig:
    """Embedding model configuration.

    Attributes:
        provider: Embedding provider (openai, azure). Only OpenAI-compatible providers supported.
        model: Model name (e.g., "text-embedding-3-small").
        api_key: API key. If None, attempts to reuse from LLM config.
        api_base: Base URL for the embedding API.
        dimensions: Optional embedding dimension override.
    """

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str = ""
    dimensions: int | None = None


@dataclass
class StorageConfig:
    """Storage configuration.

    Attributes:
        backend: Storage backend (sqlite, postgres).
        path: SQLite storage directory (default: "./momex_data").
        postgres_url: PostgreSQL connection URL.
        postgres_schema: PostgreSQL schema name for collection isolation.
        postgres_pool_min: Minimum pool connections (default: 2).
        postgres_pool_max: Maximum pool connections (default: 10).
    """

    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str = "./momex_data"
    postgres_url: str = ""
    postgres_schema: str = ""
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10

    @property
    def is_postgres(self) -> bool:
        return self.backend == "postgres"

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"


@dataclass
class MomexConfig:
    """Momex configuration.

    Example:
        config = MomexConfig(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key="sk-xxx",
            ),
        )

        # Anthropic LLM + OpenAI Embedding
        config = MomexConfig(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                api_key="sk-ant-xxx",
            ),
            embedding=EmbeddingConfig(
                provider="openai",
                api_key="sk-xxx",
            ),
        )

        # From environment variables
        config = MomexConfig.from_env()

        # From YAML file
        config = MomexConfig.from_yaml("config.yaml")
    """

    # Class-level default
    _default: "MomexConfig | None" = None

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig | None = None
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_env(cls) -> "MomexConfig":
        """Load configuration from MOMEX_* environment variables.

        Environment variables:
            LLM:
                MOMEX_LLM_PROVIDER - Provider (openai, azure, anthropic, deepseek, qwen)
                MOMEX_LLM_MODEL - Model name (required)
                MOMEX_LLM_API_KEY - API key (required)
                MOMEX_LLM_API_BASE - Base URL (required for Azure)
                MOMEX_LLM_TEMPERATURE - Temperature (default: 0.0)

            Embedding (optional, auto-inferred if not set):
                MOMEX_EMBEDDING_PROVIDER - Provider (openai, azure)
                MOMEX_EMBEDDING_MODEL - Model name
                MOMEX_EMBEDDING_API_KEY - API key (defaults to LLM key if compatible)
                MOMEX_EMBEDDING_API_BASE - Base URL
                MOMEX_EMBEDDING_DIMENSIONS - Embedding dimensions

            Storage:
                MOMEX_STORAGE_BACKEND - Backend (sqlite, postgres)
                MOMEX_STORAGE_PATH - SQLite storage path
                MOMEX_STORAGE_POSTGRES_URL - PostgreSQL URL
                MOMEX_STORAGE_POSTGRES_SCHEMA - PostgreSQL schema
        """
        # LLM config
        llm = LLMConfig(
            provider=os.getenv("MOMEX_LLM_PROVIDER", "openai"),
            model=os.getenv("MOMEX_LLM_MODEL", ""),
            api_key=os.getenv("MOMEX_LLM_API_KEY", ""),
            api_base=os.getenv("MOMEX_LLM_API_BASE", ""),
            temperature=float(os.getenv("MOMEX_LLM_TEMPERATURE", "0.0")),
        )

        # Embedding config (optional)
        embedding = None
        if os.getenv("MOMEX_EMBEDDING_PROVIDER"):
            embedding = EmbeddingConfig(
                provider=os.getenv("MOMEX_EMBEDDING_PROVIDER", "openai"),
                model=os.getenv("MOMEX_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("MOMEX_EMBEDDING_API_KEY", ""),
                api_base=os.getenv("MOMEX_EMBEDDING_API_BASE", ""),
                dimensions=int(os.getenv("MOMEX_EMBEDDING_DIMENSIONS", "0")) or None,
            )

        # Storage config
        backend = os.getenv("MOMEX_STORAGE_BACKEND", "sqlite")
        if backend not in ("sqlite", "postgres"):
            backend = "sqlite"

        storage = StorageConfig(
            backend=backend,  # type: ignore
            path=os.getenv("MOMEX_STORAGE_PATH", "./momex_data"),
            postgres_url=os.getenv("MOMEX_STORAGE_POSTGRES_URL", ""),
            postgres_schema=os.getenv("MOMEX_STORAGE_POSTGRES_SCHEMA", ""),
        )

        return cls(llm=llm, embedding=embedding, storage=storage)

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not self.llm.model:
            raise ConfigurationError(
                message="LLM model is required",
                suggestion="Set llm.model in MomexConfig or use MomexConfig.from_env()",
            )
        if not self.llm.api_key:
            raise ConfigurationError(
                message="LLM API key is required",
                suggestion="Set llm.api_key in MomexConfig or use MomexConfig.from_env()",
            )
        if self.llm.provider == "azure" and not self.llm.api_base:
            raise ConfigurationError(
                message="LLM api_base is required for Azure",
                suggestion="Set llm.api_base in MomexConfig",
            )
        if self.storage.is_postgres and not self.storage.postgres_url:
            raise ConfigurationError(
                message="PostgreSQL URL is required",
                suggestion="Set storage.postgres_url in MomexConfig",
            )

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration, auto-inferring if not explicitly set.

        If embedding is not configured:
        - For OpenAI/Azure LLM: reuse LLM credentials
        - For other providers: raise error (embedding must be configured separately)

        Returns:
            EmbeddingConfig with resolved settings.

        Raises:
            ConfigurationError: If embedding cannot be auto-inferred.
        """
        if self.embedding:
            cfg = EmbeddingConfig(
                provider=self.embedding.provider,
                model=self.embedding.model,
                api_key=self.embedding.api_key,
                api_base=self.embedding.api_base,
                dimensions=self.embedding.dimensions,
            )
            # If embedding API key not set, try to reuse LLM key if providers match
            if not cfg.api_key and cfg.provider == self.llm.provider:
                cfg.api_key = self.llm.api_key
                if not cfg.api_base:
                    cfg.api_base = self.llm.api_base
            return cfg

        # Auto-infer from LLM config
        if self.llm.provider in ("openai", "azure"):
            return EmbeddingConfig(
                provider=self.llm.provider,
                api_key=self.llm.api_key,
                api_base=self.llm.api_base,
            )
        else:
            raise ConfigurationError(
                message=f"LLM provider '{self.llm.provider}' doesn't support embeddings",
                suggestion="Set embedding config with MOMEX_EMBEDDING_* env vars or embedding parameter",
            )

    def create_embedding_model(self):
        """Create TypeAgent embedding model.

        Returns:
            AsyncEmbeddingModel instance.
        """
        from typeagent.aitools.embeddings import AsyncEmbeddingModel

        cfg = self.get_embedding_config()

        # Set API keys for the embedding model (it reads from env vars)
        if cfg.api_key:
            if cfg.provider == "azure":
                if not os.getenv("AZURE_OPENAI_API_KEY"):
                    os.environ["AZURE_OPENAI_API_KEY"] = cfg.api_key
            else:
                if not os.getenv("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = cfg.api_key

        return AsyncEmbeddingModel(
            embedding_size=cfg.dimensions,
            model_name=cfg.model or None,
            endpoint_envvar=None,
        )

    def create_llm(self):
        """Create TypeAgent LLM instance."""
        return self.llm.create_llm()

    def get_llm_config(self):
        """Get TypeAgent LLMConfig."""
        return self.llm.to_typeagent_config()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MomexConfig":
        """Load from YAML file.

        YAML format:
            llm:
              provider: openai
              model: gpt-4o
              api_key: sk-xxx  # or use env var
              api_base: ""  # required for azure
              temperature: 0.0

            embedding:  # optional
              provider: openai
              model: text-embedding-3-small
              api_key: ""  # defaults to llm.api_key if same provider

            storage:
              backend: sqlite  # or postgres
              path: ./momex_data
              postgres_url: ""
              postgres_schema: ""
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise ConfigurationError(message=f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parse LLM config
        llm_data = data.get("llm", {})
        llm = LLMConfig(
            provider=llm_data.get("provider", "openai"),
            model=llm_data.get("model", ""),
            api_key=llm_data.get("api_key", ""),
            api_base=llm_data.get("api_base", ""),
            temperature=float(llm_data.get("temperature", 0.0)),
        )

        # Parse embedding config (optional)
        embedding = None
        if "embedding" in data:
            emb_data = data["embedding"]
            embedding = EmbeddingConfig(
                provider=emb_data.get("provider", "openai"),
                model=emb_data.get("model", "text-embedding-3-small"),
                api_key=emb_data.get("api_key", ""),
                api_base=emb_data.get("api_base", ""),
                dimensions=emb_data.get("dimensions"),
            )

        # Parse storage config
        storage_data = data.get("storage", {})
        backend = storage_data.get("backend", "sqlite")
        if backend not in ("sqlite", "postgres"):
            backend = "sqlite"

        storage = StorageConfig(
            backend=backend,  # type: ignore
            path=storage_data.get("path", "./momex_data"),
            postgres_url=storage_data.get("postgres_url", ""),
            postgres_schema=storage_data.get("postgres_schema", ""),
            postgres_pool_min=storage_data.get("postgres_pool_min", 2),
            postgres_pool_max=storage_data.get("postgres_pool_max", 10),
        )

        return cls(llm=llm, embedding=embedding, storage=storage)

    def to_yaml(self, path: str | Path) -> None:
        """Save to YAML file."""
        import yaml

        data: dict = {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": self.llm.api_key,
                "temperature": self.llm.temperature,
            }
        }

        if self.llm.api_base:
            data["llm"]["api_base"] = self.llm.api_base

        if self.embedding:
            data["embedding"] = {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
            }
            if self.embedding.api_key:
                data["embedding"]["api_key"] = self.embedding.api_key
            if self.embedding.api_base:
                data["embedding"]["api_base"] = self.embedding.api_base
            if self.embedding.dimensions:
                data["embedding"]["dimensions"] = self.embedding.dimensions

        data["storage"] = {
            "backend": self.storage.backend,
        }
        if self.storage.is_sqlite:
            data["storage"]["path"] = self.storage.path
        else:
            data["storage"]["postgres_url"] = self.storage.postgres_url
            if self.storage.postgres_schema:
                data["storage"]["postgres_schema"] = self.storage.postgres_schema
            data["storage"]["postgres_pool_min"] = self.storage.postgres_pool_min
            data["storage"]["postgres_pool_max"] = self.storage.postgres_pool_max

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    @classmethod
    def set_default(
        cls,
        llm: LLMConfig | None = None,
        embedding: EmbeddingConfig | None = None,
        storage: StorageConfig | None = None,
    ) -> "MomexConfig":
        """Set global default configuration.

        Args:
            llm: LLM configuration (required).
            embedding: Embedding configuration (optional, auto-inferred).
            storage: Storage configuration (optional, defaults to SQLite).

        Returns:
            The configured MomexConfig instance.
        """
        cls._default = cls(
            llm=llm or LLMConfig(),
            embedding=embedding,
            storage=storage or StorageConfig(),
        )
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

    # Convenience properties for backward compatibility in internal code
    @property
    def is_postgres(self) -> bool:
        return self.storage.is_postgres

    @property
    def is_sqlite(self) -> bool:
        return self.storage.is_sqlite

    @property
    def storage_path(self) -> str:
        return self.storage.path
