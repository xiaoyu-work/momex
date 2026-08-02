"""Tests for MomexConfig construction, defaults, and provider resolution."""

import pytest

from momex import EmbeddingConfig, LLMConfig, Memory, MomexConfig, StorageConfig
from momex.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _clear_global_default():
    MomexConfig.clear_default()
    yield
    MomexConfig.clear_default()


class TestMomexConfigConstruction:
    def test_positional_argument_binds_to_llm(self):
        """MomexConfig(LLMConfig(...)) must set llm, not a private field.

        _default used to be an ordinary annotated attribute, which made it the
        first positional dataclass field and silently swallowed this argument.
        """
        config = MomexConfig(LLMConfig(provider="openai", model="gpt-4o", api_key="k"))
        assert config.llm.model == "gpt-4o"
        assert config.llm.api_key == "k"
        config.validate()

    def test_default_is_not_a_dataclass_field(self):
        import dataclasses

        names = [f.name for f in dataclasses.fields(MomexConfig)]
        assert names == ["llm", "embedding", "storage"]
        assert "_default" not in repr(MomexConfig())

    def test_set_and_clear_default(self):
        MomexConfig.set_default(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
            storage=StorageConfig(path="/custom/default"),
        )
        assert MomexConfig.get_default().storage.path == "/custom/default"

        MomexConfig.clear_default()
        assert MomexConfig.get_default().storage.path == "./momex_data"


class TestEmbeddingResolution:
    def test_api_version_is_preserved(self):
        """Azure embeddings need api_version to survive get_embedding_config()."""
        config = MomexConfig(
            llm=LLMConfig(
                provider="azure", model="gpt-4o", api_key="k", api_base="https://llm"
            ),
            embedding=EmbeddingConfig(
                provider="azure",
                model="text-embedding-3-small",
                api_key="ek",
                api_base="https://emb",
                api_version="2024-02-01",
            ),
        )
        resolved = config.get_embedding_config()
        assert resolved.api_version == "2024-02-01"
        assert resolved.api_base == "https://emb"
        assert resolved.api_key == "ek"

    def test_credentials_inherited_from_llm_when_provider_matches(self):
        config = MomexConfig(
            llm=LLMConfig(
                provider="azure",
                model="gpt-4o",
                api_key="shared",
                api_base="https://llm",
            ),
            embedding=EmbeddingConfig(provider="azure"),
        )
        resolved = config.get_embedding_config()
        assert resolved.api_key == "shared"
        assert resolved.api_base == "https://llm"

    def test_inferred_from_llm_when_embedding_absent(self):
        config = MomexConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k")
        )
        resolved = config.get_embedding_config()
        assert resolved.provider == "openai"
        assert resolved.api_key == "k"

    def test_non_embedding_provider_raises(self):
        config = MomexConfig(
            llm=LLMConfig(provider="anthropic", model="claude", api_key="k")
        )
        with pytest.raises(ConfigurationError):
            config.get_embedding_config()


class TestFromEnv:
    def test_embedding_built_without_provider_var(self, monkeypatch):
        """Any MOMEX_EMBEDDING_* var should produce an embedding config."""
        monkeypatch.setenv("MOMEX_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("MOMEX_LLM_API_KEY", "k")
        monkeypatch.setenv("MOMEX_EMBEDDING_MODEL", "text-embedding-3-large")

        config = MomexConfig.from_env()
        assert config.embedding is not None
        assert config.embedding.model == "text-embedding-3-large"

    def test_embedding_api_version_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MOMEX_EMBEDDING_PROVIDER", "azure")
        monkeypatch.setenv("MOMEX_EMBEDDING_API_VERSION", "2024-02-01")

        config = MomexConfig.from_env()
        assert config.embedding is not None
        assert config.embedding.api_version == "2024-02-01"
        assert config.get_embedding_config().api_version == "2024-02-01"

    def test_no_embedding_vars_leaves_embedding_none(self, monkeypatch):
        for key in (
            "PROVIDER",
            "MODEL",
            "API_KEY",
            "API_BASE",
            "API_VERSION",
            "DIMENSIONS",
        ):
            monkeypatch.delenv(f"MOMEX_EMBEDDING_{key}", raising=False)

        assert MomexConfig.from_env().embedding is None

    def test_postgres_pool_sizes_read_from_env(self, monkeypatch):
        monkeypatch.setenv("MOMEX_STORAGE_BACKEND", "postgres")
        monkeypatch.setenv("MOMEX_STORAGE_POSTGRES_POOL_MIN", "5")
        monkeypatch.setenv("MOMEX_STORAGE_POSTGRES_POOL_MAX", "25")

        storage = MomexConfig.from_env().storage
        assert storage.postgres_pool_min == 5
        assert storage.postgres_pool_max == 25

    def test_malformed_numeric_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MOMEX_LLM_TEMPERATURE", "not-a-number")
        monkeypatch.setenv("MOMEX_STORAGE_POSTGRES_POOL_MAX", "lots")

        config = MomexConfig.from_env()
        assert config.llm.temperature == 0.0
        assert config.storage.postgres_pool_max == 10


class TestYamlRoundTrip:
    def test_api_version_round_trips(self, tmp_path):
        config = MomexConfig(
            llm=LLMConfig(
                provider="azure", model="gpt-4o", api_key="k", api_base="https://llm"
            ),
            embedding=EmbeddingConfig(
                provider="azure",
                model="text-embedding-3-small",
                api_base="https://emb",
                api_version="2024-02-01",
            ),
        )
        path = tmp_path / "config.yaml"
        config.to_yaml(path)

        loaded = MomexConfig.from_yaml(path)
        assert loaded.embedding is not None
        assert loaded.embedding.api_version == "2024-02-01"
        assert loaded.embedding.api_base == "https://emb"


class TestGlobalDefaultIsHonored:
    """MemoryManager must resolve the same global default as Memory/query."""

    def test_manager_uses_global_default(self, tmp_path):
        MomexConfig.set_default(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
            storage=StorageConfig(path=str(tmp_path)),
        )
        try:
            from momex import MemoryManager

            assert MemoryManager()._storage_path == tmp_path
            assert Memory(collection="c").config.storage.path == str(tmp_path)
        finally:
            MomexConfig.clear_default()

    def test_manager_falls_back_to_plain_default(self, tmp_path):
        from momex import MemoryManager

        MomexConfig.clear_default()
        assert str(MemoryManager()._storage_path) == "momex_data"


class TestSecretHandling:
    """to_yaml() must not persist credentials unless explicitly asked."""

    def _config(self):
        return MomexConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-secret"),
            embedding=EmbeddingConfig(provider="openai", api_key="sk-emb-secret"),
            storage=StorageConfig(
                backend="postgres",
                postgres_url="postgresql://u:pw@localhost:5432/momex",
            ),
        )

    def test_secrets_omitted_by_default(self, tmp_path):
        path = tmp_path / "config.yaml"
        self._config().to_yaml(path)

        text = path.read_text(encoding="utf-8")
        assert "sk-secret" not in text
        assert "sk-emb-secret" not in text
        assert "pw@localhost" not in text
        # Non-secret settings are still written.
        assert "gpt-4o" in text
        assert "postgres" in text

    def test_include_secrets_writes_them(self, tmp_path):
        path = tmp_path / "config.yaml"
        self._config().to_yaml(path, include_secrets=True)

        text = path.read_text(encoding="utf-8")
        assert "sk-secret" in text
        assert "sk-emb-secret" in text
        assert "postgresql://u:pw@localhost:5432/momex" in text

    def test_from_yaml_recovers_secrets_from_env(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        self._config().to_yaml(path)

        monkeypatch.setenv("MOMEX_LLM_API_KEY", "sk-from-env")
        monkeypatch.setenv("MOMEX_EMBEDDING_API_KEY", "sk-emb-from-env")
        monkeypatch.setenv(
            "MOMEX_STORAGE_POSTGRES_URL", "postgresql://u:pw@db:5432/momex"
        )

        loaded = MomexConfig.from_yaml(path)
        assert loaded.llm.api_key == "sk-from-env"
        assert loaded.embedding is not None
        assert loaded.embedding.api_key == "sk-emb-from-env"
        assert loaded.storage.postgres_url == "postgresql://u:pw@db:5432/momex"
        loaded.validate()

    def test_file_value_wins_over_env(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        self._config().to_yaml(path, include_secrets=True)

        monkeypatch.setenv("MOMEX_LLM_API_KEY", "sk-from-env")

        assert MomexConfig.from_yaml(path).llm.api_key == "sk-secret"
