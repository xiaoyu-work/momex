"""Tests that configuration sources actually see a .env file.

Momex documents that it "automatically loads .env files from the current or
parent directory". Only Memory.__init__ did, which is too late for the
documented pattern -- `MomexConfig.from_env()` builds the whole config before
any Memory exists, so it read an environment the file had not reached yet and
raised "LLM model is required" naming a setting the user had in fact set.
"""

from pathlib import Path

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig


@pytest.fixture
def dotenv_dir(tmp_path, monkeypatch):
    """A directory containing a .env, with the process moved into it."""

    def write(**values):
        lines = [f"{k}={v}" for k, v in values.items()]
        (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    return write


def _clear(monkeypatch, *names):
    for name in names:
        monkeypatch.delenv(name, raising=False)


class TestFromEnv:
    def test_reads_settings_out_of_a_dotenv_file(self, dotenv_dir, monkeypatch):
        _clear(
            monkeypatch, "MOMEX_LLM_MODEL", "MOMEX_LLM_API_KEY", "MOMEX_LLM_PROVIDER"
        )
        dotenv_dir(
            MOMEX_LLM_PROVIDER="azure",
            MOMEX_LLM_MODEL="gpt-4.1-mini",
            MOMEX_LLM_API_KEY="from-file",
            MOMEX_LLM_API_BASE="https://example.openai.azure.com/",
        )

        config = MomexConfig.from_env()

        assert config.llm.provider == "azure"
        assert config.llm.model == "gpt-4.1-mini"
        assert config.llm.api_key == "from-file"
        config.validate()  # must not raise

    def test_searches_from_the_working_directory(self, dotenv_dir, monkeypatch):
        """Not from momex's own install location.

        Bare load_dotenv() walks up from the calling file, which for an
        installed package means out of site-packages -- never reaching the
        user's project.
        """
        _clear(monkeypatch, "MOMEX_LLM_MODEL", "MOMEX_LLM_API_KEY")
        directory = dotenv_dir(MOMEX_LLM_MODEL="found-via-cwd", MOMEX_LLM_API_KEY="k")
        nested = directory / "a" / "b"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)  # a parent directory, as documented

        assert MomexConfig.from_env().llm.model == "found-via-cwd"

    def test_a_real_environment_variable_still_wins(self, dotenv_dir, monkeypatch):
        """python-dotenv does not override what is already exported."""
        dotenv_dir(MOMEX_LLM_MODEL="from-file", MOMEX_LLM_API_KEY="k")
        monkeypatch.setenv("MOMEX_LLM_MODEL", "from-environment")

        assert MomexConfig.from_env().llm.model == "from-environment"

    def test_embedding_settings_are_read_too(self, dotenv_dir, monkeypatch):
        _clear(monkeypatch, *(f"MOMEX_EMBEDDING_{s}" for s in ("PROVIDER", "MODEL")))
        dotenv_dir(
            MOMEX_LLM_MODEL="gpt-4.1-mini",
            MOMEX_LLM_API_KEY="k",
            MOMEX_EMBEDDING_PROVIDER="azure",
            MOMEX_EMBEDDING_MODEL="text-embedding-3-small",
        )

        embedding = MomexConfig.from_env().embedding
        assert embedding is not None
        assert embedding.provider == "azure"
        assert embedding.model == "text-embedding-3-small"


class TestFromYaml:
    def test_secrets_are_recovered_from_a_dotenv_file(self, dotenv_dir, monkeypatch):
        """to_yaml() omits secrets by design, so from_yaml() must find them."""
        _clear(monkeypatch, "MOMEX_LLM_API_KEY")
        directory: Path = dotenv_dir(MOMEX_LLM_API_KEY="secret-from-file")
        (directory / "cfg.yaml").write_text(
            "llm:\n  provider: openai\n  model: gpt-4o\n", encoding="utf-8"
        )

        config = MomexConfig.from_yaml(directory / "cfg.yaml")

        assert config.llm.api_key == "secret-from-file"
        config.validate()


class TestMemory:
    def test_still_loads_dotenv_for_the_no_config_path(self, dotenv_dir, monkeypatch):
        """Memory() with no config must keep working off a .env alone."""
        _clear(monkeypatch, "MOMEX_LLM_MODEL")
        dotenv_dir(MOMEX_LLM_MODEL="gpt-4.1-mini", MOMEX_LLM_API_KEY="k")
        MomexConfig.clear_default()

        try:
            Memory(collection="test:dotenv")
            import os

            assert os.getenv("MOMEX_LLM_MODEL") == "gpt-4.1-mini"
        finally:
            MomexConfig.clear_default()

    def test_an_explicit_config_is_untouched(self, dotenv_dir, monkeypatch):
        """A caller who passes a config must not have it overridden by a file."""
        dotenv_dir(MOMEX_LLM_MODEL="from-file", MOMEX_LLM_API_KEY="from-file")
        config = MomexConfig(
            llm=LLMConfig(provider="openai", model="explicit", api_key="explicit"),
            storage=StorageConfig(path="."),
        )

        memory = Memory(collection="test:explicit", config=config)

        assert memory.config.llm.model == "explicit"
