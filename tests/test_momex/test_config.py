"""Tests for MomexConfig construction, defaults, and provider resolution."""

import pytest

from momex import EmbeddingConfig, LLMConfig, MomexConfig, StorageConfig
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
