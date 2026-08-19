"""Tests that a memory can say when it happened.

add() stamped every message with the wall clock at ingestion. That is right
for something said as it is stored and wrong for everything else: an imported
chat log, a migrated database, a summary written after the fact. Loading a
conversation that ran from May to October 2023 produced 419 messages all
timestamped within the eight minutes the import took.

Nothing failed visibly. The timestamp index still built, the time-range prompt
still went to the query translator -- reporting those eight minutes as the
period the collection covers -- and every temporal question was answered
against the moment of ingestion.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.timewindow import validate_timestamp


class _Result:
    messages_added = 1
    semrefs_added = 0


class _IndexSettings:
    auto_extract_knowledge = True


class _Settings:
    def __init__(self):
        self.semantic_ref_index_settings = _IndexSettings()


class _Conversation:
    def __init__(self):
        self.settings = _Settings()
        self.added: list = []

    async def add_messages_with_indexing(self, messages):
        self.added.extend(messages)
        return _Result()


@pytest.fixture
def memory(tmp_path):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    mem = Memory(collection="test:when", config=config)
    mem._conversation = _Conversation()  # type: ignore[assignment]
    mem._initialized = True
    mem._ledger._records = []
    return mem


class TestValidateTimestamp:
    def test_normalizes_to_the_stored_form(self):
        assert validate_timestamp("2023-05-08T13:56:00Z") == "2023-05-08T13:56:00Z"

    def test_accepts_a_bare_date_as_midnight(self):
        assert validate_timestamp("2023-05-08") == "2023-05-08T00:00:00Z"

    def test_converts_an_offset_to_utc(self):
        assert validate_timestamp("2023-05-08T13:56:00+02:00") == (
            "2023-05-08T11:56:00Z"
        )

    def test_assumes_utc_when_no_zone_is_given(self):
        assert validate_timestamp("2023-05-08T13:56:00") == "2023-05-08T13:56:00Z"

    @pytest.mark.parametrize(
        "bad", ["yesterday", "8 May 2023", "", "2023-13-45T00:00:00Z"]
    )
    def test_rejects_what_typeagent_cannot_parse_back(self, bad):
        """A value the index cannot read costs the collection its timeline."""
        with pytest.raises(ValueError):
            validate_timestamp(bad)


class TestAdd:
    @pytest.mark.asyncio
    async def test_records_when_the_memory_happened(self, memory):
        await memory.add(
            "Caroline went to the support group",
            infer=False,
            timestamp="2023-05-07T14:30:00Z",
        )

        (message,) = memory._conversation.added
        assert message.timestamp == "2023-05-07T14:30:00Z"

    @pytest.mark.asyncio
    async def test_defaults_to_now(self, memory):
        """Still right for a memory being stored as it is made."""
        await memory.add("I like sushi", infer=False)

        (message,) = memory._conversation.added
        assert message.timestamp.endswith("Z")
        assert message.timestamp.startswith("20")

    @pytest.mark.asyncio
    async def test_applies_to_every_message_in_one_call(self, memory):
        await memory.add(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
            infer=False,
            timestamp="2023-05-08",
        )

        stamps = {m.timestamp for m in memory._conversation.added}
        assert stamps == {"2023-05-08T00:00:00Z"}

    @pytest.mark.asyncio
    async def test_a_bad_timestamp_is_rejected_before_anything_is_written(
        self, memory
    ):
        with pytest.raises(ValueError):
            await memory.add("anything", infer=False, timestamp="last Tuesday")

        assert memory._conversation.added == []

    @pytest.mark.asyncio
    async def test_a_backfill_keeps_its_own_order(self, memory):
        """The case the default broke: imported history spanning months."""
        for day, text in (("2023-05-08", "first"), ("2023-10-22", "last")):
            await memory.add(text, infer=False, timestamp=day)

        stamps = [m.timestamp for m in memory._conversation.added]
        assert stamps == ["2023-05-08T00:00:00Z", "2023-10-22T00:00:00Z"]
