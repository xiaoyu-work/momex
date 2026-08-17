"""Tests that a collection name cannot escape the storage directory.

Collection names are caller-supplied, and the multi-tenant pattern the README
advertises (``user:<username>``) means they usually come from user input. The
sanitizer used to replace only the Windows-forbidden characters, so ".." and
"/" survived: ``user:../../pwned`` resolved outside ``storage_path``, and
``MemoryManager.delete("..:victim")`` really did remove an unrelated directory.
"""

from pathlib import Path

import pytest

from momex import (
    LLMConfig,
    Memory,
    MemoryManager,
    MomexConfig,
    StorageConfig,
    ValidationError,
)
from momex.memory import _collection_to_db_path, _collection_to_path

TRAVERSING = [
    "..",
    "user:..",
    "..:victim",
    "user:.",
    "user:",
    "",
    ":",
    "user: ",
]


def _config(path) -> MomexConfig:
    return MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(path)),
    )


class TestCollectionToPath:
    @pytest.mark.parametrize("collection", TRAVERSING)
    def test_traversing_names_are_rejected(self, collection):
        with pytest.raises(ValidationError):
            _collection_to_path(collection)

    def test_separators_stay_within_one_segment(self):
        """A ':' segment is exactly one directory, whatever it contains."""
        assert _collection_to_path("user:a/b") == Path("user", "a_b")
        assert _collection_to_path("user:a\\b") == Path("user", "a_b")

    def test_embedded_traversal_is_neutralized(self):
        """Separators are replaced, so "../../pwned" becomes one literal name."""
        assert _collection_to_path("user:../../pwned") == Path("user", ".._.._pwned")

    def test_ordinary_names_are_unchanged(self):
        assert _collection_to_path("user:xiaoyuzhang") == Path("user", "xiaoyuzhang")
        assert _collection_to_path("momex:engineering:x") == Path(
            "momex", "engineering", "x"
        )

    def test_leading_dots_are_still_allowed(self):
        """Only dot-*only* segments are bogus; ".hidden" is a fine name."""
        assert _collection_to_path("user:...hidden") == Path("user", "...hidden")

    def test_db_path_stays_under_the_storage_root(self):
        root = Path("/srv/momex_data").resolve()
        for collection in ("user:a/b", "user:../../pwned"):
            path = _collection_to_db_path(collection, str(root), "memory.db")
            assert root in path.resolve().parents


class TestMemoryRejectsTraversal:
    @pytest.mark.parametrize("collection", TRAVERSING)
    def test_db_path_rejects_traversal(self, tmp_path, collection):
        memory = Memory(collection=collection, config=_config(tmp_path))
        with pytest.raises(ValidationError):
            memory.db_path


class TestManagerRejectsTraversal:
    @pytest.mark.parametrize("collection", TRAVERSING)
    def test_exists_rejects_traversal(self, tmp_path, collection):
        manager = MemoryManager(config=_config(tmp_path / "store"))
        with pytest.raises(ValidationError):
            manager.exists(collection)

    def test_delete_cannot_remove_a_sibling_directory(self, tmp_path):
        """The regression: this used to rmtree a directory outside the root."""
        root = tmp_path / "store"
        root.mkdir()
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "important.txt").write_text("do not delete")

        manager = MemoryManager(config=_config(root))
        with pytest.raises(ValidationError):
            manager.delete("..:victim")

        assert (victim / "important.txt").exists()

    def test_rename_and_copy_reject_traversal(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        manager = MemoryManager(config=_config(root))

        with pytest.raises(ValidationError):
            manager.rename("..:victim", "user:new")
        with pytest.raises(ValidationError):
            manager.copy("user:src", "..:victim")
