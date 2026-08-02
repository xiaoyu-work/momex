"""Tests for hybrid search result fusion.

Structured search produces unbounded term-match weights while embedding search
produces cosine similarities in [0, 1]. Merging them by raw score let one scale
dominate the other, so results are fused by rank instead.
"""

from momex.memory import RRF_K, Memory, SearchItem


def _item(text: str, score: float, type_: str = "topic") -> SearchItem:
    return SearchItem(type=type_, text=text, score=score, raw=None)


def _rrf(*ranks: int) -> float:
    return sum(1.0 / (RRF_K + rank + 1) for rank in ranks)


class TestFuseResults:
    def test_embedding_hit_can_outrank_larger_structured_score(self):
        """A low-magnitude cosine score must not be buried by term weights."""
        structured = [_item("entity a", 100.0), _item("entity b", 90.0)]
        embedding = [_item("message c", 0.91, "message")]

        fused = Memory._fuse_results(structured, embedding, limit=10)

        # "message c" is rank 0 of its own list, so it ties with "entity a"
        # and beats "entity b" despite a raw score of 0.91 vs 90.0.
        assert [i.text for i in fused[:2]] == ["entity a", "message c"]
        assert fused[-1].text == "entity b"

    def test_item_in_both_lists_ranks_highest(self):
        """Agreement across both indexes is the strongest signal."""
        structured = [_item("only structured", 100.0), _item("in both", 10.0)]
        embedding = [_item("only embedding", 0.9), _item("in both", 0.8)]

        fused = Memory._fuse_results(structured, embedding, limit=10)

        assert fused[0].text == "in both"
        assert fused[0].fusion_score == _rrf(1, 1)

    def test_deduplicates_by_text_keeping_first_occurrence(self):
        structured = [_item("shared", 42.0, "entity")]
        embedding = [_item("shared", 0.7, "message")]

        fused = Memory._fuse_results(structured, embedding, limit=10)

        assert len(fused) == 1
        # The structured item is kept, so .raw/.type stay the richer ones.
        assert fused[0].type == "entity"
        assert fused[0].score == 42.0

    def test_duplicate_within_one_list_counted_once(self):
        structured = [_item("dup", 5.0), _item("dup", 4.0), _item("other", 3.0)]

        fused = Memory._fuse_results(structured, limit=10)

        assert [i.text for i in fused] == ["dup", "other"]
        assert fused[0].fusion_score == _rrf(0)

    def test_fusion_scores_are_assigned(self):
        structured = [_item("a", 9.0), _item("b", 8.0)]

        fused = Memory._fuse_results(structured, limit=10)

        assert fused[0].fusion_score == _rrf(0)
        assert fused[1].fusion_score == _rrf(1)
        assert fused[0].fusion_score > fused[1].fusion_score

    def test_respects_limit(self):
        structured = [_item(f"s{i}", 10.0 - i) for i in range(5)]
        embedding = [_item(f"e{i}", 0.9 - i / 10) for i in range(5)]

        fused = Memory._fuse_results(structured, embedding, limit=3)

        assert len(fused) == 3

    def test_empty_lists(self):
        assert Memory._fuse_results([], [], limit=5) == []
