"""Tests for tombstone (deleted semantic ref) encoding.

Tombstones are persisted in conversation metadata and grow with every
deletion, so they are stored range-encoded. The decoder must still accept the
older flat-integer-list format written by previous versions.
"""

import json

import pytest

from momex.ledger import decode_deleted_ids, encode_deleted_ids


class TestEncoding:
    def test_empty(self):
        assert encode_deleted_ids(set()) == []

    def test_isolated_ids_stay_scalars(self):
        assert encode_deleted_ids({1, 5, 9}) == [1, 5, 9]

    def test_consecutive_run_becomes_a_range(self):
        assert encode_deleted_ids({1, 2, 3, 4}) == [[1, 4]]

    def test_mixed_runs_and_singletons(self):
        assert encode_deleted_ids({1, 2, 3, 7, 10, 11}) == [[1, 3], 7, [10, 11]]

    def test_runs_compress_the_payload(self):
        """The point of the exercise: contiguous deletions stay small."""
        ids = set(range(10_000))
        flat = len(json.dumps(sorted(ids)))
        encoded = len(json.dumps(encode_deleted_ids(ids)))

        assert encoded < flat / 100


class TestRoundTrip:
    @pytest.mark.parametrize(
        "ids",
        [
            set(),
            {0},
            {1, 5, 9},
            {1, 2, 3, 4},
            {1, 2, 3, 7, 10, 11},
            set(range(500)) | {1000, 2000, 2001},
        ],
    )
    def test_round_trip(self, ids):
        encoded = json.dumps(encode_deleted_ids(ids))
        assert decode_deleted_ids(json.loads(encoded)) == ids


class TestBackwardCompatibility:
    def test_reads_legacy_flat_list(self):
        assert decode_deleted_ids([1, 2, 3]) == {1, 2, 3}

    def test_reads_legacy_string_digits(self):
        assert decode_deleted_ids(["4", "5"]) == {4, 5}

    def test_reads_mixed_legacy_and_range(self):
        assert decode_deleted_ids([1, [3, 5], "7"]) == {1, 3, 4, 5, 7}


class TestMalformedInput:
    def test_non_list_yields_empty(self):
        assert decode_deleted_ids({"a": 1}) == set()
        assert decode_deleted_ids(None) == set()

    def test_bad_entries_are_skipped(self):
        assert decode_deleted_ids([1, "abc", None, [1, 2, 3], {}, [5, 6]]) == {
            1,
            5,
            6,
        }

    def test_inverted_range_skipped(self):
        assert decode_deleted_ids([[9, 2]]) == set()

    def test_absurd_range_skipped(self):
        assert decode_deleted_ids([[0, 10**12]]) == set()

    def test_booleans_are_not_treated_as_ints(self):
        assert decode_deleted_ids([True, False]) == set()
