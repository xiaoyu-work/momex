# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL storage provider for TypeAgent."""

from .collections import PostgresMessageCollection, PostgresSemanticRefCollection
from .messageindex import PostgresMessageTextIndex
from .propindex import PostgresPropertyIndex
from .provider import PostgresStorageProvider
from .reltermsindex import (
    PostgresRelatedTermsAliases,
    PostgresRelatedTermsFuzzy,
    PostgresRelatedTermsIndex,
)
from .semrefindex import PostgresTermToSemanticRefIndex
from .timestampindex import PostgresTimestampToTextRangeIndex

__all__ = [
    "PostgresStorageProvider",
    "PostgresMessageCollection",
    "PostgresSemanticRefCollection",
    "PostgresTermToSemanticRefIndex",
    "PostgresPropertyIndex",
    "PostgresTimestampToTextRangeIndex",
    "PostgresMessageTextIndex",
    "PostgresRelatedTermsIndex",
    "PostgresRelatedTermsAliases",
    "PostgresRelatedTermsFuzzy",
]
