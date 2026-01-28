# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL storage provider for TypeAgent."""

from .provider import PostgresStorageProvider
from .collections import PostgresMessageCollection, PostgresSemanticRefCollection
from .semrefindex import PostgresTermToSemanticRefIndex
from .propindex import PostgresPropertyIndex
from .timestampindex import PostgresTimestampToTextRangeIndex
from .messageindex import PostgresMessageTextIndex
from .reltermsindex import (
    PostgresRelatedTermsIndex,
    PostgresRelatedTermsAliases,
    PostgresRelatedTermsFuzzy,
)

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
