# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from openai import AsyncAzureOpenAI, AsyncOpenAI, DEFAULT_MAX_RETRIES
from openai.types import Embedding
import tiktoken
from tiktoken import model as tiktoken_model
from tiktoken.core import Encoding

from .auth import AzureTokenProvider, get_shared_token_provider
from .utils import timelog

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


DEFAULT_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_SIZE = 1536  # Default embedding size (required for ada-002)
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"
TEST_MODEL_NAME = "test"
MAX_BATCH_SIZE = 2048
MAX_TOKEN_SIZE = 4096
MAX_TOKENS_PER_BATCH = 300_000
MAX_CHAR_SIZE = MAX_TOKEN_SIZE * 3
MAX_CHARS_PER_BATCH = MAX_TOKENS_PER_BATCH * 3

model_to_embedding_size_and_envvar: dict[str, tuple[int | None, str]] = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-3-small": (1536, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-3-large": (3072, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    TEST_MODEL_NAME: (3, "SIR_NOT_APPEARING_IN_THIS_FILM"),
}

model_to_embedding_size: dict[str, int] = {
    model: size
    for model, (size, _) in model_to_embedding_size_and_envvar.items()
    if size is not None
}

model_to_envvar: dict[str, str] = {
    model: envvar
    for model, (_, envvar) in model_to_embedding_size_and_envvar.items()
    if model != TEST_MODEL_NAME
}


@runtime_checkable
class IEmbedder(Protocol):
    """Minimal provider interface for embedding models.

    Implement this protocol to add support for a new embedding provider
    (e.g. Anthropic, Gemini, local models). Only raw embedding computation
    is required; caching is handled by :class:`CachingEmbeddingModel`.
    """

    @property
    def model_name(self) -> str: ...

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        """Compute a single embedding without caching."""
        ...

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        """Compute embeddings for a batch of strings.

        Raises :class:`ValueError` if *input* is empty.
        """
        ...


@runtime_checkable
class IEmbeddingModel(Protocol):
    """Consumer-facing interface for embedding models with caching."""

    @property
    def model_name(self) -> str: ...

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        """Cache an already-computed embedding under the given key."""
        ...

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        """Compute a single embedding without caching."""
        ...

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        """Compute embeddings for a batch of strings."""
        ...

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        """Retrieve a single embedding, using cache if available."""
        ...

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        """Retrieve embeddings for multiple keys, using cache if available."""
        ...


class CachingEmbeddingModel:
    """Wraps an :class:`IEmbedder` with an in-memory embedding cache."""

    def __init__(self, embedder: IEmbedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, NormalizedEmbedding] = {}

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def embedding_size(self) -> int:
        embedding_size = getattr(self._embedder, "embedding_size", None)
        if embedding_size is not None:
            return int(embedding_size)
        return model_to_embedding_size.get(self.model_name, DEFAULT_EMBEDDING_SIZE)

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        existing = self._cache.get(key)
        if existing is not None:
            assert np.array_equal(existing, embedding)
        else:
            self._cache[key] = embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        return await self._embedder.get_embedding_nocache(input)

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        return await self._embedder.get_embeddings_nocache(input)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = await self._embedder.get_embedding_nocache(key)
        self._cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        if not keys:
            raise ValueError("Cannot embed an empty list")
        missing_keys = [k for k in keys if k not in self._cache]
        if missing_keys:
            fresh = await self._embedder.get_embeddings_nocache(missing_keys)
            for i, k in enumerate(missing_keys):
                self._cache[k] = fresh[i]
        return np.array([self._cache[k] for k in keys], dtype=np.float32)


class AsyncEmbeddingModel:
    model_name: str
    embedding_size: int
    endpoint_envvar: str
    use_azure: bool
    azure_token_provider: AzureTokenProvider | None
    async_client: AsyncOpenAI | None
    azure_endpoint: str
    azure_api_version: str
    encoding: Encoding | None
    max_chunk_size: int
    max_size_per_batch: int

    _embedding_cache: dict[str, NormalizedEmbedding]

    def __init__(
        self,
        embedding_size: int | None = None,
        model_name: str | None = None,
        endpoint_envvar: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key: str | None = None,
        api_base: str | None = None,
        provider: str | None = None,
        api_version: str | None = None,
    ):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        self.model_name = model_name

        suggested_embedding_size, suggested_endpoint_envvar = (
            model_to_embedding_size_and_envvar.get(model_name, (None, None))
        )

        if embedding_size is None:
            embedding_size = (
                suggested_embedding_size
                if suggested_embedding_size is not None
                else DEFAULT_EMBEDDING_SIZE
            )
        self.embedding_size = embedding_size

        if (
            model_name == DEFAULT_MODEL_NAME
            and embedding_size != DEFAULT_EMBEDDING_SIZE
        ):
            raise ValueError(
                f"Cannot customize embedding_size for default model {DEFAULT_MODEL_NAME}"
            )

        openai_api_key = (
            api_key if provider == "openai" else os.getenv("OPENAI_API_KEY")
        )
        azure_api_key = (
            api_key if provider == "azure" else os.getenv("AZURE_OPENAI_API_KEY")
        )

        if provider == "azure":
            self.use_azure = True
        elif provider == "openai":
            self.use_azure = False
        else:
            self.use_azure = bool(azure_api_key) and not bool(openai_api_key)

        if endpoint_envvar is None:
            if openai_api_key and not self.use_azure:
                endpoint_envvar = "OPENAI_BASE_URL"
            elif suggested_endpoint_envvar is not None:
                endpoint_envvar = suggested_endpoint_envvar
            else:
                endpoint_envvar = DEFAULT_ENVVAR

        self.endpoint_envvar = endpoint_envvar
        self.azure_token_provider = None

        if self.model_name == TEST_MODEL_NAME:
            self.async_client = None
        elif self.use_azure:
            actual_api_key = api_key or azure_api_key
            if not actual_api_key:
                raise ValueError(
                    "Azure API key not provided and AZURE_OPENAI_API_KEY not found in environment."
                )
            with timelog("Using Azure OpenAI"):
                self._setup_azure(actual_api_key, api_base, api_version, max_retries)
        else:
            actual_api_key = api_key or openai_api_key
            if not actual_api_key:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY not found in environment."
                )
            endpoint = api_base or os.getenv(self.endpoint_envvar)
            with timelog("Using OpenAI"):
                self.async_client = AsyncOpenAI(
                    base_url=endpoint,
                    api_key=actual_api_key,
                    max_retries=max_retries,
                )

        if self.model_name in tiktoken_model.MODEL_TO_ENCODING:
            encoding_name = tiktoken.encoding_name_for_model(self.model_name)
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.max_chunk_size = MAX_TOKEN_SIZE
            self.max_size_per_batch = MAX_TOKENS_PER_BATCH
        else:
            self.encoding = None
            self.max_chunk_size = MAX_CHAR_SIZE
            self.max_size_per_batch = MAX_CHARS_PER_BATCH

        self._embedding_cache = {}

    def _setup_azure(
        self,
        azure_api_key: str,
        api_base: str | None = None,
        api_version: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        from .utils import get_azure_api_key, parse_azure_endpoint

        azure_api_key = get_azure_api_key(azure_api_key)

        if api_base:
            self.azure_endpoint = api_base.rstrip("/")
            self.azure_api_version = api_version or "2024-12-01-preview"
        else:
            self.azure_endpoint, self.azure_api_version = parse_azure_endpoint(
                self.endpoint_envvar
            )
            if api_version:
                self.azure_api_version = api_version

        if azure_api_key.lower() == "identity" or (
            os.getenv("AZURE_OPENAI_API_KEY", "").lower() == "identity"
            and azure_api_key != os.getenv("AZURE_OPENAI_API_KEY")
        ):
            self.azure_token_provider = get_shared_token_provider()

        self.async_client = AsyncAzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=azure_api_key,
            max_retries=max_retries,
        )

    async def refresh_auth(self) -> None:
        """Update client when using a token provider and it is nearly expired."""
        assert self.azure_token_provider
        refresh_token = self.azure_token_provider.refresh_token
        loop = asyncio.get_running_loop()
        azure_api_key = await loop.run_in_executor(None, refresh_token)
        assert self.azure_api_version
        assert self.azure_endpoint
        self.async_client = AsyncAzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=azure_api_key,
        )

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        existing = self._embedding_cache.get(key)
        if existing is not None:
            assert np.array_equal(existing, embedding)
        else:
            self._embedding_cache[key] = embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        embeddings = await self.get_embeddings_nocache([input])
        return embeddings[0]

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            empty = np.array([], dtype=np.float32)
            empty.shape = (0, self.embedding_size)
            return empty
        if self.azure_token_provider and self.azure_token_provider.needs_refresh():
            await self.refresh_auth()
        extra_args = {}
        if self.model_name != DEFAULT_MODEL_NAME:
            extra_args["dimensions"] = self.embedding_size
        if self.async_client is None:
            fake_data: list[NormalizedEmbedding] = []
            for item in input:
                if not item:
                    raise ValueError("Empty input text")
                floats = _compute_fake_embedding(item, self.embedding_size)
                array = np.array(floats, dtype=np.float64)
                normalized = array / np.sqrt(np.dot(array, array))
                dot = np.dot(normalized, normalized)
                assert (
                    abs(dot - 1.0) < 1e-15
                ), f"Embedding {normalized} is not normalized: {dot}"
                fake_data.append(normalized.astype(np.float32))
            assert len(fake_data) == len(input), (len(fake_data), "!=", len(input))
            return np.array(fake_data, dtype=np.float32)

        batches: list[list[str]] = []
        batch: list[str] = []
        batch_sum = 0
        for sentence in input:
            truncated_input, truncated_input_size = await self.truncate_input(sentence)
            if (
                len(batch) >= MAX_BATCH_SIZE
                or batch_sum + truncated_input_size > self.max_size_per_batch
            ):
                batches.append(batch)
                batch = []
                batch_sum = 0
            batch.append(truncated_input)
            batch_sum += truncated_input_size
        if batch:
            batches.append(batch)

        data: list[Embedding] = []
        assert self.async_client is not None
        for batch in batches:
            embeddings_data = (
                await self.async_client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    encoding_format="float",
                    **extra_args,
                )
            ).data
            data.extend(embeddings_data)

        assert len(data) == len(input), (len(data), "!=", len(input))
        return np.array([d.embedding for d in data], dtype=np.float32)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        """Retrieve an embedding, using the cache."""
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        embedding = await self.get_embedding_nocache(key)
        self._embedding_cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        """Retrieve embeddings for multiple keys, using the cache."""
        embeddings: list[NormalizedEmbedding | None] = []
        missing_keys: list[str] = []

        for key in keys:
            if key in self._embedding_cache:
                embeddings.append(self._embedding_cache[key])
            else:
                embeddings.append(None)
                missing_keys.append(key)

        if missing_keys:
            new_embeddings = await self.get_embeddings_nocache(missing_keys)
            for key, embedding in zip(missing_keys, new_embeddings):
                self._embedding_cache[key] = embedding

            for i, key in enumerate(keys):
                if embeddings[i] is None:
                    embeddings[i] = self._embedding_cache[key]
        return np.array(embeddings, dtype=np.float32).reshape(
            (len(keys), self.embedding_size)
        )

    async def truncate_input(self, input: str) -> tuple[str, int]:
        """Truncate input strings to fit within model limits."""
        if self.encoding is None:
            if len(input) > self.max_chunk_size:
                return input[: self.max_chunk_size], self.max_chunk_size
            return input, len(input)

        tokens = self.encoding.encode(input)
        if len(tokens) > self.max_chunk_size:
            truncated_tokens = tokens[: self.max_chunk_size]
            return self.encoding.decode(truncated_tokens), self.max_chunk_size
        return input, len(tokens)


def _compute_fake_embedding(input_text: str, embedding_size: int) -> list[float]:
    """Generate a deterministic fake embedding for testing."""

    def hashish(s: str) -> int:
        h = 0
        for ch in s:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    prime = 1961
    floats: list[float] = []
    length = len(input_text)
    for i in range(embedding_size):
        cut = i % length
        scrambled = input_text[cut:] + input_text[:cut]
        hashed = hashish(scrambled)
        reduced = (hashed % prime) / prime
        floats.append(reduced)
    return floats
