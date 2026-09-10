"""Token-bounded, source-cited context for any agent framework."""

from collections.abc import Sequence

import tiktoken

from .results import ContextCitation, ContextResult, SearchItem, SourceReference
from .visibility import MemoryStatus


def format_context(
    items: Sequence[SearchItem],
    *,
    token_budget: int = 2048,
    encoding_name: str = "cl100k_base",
) -> ContextResult:
    """Keep retrieval order, cite evidence and merge overlapping source turns.

    The budget includes labels and metadata, not just source text. No model is
    called. Historical/unconfirmed inputs remain explicitly labeled rather than
    being silently promoted to current facts.
    """
    if token_budget < 0:
        raise ValueError("token_budget cannot be negative")
    encoding = tiktoken.get_encoding(encoding_name)

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    blocks: list[str] = []
    citations: list[ContextCitation] = []
    seen_sources: set[tuple[str | None, str | int, str]] = set()
    seen_knowledge: set[tuple[str | None, str, str, str]] = set()
    truncated = False

    for item in items:
        candidates: list[tuple[str, tuple[SourceReference, ...], MemoryStatus]]
        if item.type == "message" and item.sources:
            candidates = [
                (source.text, (source,), source.status) for source in item.sources
            ]
        else:
            key = item.collection, item.type, item.status, item.text
            if key in seen_knowledge:
                continue
            seen_knowledge.add(key)
            candidates = [(item.text, item.sources, item.status)]

        for body, sources, status in candidates:
            if item.type == "message" and sources:
                source_key = sources[0].key
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
            if not body:
                continue
            label = f"m{len(citations) + 1}"
            origin = (
                sources[0]
                if sources
                else SourceReference(
                    text=body,
                    collection=item.collection,
                    timestamp=item.timestamp,
                    speaker=item.speaker,
                    status=status,
                )
            )
            metadata = " | ".join(
                value
                for value in (origin.collection, origin.timestamp, origin.speaker)
                if value
            )
            header = (
                f"[{label}] [{status}]" + (f" {metadata}" if metadata else "") + "\n"
            )
            prefix = "\n\n".join(blocks)
            if prefix:
                prefix += "\n\n"
            block = header + body
            if count(prefix + block) > token_budget:
                truncated = True
                # Character boundaries cannot split a multibyte token's text.
                low, high = 0, len(body)
                while low < high:
                    middle = (low + high + 1) // 2
                    if count(prefix + header + body[:middle] + "...") <= token_budget:
                        low = middle
                    else:
                        high = middle - 1
                if low == 0:
                    continue
                block = header + body[:low] + "..."
            blocks.append(block)
            citations.append(ContextCitation(label=label, sources=sources))

    text = "\n\n".join(blocks)
    return ContextResult(
        text=text,
        token_count=count(text),
        citations=citations,
        truncated=truncated,
        encoding_name=encoding_name,
    )
