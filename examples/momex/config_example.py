"""Momex Configuration Example."""

import asyncio

from momex import Memory, MomexConfig
from momex.config import EmbeddingConfig, LLMConfig


async def main():
    # Configure once (use MOMEX_API_KEY env var for the key)
    MomexConfig.set_default(
        llm=LLMConfig(
            provider="openai",  # openai, azure, anthropic, deepseek, qwen
            model="gpt-4o",
        ),
        embedding=EmbeddingConfig(
            model="text-embedding-3-small",
        ),
    )

    # Create memory
    memory = Memory(collection="user:test")

    # Use it
    await memory.add("I like Python")
    results = await memory.search("What do I like?")
    for item in results:
        print(f"[{item.type}] {item.text}")


if __name__ == "__main__":
    asyncio.run(main())
