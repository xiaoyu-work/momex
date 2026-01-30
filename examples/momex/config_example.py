"""Momex Configuration Example."""

import asyncio
from momex import Memory, MomexConfig


async def main():
    # Configure once (use MOMEX_API_KEY env var for the key)
    MomexConfig.set_default(
        provider="openai",  # openai, azure, anthropic, deepseek, qwen
        model="gpt-4o",
        embedding_model="text-embedding-3-small",
    )

    # Create memory
    memory = Memory(collection="user:test")

    # Use it
    await memory.add("I like Python")
    answer = await memory.query("What do I like?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
