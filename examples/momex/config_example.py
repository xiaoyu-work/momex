"""Momex Configuration Example."""

import asyncio
from momex import Memory, MomexConfig


async def main():
    # Configure (code, YAML, or env vars)
    config = MomexConfig(
        provider="openai",  # openai, azure, anthropic, deepseek, qwen
        model="gpt-4o",
        api_key="sk-xxx",
    )

    # Or load from YAML
    # config = MomexConfig.from_yaml("config.yaml")

    # Create memory
    memory = Memory(collection="user:test", config=config)

    # Use it
    await memory.add("I like Python")
    answer = await memory.query("What do I like?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
