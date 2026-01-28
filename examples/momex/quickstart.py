"""Momex Quick Start Example.

This example demonstrates the basic usage of Momex for structured memory.

Prerequisites:
    Set LLM via environment variables:
        export OPENAI_API_KEY=sk-xxx
        export OPENAI_MODEL=gpt-4o
    Or for Azure:
        export AZURE_OPENAI_API_KEY=xxx
        export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com

Run:
    python examples/momex/quickstart.py
"""

import asyncio
from momex import Memory


async def main():
    # Create memory for a user
    print("Creating memory for user:alice...")
    memory = Memory(collection="user:alice")

    # ===========================================
    # Method 1: Add memories manually
    # ===========================================
    print("\n--- Adding memories manually ---")
    await memory.add("I like Python programming")
    await memory.add("Project deadline is Friday")
    print("Added 2 memories")

    # ===========================================
    # Method 2: Add from conversation
    # ===========================================
    print("\n--- Adding from conversation ---")
    conversation = [
        {"role": "user", "content": "I'm working on a FastAPI backend project."},
        {"role": "assistant", "content": "That sounds interesting!"},
        {"role": "user", "content": "Yes, and I enjoy hiking on weekends."},
    ]

    result = await memory.add_conversation(conversation)
    if result.success:
        print("Conversation memories stored!")
    else:
        print(f"Error: {result.error}")

    # ===========================================
    # Query memories
    # ===========================================
    print("\n--- Querying memories ---")

    answer = await memory.query("What programming language does the user like?")
    print(f"Q: What programming language?\nA: {answer}")

    answer = await memory.query("What project is the user working on?")
    print(f"\nQ: What project?\nA: {answer}")

    answer = await memory.query("What does the user do on weekends?")
    print(f"\nQ: Weekend activity?\nA: {answer}")

    # ===========================================
    # Stats
    # ===========================================
    print("\n--- Stats ---")
    stats = await memory.stats()
    print(f"Total memories: {stats['total_memories']}")


if __name__ == "__main__":
    asyncio.run(main())
