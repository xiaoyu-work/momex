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
    print("Creating memory for user:xiaoyuzhang...")
    memory = Memory(collection="user:xiaoyuzhang")

    # Clear previous data
    await memory.clear()

    # ===========================================
    # Add memories
    # ===========================================
    print("\n--- Adding memories ---")
    result = await memory.add("I like Python programming")
    print(f"Added: {result.messages_added} messages, {result.entities_extracted} semantic refs")

    result = await memory.add("Project deadline is Friday")
    print(f"Added: {result.messages_added} messages, {result.entities_extracted} semantic refs")

    # ===========================================
    # Add from conversation
    # ===========================================
    print("\n--- Adding from conversation ---")
    conversation = [
        {"role": "user", "content": "I'm working on a FastAPI backend project."},
        {"role": "assistant", "content": "That sounds interesting!"},
        {"role": "user", "content": "Yes, and I enjoy hiking on weekends."},
    ]

    result = await memory.add(conversation)
    print(f"Added: {result.messages_added} messages, {result.entities_extracted} semantic refs")

    # ===========================================
    # Search - returns structured results
    # ===========================================
    print("\n--- Searching memories ---")
    results = await memory.search("programming")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # ===========================================
    # Query - returns LLM answer
    # ===========================================
    print("\n--- Querying memories ---")

    answer = await memory.query("What programming language does the user like?")
    print(f"Q: What programming language?\nA: {answer}")

    answer = await memory.query("What does the user do on weekends?")
    print(f"\nQ: Weekend activity?\nA: {answer}")

    # ===========================================
    # Stats
    # ===========================================
    print("\n--- Stats ---")
    stats = await memory.stats()
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total semantic refs: {stats['total_semantic_refs']}")


if __name__ == "__main__":
    asyncio.run(main())
