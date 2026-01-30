"""Momex Async API Example.

This example demonstrates using the async API for better performance
when integrating with async frameworks like FastAPI, aiohttp, etc.

All Momex methods are async by default.

Prerequisites:
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=gpt-4o
"""

import asyncio
from momex import Memory


async def main():
    # Create memory with hierarchical identity
    memory = Memory(collection="momex:engineering:xiaoyuzhang")
    await memory.clear()

    # Add memories asynchronously
    print("Adding memories asynchronously...")

    # Add multiple items concurrently
    tasks = [
        memory.add("Python is my favorite language"),
        memory.add("Currently learning machine learning"),
        memory.add("The project uses FastAPI framework"),
    ]

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"  Added item {i + 1}: {result.messages_added} messages, {result.entities_extracted} refs")

    # Query asynchronously
    print("\n--- Async Queries ---")

    # Run multiple queries concurrently
    query_tasks = [
        memory.query("What programming language does the user like?"),
        memory.query("What is the user learning?"),
        memory.query("What framework is used?"),
    ]

    answers = await asyncio.gather(*query_tasks)
    questions = [
        "What programming language?",
        "What is the user learning?",
        "What framework is used?",
    ]
    for q, a in zip(questions, answers):
        print(f"\nQ: {q}\nA: {a}")

    # Search asynchronously
    print("\n--- Async Search ---")
    results = await memory.search("programming")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # Get stats
    stats = await memory.stats()
    print(f"\n--- Stats ---")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total semantic refs: {stats['total_semantic_refs']}")


if __name__ == "__main__":
    asyncio.run(main())
