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
        print(
            f"  Added item {i + 1}: {result.messages_added} messages, {result.entities_extracted} refs"
        )

    # Search asynchronously
    print("\n--- Async Search ---")

    # Run multiple searches concurrently
    search_tasks = [
        memory.search("What programming language does the user like?"),
        memory.search("What is the user learning?"),
        memory.search("What framework is used?"),
    ]

    all_results = await asyncio.gather(*search_tasks)
    questions = [
        "What programming language?",
        "What is the user learning?",
        "What framework is used?",
    ]
    for q, results in zip(questions, all_results):
        print(f"\nQ: {q}")
        for item in results:
            print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # Additional search
    print("\n--- Search by topic ---")
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
