"""Momex Async API Example.

This example demonstrates using the async API for better performance
when integrating with async frameworks like FastAPI, aiohttp, etc.
"""

import asyncio

from momex import Memory, query_async


async def main():
    # Create memory with hierarchical identity
    memory = Memory(collection="company:engineering:alice")

    # Add memories asynchronously
    print("Adding memories asynchronously...")

    # Add multiple items concurrently
    tasks = [
        memory.add_async("Python is my favorite language", speaker="Alice"),
        memory.add_async("Currently learning machine learning", speaker="Alice"),
        memory.add_async("The project uses FastAPI framework", speaker="Alice"),
    ]

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"  Added item {i + 1}: {result.messages_added} messages")

    # Or use batch add for better performance
    print("\nAdding batch memories...")
    batch_result = await memory.add_batch_async(
        [
            {"text": "TensorFlow is used for deep learning", "speaker": "Alice"},
            {"text": "PyTorch is more flexible", "speaker": "Alice"},
        ]
    )
    print(f"  Batch added: {batch_result.messages_added} messages")

    # Query asynchronously
    print("\n--- Async Queries ---")

    # Query single collection
    answer = await memory.query_async("What programming language does Alice like?")
    print(f"\nAlice's collection: {answer}")

    # Query with prefix (async version)
    answer = await query_async("company:engineering", "What deep learning frameworks are there?")
    print(f"\nEngineering team: {answer}")

    # Get stats
    stats = await memory.stats_async()
    print(f"\n--- Stats ---\n{stats}")


if __name__ == "__main__":
    asyncio.run(main())
