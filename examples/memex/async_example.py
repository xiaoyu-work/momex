"""Memex Async API Example.

This example demonstrates using the async API for better performance
when integrating with async frameworks like FastAPI, aiohttp, etc.
"""

import asyncio

from memex import Memory


async def main():
    # Create memory for a collection
    memory = Memory(collection="user:async_demo")

    # Add memories asynchronously
    print("Adding memories asynchronously...")

    # Add multiple items concurrently
    tasks = [
        memory.add_async("Python是一种编程语言", speaker="教程"),
        memory.add_async("机器学习需要大量数据", speaker="课程"),
        memory.add_async("深度学习是机器学习的子集", speaker="课程"),
    ]

    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"  Added item {i + 1}: {result.messages_added} messages, {result.entities_extracted} entities")

    # Or use batch add for better performance
    print("\nAdding batch memories...")
    batch_result = await memory.add_batch_async(
        [
            {"text": "TensorFlow是Google的深度学习框架", "speaker": "文档"},
            {"text": "PyTorch是Facebook的深度学习框架", "speaker": "文档"},
            {"text": "Keras是高级神经网络API", "speaker": "文档"},
        ]
    )
    print(f"  Batch added: {batch_result.messages_added} messages")

    # Query asynchronously
    print("\n--- Async Queries ---")

    questions = [
        "什么是深度学习?",
        "有哪些深度学习框架?",
        "TensorFlow是谁开发的?",
    ]

    # Query concurrently
    query_tasks = [memory.query_async(q) for q in questions]
    answers = await asyncio.gather(*query_tasks)

    for q, a in zip(questions, answers):
        print(f"\nQ: {q}")
        print(f"A: {a}")

    # Get stats
    stats = await memory.stats_async()
    print(f"\n--- Stats ---\n{stats}")


if __name__ == "__main__":
    asyncio.run(main())
