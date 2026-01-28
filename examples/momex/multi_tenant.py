"""Momex Hierarchical Query Example.

This example demonstrates how to use hierarchical collections
and prefix queries for personal and team assistants.

Prerequisites:
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=gpt-4o
"""

import asyncio
from momex import Memory, MemoryManager, MomexConfig, query


async def main():
    # Configure storage location
    config = MomexConfig(storage_path="./hierarchical_data")

    print("Creating memories for multiple people...\n")

    # Engineering team
    xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang", config=config)
    await xiaoyuzhang.clear()
    print(f"Xiaoyuzhang's DB: {xiaoyuzhang.db_path}")
    await xiaoyuzhang.add("I like Python programming")
    await xiaoyuzhang.add("Working on user authentication module")

    gvanrossum = Memory(collection="momex:engineering:gvanrossum", config=config)
    await gvanrossum.clear()
    print(f"Gvanrossum's DB: {gvanrossum.db_path}")
    await gvanrossum.add("I am a Java developer")
    await gvanrossum.add("Responsible for backend API development")

    # Marketing team
    charlie = Memory(collection="momex:marketing:charlie", config=config)
    await charlie.clear()
    print(f"Charlie's DB: {charlie.db_path}")
    await charlie.add("I handle product marketing")

    # Query with different scopes
    print("\n--- Query single person (Xiaoyuzhang) ---")
    answer = await query("momex:engineering:xiaoyuzhang", "What is Xiaoyuzhang working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query engineering team ---")
    answer = await query("momex:engineering", "What are the team members working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query entire momex ---")
    answer = await query("momex", "Who works at momex?", config=config)
    print(f"Answer: {answer}")

    # Use MemoryManager to list collections
    print("\n--- Using MemoryManager ---")
    manager = MemoryManager(config=config)

    print("\nAll collections:")
    for coll in manager.list_collections():
        print(f"  - {coll}")

    print("\nEngineering team only:")
    for coll in manager.list_collections(prefix="momex:engineering"):
        print(f"  - {coll}")


if __name__ == "__main__":
    asyncio.run(main())
