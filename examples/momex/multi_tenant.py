"""Momex Hierarchical Query Example.

This example demonstrates how to use hierarchical collections
and prefix queries for personal and team assistants.
"""

from momex import Memory, MemoryManager, MomexConfig, query


def main():
    # Configure storage location
    config = MomexConfig(
        storage_path="./hierarchical_data",
    )

    print("Creating memories for multiple people...\n")

    # Engineering team
    xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang", config=config)
    print(f"Xiaoyuzhang's DB: {xiaoyuzhang.db_path}")
    xiaoyuzhang.add("I like Python programming", speaker="Xiaoyuzhang")
    xiaoyuzhang.add("Working on user authentication module", speaker="Xiaoyuzhang")

    gvanrossum = Memory(collection="momex:engineering:gvanrossum", config=config)
    print(f"Gvanrossum's DB: {gvanrossum.db_path}")
    gvanrossum.add("I am a Java developer", speaker="Gvanrossum")
    gvanrossum.add("Responsible for backend API development", speaker="Gvanrossum")

    # Marketing team
    charlie = Memory(collection="momex:marketing:charlie", config=config)
    print(f"Charlie's DB: {charlie.db_path}")
    charlie.add("I handle product marketing", speaker="Charlie")

    # Query with different scopes
    print("\n--- Query single person (Xiaoyuzhang) ---")
    answer = query("momex:engineering:xiaoyuzhang", "What is Xiaoyuzhang working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query engineering team ---")
    answer = query("momex:engineering", "What are the team members working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query entire momex ---")
    answer = query("momex", "Who works at momex?", config=config)
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
    main()
