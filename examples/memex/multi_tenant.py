"""Memex Hierarchical Query Example.

This example demonstrates how to use hierarchical collections
and prefix queries for personal and team assistants.
"""

from memex import Memory, MemoryManager, MemexConfig, query


def main():
    # Configure storage location
    config = MemexConfig(
        storage_path="./hierarchical_data",
    )

    print("Creating memories for multiple people...\n")

    # Engineering team
    alice = Memory(collection="company:engineering:alice", config=config)
    print(f"Alice's DB: {alice.db_path}")
    alice.add("I like Python programming", speaker="Alice")
    alice.add("Working on user authentication module", speaker="Alice")

    bob = Memory(collection="company:engineering:bob", config=config)
    print(f"Bob's DB: {bob.db_path}")
    bob.add("I am a Java developer", speaker="Bob")
    bob.add("Responsible for backend API development", speaker="Bob")

    # Marketing team
    charlie = Memory(collection="company:marketing:charlie", config=config)
    print(f"Charlie's DB: {charlie.db_path}")
    charlie.add("I handle product marketing", speaker="Charlie")

    # Query with different scopes
    print("\n--- Query single person (Alice) ---")
    answer = query("company:engineering:alice", "What is Alice working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query engineering team ---")
    answer = query("company:engineering", "What are the team members working on?", config=config)
    print(f"Answer: {answer}")

    print("\n--- Query entire company ---")
    answer = query("company", "Who works at the company?", config=config)
    print(f"Answer: {answer}")

    # Use MemoryManager to list collections
    print("\n--- Using MemoryManager ---")
    manager = MemoryManager(config=config)

    print("\nAll collections:")
    for coll in manager.list_collections():
        print(f"  - {coll}")

    print("\nEngineering team only:")
    for coll in manager.list_collections(prefix="company:engineering"):
        print(f"  - {coll}")


if __name__ == "__main__":
    main()
