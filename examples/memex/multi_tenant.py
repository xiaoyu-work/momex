"""Memex Collection-Based Isolation Example.

This example demonstrates how to use Memex with collection-based isolation
and MemoryPool for querying across multiple collections.
"""

from memex import Memory, MemoryPool, MemoryManager, MemexConfig


def main():
    # Configure storage location
    config = MemexConfig(
        storage_path="./multi_collection_data",
    )

    # Create memories for different collections
    print("Creating memories for multiple collections...\n")

    # Personal collection for Alice
    alice_memory = Memory(collection="user:alice", config=config)
    print(f"Alice's DB: {alice_memory.db_path}")
    alice_memory.add("我喜欢Python编程", speaker="Alice")
    alice_memory.add("下周要去北京出差", speaker="Alice")

    # Personal collection for Bob
    bob_memory = Memory(collection="user:bob", config=config)
    print(f"Bob's DB: {bob_memory.db_path}")
    bob_memory.add("我是Java开发者", speaker="Bob")
    bob_memory.add("正在学习机器学习", speaker="Bob")

    # Team collection
    team_memory = Memory(collection="team:engineering", config=config)
    print(f"Team's DB: {team_memory.db_path}")
    team_memory.add("团队使用PostgreSQL数据库", speaker="架构师")
    team_memory.add("代码审查需要两人以上批准", speaker="规范")

    # Query individual collections - they are isolated
    print("\n--- Querying isolated memories ---\n")

    print("Alice's query: 'What programming language do I like?'")
    print(f"Answer: {alice_memory.query('What programming language do I like?')}\n")

    print("Bob's query: 'What am I learning?'")
    print(f"Answer: {bob_memory.query('What am I learning?')}\n")

    # Use MemoryPool to query across collections
    print("--- Using MemoryPool for cross-collection queries ---\n")

    pool = MemoryPool(
        collections=["user:alice", "team:engineering"],
        default_collection="user:alice",
        config=config,
    )

    print("Pool query (alice + team): 'What database does the team use?'")
    answer = pool.query("What database does the team use?")
    print(f"Answer: {answer}\n")

    # Add to multiple collections at once
    pool.add("项目会议定在周三下午3点", collections=["user:alice", "team:engineering"])
    print("Added meeting note to both alice and team collections")

    # Use MemoryManager to list collections
    print("\n--- Using MemoryManager ---")
    manager = MemoryManager(config=config)
    collections = manager.list_collections()
    print(f"All collections: {collections}")

    for coll in collections:
        info = manager.info(coll)
        print(f"  {coll}: {info['size']}")


if __name__ == "__main__":
    main()
