"""Memex Multi-Tenant Example.

This example demonstrates how to use Memex with multi-tenant isolation.

Each user/organization gets their own isolated memory storage.
"""

from memex import Memory, MemexConfig


def main():
    # Configure storage location
    config = MemexConfig(
        storage_path="./multi_tenant_data",
    )

    # Create memories for different users in different organizations
    print("Creating memories for multiple tenants...\n")

    # Organization: Acme Corp
    # User: Alice
    alice_memory = Memory(
        user_id="alice",
        org_id="acme",
        config=config,
    )
    print(f"Alice's DB: {alice_memory.db_path}")
    alice_memory.add("我喜欢Python编程", speaker="Alice")
    alice_memory.add("下周要去北京出差", speaker="Alice")

    # Organization: Acme Corp
    # User: Bob
    bob_memory = Memory(
        user_id="bob",
        org_id="acme",
        config=config,
    )
    print(f"Bob's DB: {bob_memory.db_path}")
    bob_memory.add("我是Java开发者", speaker="Bob")
    bob_memory.add("正在学习机器学习", speaker="Bob")

    # Organization: Globex
    # User: Charlie
    charlie_memory = Memory(
        user_id="charlie",
        org_id="globex",
        config=config,
    )
    print(f"Charlie's DB: {charlie_memory.db_path}")
    charlie_memory.add("我们公司使用TypeScript", speaker="Charlie")

    # Query each user's memories - they are isolated
    print("\n--- Querying isolated memories ---\n")

    print("Alice's query: 'What programming language do I like?'")
    print(f"Answer: {alice_memory.query('What programming language do I like?')}\n")

    print("Bob's query: 'What am I learning?'")
    print(f"Answer: {bob_memory.query('What am I learning?')}\n")

    print("Charlie's query: 'What language does my company use?'")
    print(f"Answer: {charlie_memory.query('What language does my company use?')}\n")

    # Show stats for each
    print("--- Memory Stats ---")
    print(f"Alice: {alice_memory.stats()}")
    print(f"Bob: {bob_memory.stats()}")
    print(f"Charlie: {charlie_memory.stats()}")


if __name__ == "__main__":
    main()
