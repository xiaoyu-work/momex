"""Memex Conversation Example.

This example demonstrates how to extract memories from conversations
using the add_conversation() method, similar to mem0's approach.

The method uses a multi-stage process:
1. Extract facts from the conversation (using configurable fact types)
2. For each fact, vector search to find similar existing memories
3. LLM decides ADD/UPDATE/DELETE/NONE based on similar memories
4. Execute operations
"""

from pathlib import Path

from memex import FactType, Memory, MemexConfig


def main():
    # Option 1: Use default config
    config = MemexConfig(
        storage_path="./conversation_data",
    )

    # Option 2: Load from YAML file (uncomment to use)
    # config = MemexConfig.from_yaml("memex_config.yaml")

    # Option 3: Custom fact types in code
    # config = MemexConfig(
    #     storage_path="./conversation_data",
    #     fact_types=[
    #         FactType(name="Technical Skills", description="Programming languages and frameworks"),
    #         FactType(name="Project Info", description="Current projects and deadlines"),
    #     ],
    #     similarity_threshold=0.6,
    # )

    # Create memory for a user
    memory = Memory(collection="user:alice", config=config)

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "Hi, my name is Alice and I'm a software engineer."},
        {"role": "assistant", "content": "Nice to meet you, Alice! What kind of software do you work on?"},
        {"role": "user", "content": "I mainly work on Python backend services. I love using FastAPI."},
        {"role": "assistant", "content": "FastAPI is great! Do you have any other interests?"},
        {"role": "user", "content": "Yes, I enjoy hiking on weekends and I'm learning Japanese."},
    ]

    print("Processing conversation...")
    print("=" * 50)
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    print("=" * 50)

    # Extract facts from conversation and add to memory
    result = memory.add_conversation(conversation)

    print(f"\nFacts extracted: {len(result.facts_extracted)}")
    for fact in result.facts_extracted:
        print(f"  - {fact}")

    print(f"\nOperations performed:")
    print(f"  Added: {result.memories_added}")
    print(f"  Updated: {result.memories_updated}")
    print(f"  Deleted: {result.memories_deleted}")

    if result.operations:
        print(f"\nOperation details:")
        for op in result.operations:
            print(f"  [{op.event}] {op.text}")

    # Query the memories
    print("\n--- Querying memories ---")
    answer = memory.query("What is Alice's profession?")
    print(f"Q: What is Alice's profession?\nA: {answer}")

    answer = memory.query("What are Alice's hobbies?")
    print(f"\nQ: What are Alice's hobbies?\nA: {answer}")

    # Simulate another conversation with updates
    print("\n" + "=" * 50)
    print("Processing follow-up conversation...")
    print("=" * 50)

    followup_conversation = [
        {"role": "user", "content": "I've switched to using Django now instead of FastAPI."},
        {"role": "assistant", "content": "Interesting! What made you switch?"},
        {"role": "user", "content": "The project needed more built-in features. Also, I passed my JLPT N3 exam!"},
    ]

    for msg in followup_conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}")

    result2 = memory.add_conversation(followup_conversation)

    print(f"\nNew facts extracted: {len(result2.facts_extracted)}")
    for fact in result2.facts_extracted:
        print(f"  - {fact}")

    print(f"\nOperations:")
    for op in result2.operations:
        print(f"  [{op.event}] {op.text}")
        if op.old_memory:
            print(f"      (was: {op.old_memory})")

    # Query updated memories
    print("\n--- Querying updated memories ---")
    answer = memory.query("What Python framework does Alice use?")
    print(f"Q: What Python framework does Alice use?\nA: {answer}")

    answer = memory.query("What is Alice's Japanese level?")
    print(f"\nQ: What is Alice's Japanese level?\nA: {answer}")


if __name__ == "__main__":
    main()
