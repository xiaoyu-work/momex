"""Momex Conversation Example.

This example demonstrates how to store memories from conversations
using the add_conversation() method.

Prerequisites:
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=gpt-4o
"""

import asyncio
from momex import Memory, MomexConfig


async def main():
    # Configure storage
    config = MomexConfig(storage_path="./conversation_data")

    # Create memory for a user
    memory = Memory(collection="user:xiaoyuzhang", config=config)

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

    # Store memories from conversation
    result = await memory.add_conversation(conversation)

    if result.success:
        print("\nMemories stored successfully!")
    else:
        print(f"\nError: {result.error}")

    # Query the memories
    print("\n--- Querying memories ---")

    answer = await memory.query("What is Alice's profession?")
    print(f"Q: What is Alice's profession?\nA: {answer}")

    answer = await memory.query("What are Alice's hobbies?")
    print(f"\nQ: What are Alice's hobbies?\nA: {answer}")

    answer = await memory.query("What framework does Alice use?")
    print(f"\nQ: What framework does Alice use?\nA: {answer}")

    # Simulate a follow-up conversation with updates
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

    result2 = await memory.add_conversation(followup_conversation)

    if result2.success:
        print("\nMemories updated successfully!")

    # Query updated memories
    print("\n--- Querying updated memories ---")

    answer = await memory.query("What Python framework does Alice use?")
    print(f"Q: What Python framework does Alice use?\nA: {answer}")

    answer = await memory.query("What is Alice's Japanese level?")
    print(f"\nQ: What is Alice's Japanese level?\nA: {answer}")

    # Show stats
    stats = await memory.stats()
    print(f"\n--- Stats ---")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Deleted memories: {stats['deleted_memories']}")


if __name__ == "__main__":
    asyncio.run(main())
