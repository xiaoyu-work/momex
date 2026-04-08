"""Momex Conversation Example.

This example demonstrates how to store memories from conversations.

Prerequisites:
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_MODEL=gpt-4o
"""

import asyncio

from momex import Memory, MomexConfig
from momex.config import StorageConfig


async def main():
    # Configure storage
    config = MomexConfig(storage=StorageConfig(path="./conversation_data"))

    # Create memory for a user
    memory = Memory(collection="user:xiaoyuzhang", config=config)
    await memory.clear()

    # Simulate a conversation
    conversation = [
        {
            "role": "user",
            "content": "Hi, my name is Xiaoyu and I'm a software engineer.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Xiaoyu! What kind of software do you work on?",
        },
        {
            "role": "user",
            "content": "I mainly work on Python backend services. I love using FastAPI.",
        },
        {
            "role": "assistant",
            "content": "FastAPI is great! Do you have any other interests?",
        },
        {
            "role": "user",
            "content": "Yes, I enjoy hiking on weekends and I'm learning Japanese.",
        },
    ]

    print("Processing conversation...")
    print("=" * 50)
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    print("=" * 50)

    # Store memories from conversation
    result = await memory.add(conversation)
    print(
        f"\nAdded {result.messages_added} messages, extracted {result.entities_extracted} semantic refs"
    )

    # Search for structured results
    print("\n--- Search results ---")
    results = await memory.search("programming")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # Search the memories
    print("\n--- Searching memories ---")

    results = await memory.search("What is the user's profession?")
    print(f"Q: What is the user's profession?")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    results = await memory.search("What are the user's hobbies?")
    print(f"\nQ: What are the user's hobbies?")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    results = await memory.search("What framework does the user use?")
    print(f"\nQ: What framework does the user use?")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # Simulate a follow-up conversation
    print("\n" + "=" * 50)
    print("Processing follow-up conversation...")
    print("=" * 50)

    followup_conversation = [
        {
            "role": "user",
            "content": "I've switched to using Django now instead of FastAPI.",
        },
        {"role": "assistant", "content": "Interesting! What made you switch?"},
        {
            "role": "user",
            "content": "The project needed more built-in features. Also, I passed my JLPT N3 exam!",
        },
    ]

    for msg in followup_conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}")

    result = await memory.add(followup_conversation)
    print(
        f"\nAdded {result.messages_added} messages, extracted {result.entities_extracted} semantic refs"
    )

    # Search updated memories
    print("\n--- Searching updated memories ---")

    results = await memory.search("What Python framework does the user use?")
    print(f"Q: What Python framework does the user use?")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    results = await memory.search("What is the user's Japanese level?")
    print(f"\nQ: What is the user's Japanese level?")
    for item in results:
        print(f"  [{item.type}] {item.text} (score={item.score:.2f})")

    # Show stats
    stats = await memory.stats()
    print(f"\n--- Stats ---")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total semantic refs: {stats['total_semantic_refs']}")


if __name__ == "__main__":
    asyncio.run(main())
