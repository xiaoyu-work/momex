"""Memex Quick Start Example.

This example demonstrates the basic usage of Memex for structured memory.

Before running:
1. Set your OpenAI API key:
   export OPENAI_API_KEY=your-api-key
   export OPENAI_MODEL=gpt-4o

2. Or create a .env file with:
   OPENAI_API_KEY=your-api-key
   OPENAI_MODEL=gpt-4o

Run:
    python examples/memex/quickstart.py
"""

from memex import Memory, query


def main():
    # Create memory with hierarchical identity
    print("Creating memory for company:engineering:alice...")
    memory = Memory(collection="company:engineering:alice")

    # Add some memories
    print("\nAdding memories...")
    memory.add("I like Python programming", speaker="Alice")
    memory.add("Need to finish API development by Friday", speaker="Alice")
    memory.add("Project deadline is the 15th of next month", speaker="Manager")

    # Get statistics
    stats = memory.stats()
    print(f"\nMemory stats: {stats}")

    # Query single collection
    print("\n--- Querying Alice's memories only ---")
    answer = memory.query("What programming language does Alice like?")
    print(f"Answer: {answer}")

    # Query with prefix (would search all engineering if others existed)
    print("\n--- Querying with prefix ---")
    answer = query("company:engineering:alice", "When is the project deadline?")
    print(f"Answer: {answer}")

    # Search by keyword
    print("\n--- Searching for 'API' ---")
    results = memory.search("API")
    for item in results:
        print(f"  - {item.text}")

    # Export memories
    export_path = "./memex_export.json"
    print(f"\nExporting to {export_path}...")
    memory.export(export_path)
    print("Done!")


if __name__ == "__main__":
    main()
