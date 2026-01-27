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

from memex import Memory


def main():
    # Create memory for a user
    print("Creating memory for user_123...")
    memory = Memory(user_id="user_123")

    # Add some memories
    print("\nAdding memories...")
    memory.add("今天和张三开会讨论了项目进度", speaker="我")
    memory.add("张三说下周五之前能完成API开发", speaker="会议记录")
    memory.add("李四负责前端开发，王五负责后端开发", speaker="会议记录")
    memory.add("项目截止日期是下个月15号", speaker="经理")

    # Get statistics
    stats = memory.stats()
    print(f"\nMemory stats: {stats}")

    # Query with natural language
    print("\n--- Querying memories ---")

    questions = [
        "谁负责API开发?",
        "项目截止日期是什么时候?",
        "李四负责什么?",
        "下周五有什么安排?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = memory.query(q)
        print(f"A: {answer}")

    # Search by keyword
    print("\n--- Searching for '张三' ---")
    results = memory.search("张三")
    for item in results:
        print(f"  - {item.text}")

    # Export memories
    export_path = "./memex_export.json"
    print(f"\nExporting to {export_path}...")
    memory.export(export_path)
    print("Done!")


if __name__ == "__main__":
    main()
