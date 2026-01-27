"""Memex prompts for fact extraction and memory management."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import FactType


def build_fact_extraction_prompt(fact_types: list[FactType]) -> str:
    """Build the fact extraction prompt with configurable fact types.

    Args:
        fact_types: List of FactType objects defining what to extract.

    Returns:
        The complete prompt string.
    """
    # Build the types section
    types_section = "Types of Information to Remember:\n\n"
    for i, ft in enumerate(fact_types, 1):
        types_section += f"{i}. {ft.name}: {ft.description}\n"

    return f"""You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

{types_section}
Examples:

Input: Hi.
Output: {{"facts": []}}

Input: There are branches in trees.
Output: {{"facts": []}}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {{"facts": ["Looking for a restaurant in San Francisco"]}}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {{"facts": ["Had a meeting with John at 3pm", "Discussed the new project"]}}

Input: Hi, my name is John. I am a software engineer.
Output: {{"facts": ["Name is John", "Is a software engineer"]}}

Input: My favourite movies are Inception and Interstellar.
Output: {{"facts": ["Favourite movies are Inception and Interstellar"]}}

Return the facts and preferences in JSON format as shown above.

Remember:
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- Do not return anything from the example prompts provided above.
- If you do not find anything relevant in the conversation, return an empty list for "facts".
- Create facts based on the user messages only. Do not extract facts from assistant or system messages.
- Make sure to return the response in JSON format with a key "facts" and a list of strings as value.
- Detect the language of the user input and record the facts in the same language.

Following is a conversation between the user and the assistant. Extract the relevant facts and preferences about the user from the conversation:
"""

MEMORY_UPDATE_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) ADD into the memory, (2) UPDATE the memory, (3) DELETE from the memory, and (4) NONE (no change).

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element (ONLY if it's truly NEW information)
- UPDATE: Update an existing memory element (when new fact is about the SAME TOPIC)
- DELETE: Delete an existing memory element (when new fact contradicts existing)
- NONE: Make no change (if the fact is already present or irrelevant)

Guidelines:

1. **ADD**: ONLY if the new fact is about a completely different topic not covered by any existing memory.
   - Example: Memory has "Likes Python" → New fact "Has a dog named Max" → ADD (different topic)

2. **UPDATE**: If the new fact is about the SAME TOPIC as an existing memory, even if worded differently.
   - Example: Memory has "Likes Python" → New fact "Really loves Python programming" → UPDATE (same topic: Python preference)
   - Example: Memory has "Works at Google" → New fact "Senior engineer at Google" → UPDATE (same topic: job)
   - The new text should be the more complete or more recent version.
   - Use the existing memory's ID when updating.

3. **DELETE**: If the new fact CONTRADICTS existing memory.
   - Example: Memory has "Loves cheese pizza" → New fact "Dislikes cheese pizza" → DELETE

4. **NONE**: If the new fact expresses exactly the same meaning as existing memory.
   - Example: Memory has "Likes Python" → New fact "Likes Python" → NONE

IMPORTANT: Prefer UPDATE over ADD when facts are semantically related. Two facts about the same topic (e.g., both about Python preference, both about job, both about location) should result in UPDATE, not ADD.

You must return your response in the following JSON structure only:

{
    "memory": [
        {
            "id": "<ID of the memory to update/delete, or new ID for ADD>",
            "text": "<Content of the memory>",
            "event": "<Operation: ADD, UPDATE, DELETE, or NONE>",
            "old_memory": "<Old memory content, only if event is UPDATE>"
        }
    ]
}

Instructions:
- If the current memory is empty, add all new facts with event "ADD".
- For additions, generate a new numeric ID (starting from the highest existing ID + 1).
- For updates and deletes, use the existing memory ID.
- Return only valid JSON format.
"""


def get_fact_extraction_prompt(
    conversation: str,
    fact_types: list[FactType] | None = None,
) -> str:
    """Get the fact extraction prompt with conversation appended.

    Args:
        conversation: The conversation text to extract facts from.
        fact_types: Custom fact types. If None, uses default types.

    Returns:
        The complete prompt with conversation.
    """
    if fact_types is None:
        from .config import DEFAULT_FACT_TYPES
        fact_types = DEFAULT_FACT_TYPES

    prompt = build_fact_extraction_prompt(fact_types)
    return prompt + "\n\n" + conversation


def get_memory_update_prompt(
    existing_memories: list[dict],
    new_facts: list[str],
) -> str:
    """Get the memory update prompt with context."""
    if existing_memories:
        memory_part = f"""
Current memory:
```
{existing_memories}
```
"""
    else:
        memory_part = """
Current memory is empty.
"""

    return f"""{MEMORY_UPDATE_PROMPT}

{memory_part}

The new retrieved facts are:
```
{new_facts}
```

Analyze the new facts and determine whether each should be ADD, UPDATE, DELETE, or NONE.
Return only valid JSON.
"""
