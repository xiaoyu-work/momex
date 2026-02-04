"""Momex Agent - High-level conversational API with automatic memory management.

Provides a simple chat interface that automatically:
1. Classifies user input (LLM decides short-term vs long-term)
2. Retrieves relevant memories as context
3. Generates responses
4. Manages conversation history

Example:
    >>> from momex import Agent, MomexConfig
    >>>
    >>> async def main():
    ...     config = MomexConfig(provider="openai", model="gpt-4o")
    ...     agent = Agent("user:xiaoyuzhang", config)
    ...
    ...     # Just chat - memory is handled automatically
    ...     response = await agent.chat("My name is Alice, I'm a Python developer")
    ...     print(response.content)
    ...
    ...     response = await agent.chat("What's my name?")
    ...     print(response.content)  # "Your name is Alice"
    ...
    ...     await agent.close()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import MomexConfig
from .memory import Memory
from .short_term import ShortTermMemory, Message


@dataclass
class ChatResponse:
    """Response from Agent.chat()."""

    content: str


class Agent:
    """High-level conversational agent with automatic memory management.

    Combines ShortTermMemory (conversation history) and Memory (long-term knowledge)
    to provide a simple chat interface. The agent automatically:

    1. Records all messages to short-term memory
    2. Uses LLM to decide if user input should be stored long-term
    3. Retrieves relevant context from both memory types
    4. Generates contextual responses

    Example:
        >>> agent = Agent("user:alice", config)
        >>>
        >>> # Automatic memory management - LLM decides what to remember
        >>> r = await agent.chat("I'm a software engineer at Google")
        >>> print(r.content)  # Identity info stored to long-term memory
        >>>
        >>> r = await agent.chat("What's the weather today?")
        >>> print(r.content)  # Temporary query, not stored long-term
        >>>
        >>> r = await agent.chat("What do I do for work?")
        >>> print(r.content)  # "You're a software engineer at Google"
        >>>
        >>> # Session management
        >>> sessions = agent.list_sessions()
        >>> agent.load_session(sessions[0].session_id)
        >>>
        >>> await agent.close()
    """

    def __init__(
        self,
        collection: str,
        config: MomexConfig | None = None,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_short_term: int = 100,
        max_context_messages: int = 10,
        max_retrieved_memories: int = 5,
    ):
        """Initialize the Agent.

        Args:
            collection: Collection name for memory storage (e.g., "user:xiaoyuzhang").
            config: Momex configuration. Uses default if None.
            session_id: Resume an existing session. Creates new session if None.
            system_prompt: Custom system prompt. Uses default if None.
            max_short_term: Maximum messages in short-term memory.
            max_context_messages: Max recent messages to include in context.
            max_retrieved_memories: Max long-term memories to retrieve.
        """
        self.collection = collection
        self.config = config or MomexConfig.get_default()

        # Long-term memory (structured RAG)
        self._long_term = Memory(collection, self.config)

        # Short-term memory (conversation history)
        self._short_term = ShortTermMemory(
            collection,
            self.config,
            session_id=session_id,
            max_messages=max_short_term,
        )

        # Configuration
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_messages = max_context_messages
        self.max_retrieved_memories = max_retrieved_memories

        # LLM client (lazy initialized)
        self._llm = None

    def _default_system_prompt(self) -> str:
        """Default system prompt for the agent."""
        return (
            "You are a helpful AI assistant with memory capabilities. "
            "You can remember information about the user from previous conversations. "
            "Use the provided context to give personalized and relevant responses. "
            "Be concise and helpful."
        )

    def _get_llm(self):
        """Get or create LLM client."""
        if self._llm is None:
            self._llm = self.config.create_llm()
        return self._llm

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._short_term.session_id

    # =========================================================================
    # Core Chat API
    # =========================================================================

    async def chat(self, user_input: str) -> ChatResponse:
        """Send a message and get a response.

        This is the main API. It automatically:
        1. Records the message to short-term memory
        2. Decides if input should be stored in long-term memory
        3. Retrieves relevant context
        4. Generates a response

        Args:
            user_input: User's message.

        Returns:
            ChatResponse with the assistant's reply.

        Example:
            >>> response = await agent.chat("I love Python programming")
            >>> print(response.content)
        """
        # 1. Add to short-term memory
        self._short_term.add(user_input, role="user")

        # 2. LLM decides if this should go to long-term memory
        should_store, extract_knowledge = await self._classify_memory(user_input)

        # 3. Store in long-term memory if appropriate
        if should_store:
            await self._long_term.add(
                user_input,
                infer=extract_knowledge,
                detect_contradictions=True,
            )

        # 4. Retrieve relevant context from long-term memory
        retrieved = await self._retrieve_context(user_input)

        # 5. Generate response
        response_text = await self._generate_response(user_input, retrieved)

        # 6. Add assistant response to short-term memory
        self._short_term.add(response_text, role="assistant")

        return ChatResponse(content=response_text)

    async def _classify_memory(self, content: str) -> tuple[bool, bool]:
        """Use LLM to decide if content should be stored long-term.

        Returns:
            Tuple of (should_store_long_term, should_extract_knowledge)
        """
        prompt = f"""Analyze the following user input and decide if it should be stored in long-term memory.

User input: "{content}"

Guidelines for LONG-TERM memory (store_long_term: true):
- Personal identity (name, age, occupation, location)
- Preferences and opinions (likes, dislikes, favorites)
- Important facts (contact info, important dates, relationships)
- Explicit requests to remember something
- Significant life events or experiences

Guidelines for SHORT-TERM only (store_long_term: false):
- Temporary requests or questions (help me, what is, how to)
- Greetings and small talk (hello, thanks, bye)
- References requiring context (this, that, it)
- One-time tasks or queries
- Questions about the AI itself

Respond with JSON only:
{{"store_long_term": true or false, "extract_knowledge": true or false}}

JSON response:"""

        try:
            llm = self._get_llm()
            response = await llm.complete(prompt, max_tokens=100)
            text = response.content.strip()

            # Extract JSON from response
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            return (
                bool(result.get("store_long_term", False)),
                bool(result.get("extract_knowledge", True)),
            )
        except Exception:
            # Default: don't store long-term on error
            return (False, False)

    async def _retrieve_context(self, query: str) -> list[str]:
        """Retrieve relevant memories from long-term storage."""
        try:
            results = await self._long_term.search(
                query, limit=self.max_retrieved_memories
            )
            return [f"[{r.type}] {r.text}" for r in results]
        except Exception:
            return []

    async def _generate_response(
        self,
        user_input: str,
        retrieved_context: list[str],
    ) -> str:
        """Generate a response using LLM with context."""
        parts = []

        # System prompt
        parts.append(f"System: {self.system_prompt}")

        # Long-term memory context
        if retrieved_context:
            context_str = "\n".join(f"- {c}" for c in retrieved_context)
            parts.append(f"[User's Long-term Memory]\n{context_str}")

        # Recent conversation history (short-term)
        recent = self._short_term.get(limit=self.max_context_messages)
        if len(recent) > 1:  # Exclude current message
            history_lines = []
            for msg in recent[:-1]:  # Skip the just-added user message
                history_lines.append(f"{msg.role.capitalize()}: {msg.content}")
            if history_lines:
                parts.append(f"[Recent Conversation]\n" + "\n".join(history_lines))

        # Current user input
        parts.append(f"User: {user_input}")
        parts.append("Assistant:")

        prompt = "\n\n".join(parts)

        llm = self._get_llm()
        response = await llm.complete(prompt)
        return response.content.strip()

    # =========================================================================
    # Session Management (delegates to ShortTermMemory)
    # =========================================================================

    def new_session(self) -> str:
        """Start a new conversation session.

        Returns:
            The new session ID.
        """
        return self._short_term.new_session()

    def load_session(self, session_id: str) -> bool:
        """Load an existing session.

        Args:
            session_id: Session ID to load.

        Returns:
            True if session was loaded, False if not found.
        """
        return self._short_term.load_session(session_id)

    def list_sessions(self, limit: int = 50):
        """List all sessions.

        Args:
            limit: Maximum sessions to return.

        Returns:
            List of SessionInfo objects.
        """
        return self._short_term.list_sessions(limit=limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self._short_term.delete_session(session_id)

    def get_history(self) -> list[Message]:
        """Get conversation history for current session.

        Returns:
            List of Message objects.
        """
        return self._short_term.get_all()

    def clear_history(self) -> None:
        """Clear conversation history for current session."""
        self._short_term.clear()

    # =========================================================================
    # Session Summary & Cleanup
    # =========================================================================

    async def end_session(self, save_summary: bool = True) -> str | None:
        """End the current session.

        Optionally saves a summary of the conversation to long-term memory.

        Args:
            save_summary: If True, generate and save a session summary.

        Returns:
            The summary if generated, None otherwise.
        """
        summary = None

        if save_summary:
            messages = self._short_term.get_all()
            if len(messages) >= 3:  # Only summarize if enough messages
                summary = await self._generate_summary(messages)
                if summary:
                    await self._long_term.add(
                        f"[Session Summary] {summary}",
                        infer=True,
                    )

        self._short_term.clear()
        return summary

    async def _generate_summary(self, messages: list[Message]) -> str | None:
        """Generate a summary of the conversation."""
        if not messages:
            return None

        conversation = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in messages
        )

        prompt = f"""Summarize this conversation in 1-2 sentences.
Focus only on important facts worth remembering long-term (user preferences, personal info, decisions made).
If nothing is worth remembering, respond with just "NONE".

Conversation:
{conversation}

Summary:"""

        try:
            llm = self._get_llm()
            response = await llm.complete(prompt, max_tokens=100)
            summary = response.content.strip()

            if summary.upper() == "NONE" or not summary:
                return None
            return summary
        except Exception:
            return None

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of messages deleted.
        """
        return self._short_term.cleanup_expired()

    # =========================================================================
    # Resource Management
    # =========================================================================

    async def close(self) -> None:
        """Close all connections and release resources.

        Always call this when done with the agent.
        """
        self._short_term.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False

    # =========================================================================
    # Stats & Info
    # =========================================================================

    def stats(self) -> dict[str, Any]:
        """Get combined statistics.

        Returns:
            Dict with short-term and session info.
        """
        short_term_stats = self._short_term.stats()
        return {
            "collection": self.collection,
            "session_id": self.session_id,
            "short_term": short_term_stats,
        }

    async def stats_async(self) -> dict[str, Any]:
        """Get combined statistics including long-term memory.

        Returns:
            Dict with both short-term and long-term stats.
        """
        short_term_stats = self._short_term.stats()
        long_term_stats = await self._long_term.stats()
        return {
            "collection": self.collection,
            "session_id": self.session_id,
            "short_term": short_term_stats,
            "long_term": long_term_stats,
        }
