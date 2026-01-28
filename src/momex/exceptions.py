"""Momex custom exceptions.

Structured exception classes with error codes, suggestions, and debug information
for better error handling in commercial applications.

Example:
    try:
        memory = Memory(collection="user:xiaoyuzhang")
        await memory.query("What do I like?")
    except CollectionNotFoundError as e:
        print(f"Error {e.error_code}: {e.message}")
        print(f"Suggestion: {e.suggestion}")
    except LLMError as e:
        logger.error(f"LLM failed: {e.details}")
"""

from __future__ import annotations

from typing import Any


class MomexError(Exception):
    """Base exception for all Momex-related errors.

    Attributes:
        message: Human-readable error message.
        error_code: Unique error identifier for programmatic handling (e.g., "MOMEX_001").
        details: Additional context about the error.
        suggestion: User-friendly suggestion for resolving the error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "MOMEX_000",
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r}, "
            f"suggestion={self.suggestion!r})"
        )

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class CollectionNotFoundError(MomexError):
    """Raised when a collection does not exist.

    Example:
        raise CollectionNotFoundError(
            collection="user:xiaoyuzhang",
            suggestion="Check the collection name or create it first."
        )
    """

    def __init__(
        self,
        collection: str,
        message: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message=message or f"Collection '{collection}' not found.",
            error_code="MOMEX_101",
            details={"collection": collection},
            suggestion=suggestion or "Check the collection name or use MemoryManager.list_collections() to see available collections.",
        )


class MemoryNotFoundError(MomexError):
    """Raised when a memory item does not exist.

    Example:
        raise MemoryNotFoundError(memory_id=123, collection="user:xiaoyuzhang")
    """

    def __init__(
        self,
        memory_id: int | str,
        collection: str | None = None,
        message: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {"memory_id": memory_id}
        if collection:
            details["collection"] = collection

        super().__init__(
            message=message or f"Memory with ID '{memory_id}' not found.",
            error_code="MOMEX_102",
            details=details,
            suggestion=suggestion or "Check the memory ID or use search() to find memories.",
        )


class ConfigurationError(MomexError):
    """Raised when configuration is invalid or missing.

    Example:
        raise ConfigurationError(
            message="Invalid YAML format",
            config_path="momex_config.yaml"
        )
    """

    def __init__(
        self,
        message: str,
        config_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if config_path:
            details["config_path"] = config_path

        super().__init__(
            message=message,
            error_code="MOMEX_201",
            details=details,
            suggestion=suggestion or "Check your configuration file or environment variables.",
        )


class ValidationError(MomexError):
    """Raised when input validation fails.

    Example:
        raise ValidationError(
            message="At least one collection is required",
            field="collections"
        )
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            error_code="MOMEX_301",
            details=details,
            suggestion=suggestion,
        )


class StorageError(MomexError):
    """Raised when database or storage operations fail.

    Example:
        raise StorageError(
            message="Failed to write to database",
            operation="add",
            db_path="momex_data/user/xiaoyuzhang/memory.db"
        )
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        db_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if operation:
            details["operation"] = operation
        if db_path:
            details["db_path"] = db_path

        super().__init__(
            message=message,
            error_code="MOMEX_401",
            details=details,
            suggestion=suggestion or "Check disk space and file permissions.",
        )


class EmbeddingError(MomexError):
    """Raised when embedding generation fails.

    Example:
        raise EmbeddingError(
            message="Failed to generate embeddings",
            model="text-embedding-ada-002"
        )
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if model:
            details["model"] = model

        super().__init__(
            message=message,
            error_code="MOMEX_501",
            details=details,
            suggestion=suggestion or "Check your OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.",
        )


class LLMError(MomexError):
    """Raised when LLM operations fail (fact extraction, query, etc.).

    Example:
        raise LLMError(
            message="Failed to extract facts from conversation",
            operation="fact_extraction"
        )
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code="MOMEX_502",
            details=details,
            suggestion=suggestion or "Check your LLM configuration (OPENAI_API_KEY, OPENAI_MODEL).",
        )


class ExportError(MomexError):
    """Raised when export operations fail.

    Example:
        raise ExportError(
            message="Failed to export memories",
            export_path="backup.json"
        )
    """

    def __init__(
        self,
        message: str,
        export_path: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        details = {}
        if export_path:
            details["export_path"] = export_path

        super().__init__(
            message=message,
            error_code="MOMEX_601",
            details=details,
            suggestion=suggestion or "Check the export path and file permissions.",
        )
