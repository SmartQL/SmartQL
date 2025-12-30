"""
Query result class for representing generated SQL and execution results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FormatType(str, Enum):
    """Result format types for UI rendering hints."""

    SCALAR = "scalar"
    PAIR = "pair"
    RECORD = "record"
    LIST = "list"
    PAIR_LIST = "pair_list"
    TABLE = "table"
    RAW = "raw"


# Valid format type values for validation
VALID_FORMAT_TYPES = {f.value for f in FormatType}


def detect_format_type(
    rows: list[dict[str, Any]] | None,
    llm_hint: str | None = None,
) -> FormatType:
    """
    Detect the format type based on result shape, with optional LLM hint.

    Priority:
        1. LLM hint (if valid)
        2. Structural detection

    Structural detection:
        1 row,  1 col  → scalar
        1 row,  2 cols → pair
        1 row,  3+ cols → record
        2+ rows, 1 col  → list
        2+ rows, 2+ cols → table
        0 rows → scalar (empty result)

    Note: pair_list requires LLM hint or explicit format_hint from caller.
    """
    if llm_hint and llm_hint in VALID_FORMAT_TYPES:
        return FormatType(llm_hint)

    if not rows:
        return FormatType.SCALAR

    row_count = len(rows)
    col_count = len(rows[0]) if rows else 0

    if row_count == 1:
        if col_count == 1:
            return FormatType.SCALAR
        elif col_count == 2:
            return FormatType.PAIR
        else:
            return FormatType.RECORD
    else:
        if col_count == 1:
            return FormatType.LIST
        else:
            return FormatType.TABLE


@dataclass
class QueryResult:
    """
    Represents the result of a natural language to SQL conversion.
    """

    # The generated SQL query
    sql: str

    # Human-readable explanation of the query
    explanation: str | None = None

    # Confidence score (0.0 to 1.0)
    confidence: float = 0.0

    # The original question
    question: str | None = None

    # Execution results (if executed)
    rows: list[dict[str, Any]] | None = None
    row_count: int | None = None

    # Format type hints for UI rendering
    format_hint: str | None = None
    llm_format_hint: str | None = None
    format_type: str | None = None

    # Human-readable response summarizing the results
    human_response: str | None = None

    # Validation status
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)

    # Execution error (if any)
    execution_error: str | None = None

    # Timing information
    generation_time_ms: float | None = None
    execution_time_ms: float | None = None

    # Parsed intent (for debugging)
    intent: dict[str, Any] | None = None

    def compute_format_type(self) -> str:
        """
        Compute and set format_type based on priority:
        1. User's format_hint (if valid)
        2. LLM's format suggestion (if valid)
        3. Structural detection from rows
        """
        if self.format_hint and self.format_hint in VALID_FORMAT_TYPES:
            self.format_type = self.format_hint
        else:
            detected = detect_format_type(self.rows, llm_hint=self.llm_format_hint)
            self.format_type = detected.value
        return self.format_type

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "sql": self.sql,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "question": self.question,
            "rows": self.rows,
            "row_count": self.row_count,
            "format_hint": self.format_hint,
            "llm_format_hint": self.llm_format_hint,
            "format_type": self.format_type,
            "human_response": self.human_response,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "execution_error": self.execution_error,
            "generation_time_ms": self.generation_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryResult:
        """Create a QueryResult from a dictionary."""
        return cls(
            sql=data.get("sql", ""),
            explanation=data.get("explanation"),
            confidence=data.get("confidence", 0.0),
            question=data.get("question"),
            rows=data.get("rows"),
            row_count=data.get("row_count"),
            format_hint=data.get("format_hint"),
            llm_format_hint=data.get("llm_format_hint"),
            format_type=data.get("format_type"),
            human_response=data.get("human_response"),
            is_valid=data.get("is_valid", False),
            validation_errors=data.get("validation_errors", []),
            execution_error=data.get("execution_error"),
            generation_time_ms=data.get("generation_time_ms"),
            execution_time_ms=data.get("execution_time_ms"),
            intent=data.get("intent"),
        )

    def to_json(self) -> str:
        """Convert the result to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_dataframe(self) -> Any:
        """
        Convert execution results to a pandas DataFrame.

        Returns:
            pandas DataFrame (requires pandas to be installed)
        """
        if not self.rows:
            raise ValueError("No execution results available")

        try:
            import pandas as pd

            return pd.DataFrame(self.rows)
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            )

    @property
    def success(self) -> bool:
        """Check if the query was generated and validated successfully."""
        return self.is_valid and not self.execution_error

    def __str__(self) -> str:
        """String representation."""
        lines = [f"SQL: {self.sql}"]
        if self.explanation:
            lines.append(f"Explanation: {self.explanation}")
        if self.confidence:
            lines.append(f"Confidence: {self.confidence:.0%}")
        if self.row_count is not None:
            lines.append(f"Rows: {self.row_count}")
        if self.validation_errors:
            lines.append(f"Validation Errors: {', '.join(self.validation_errors)}")
        if self.execution_error:
            lines.append(f"Execution Error: {self.execution_error}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<QueryResult sql='{self.sql[:50]}...' valid={self.is_valid}>"
