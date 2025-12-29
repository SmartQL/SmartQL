"""
Query result class for representing generated SQL and execution results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QueryResult:
    """
    Represents the result of a natural language to SQL conversion.
    """
    
    # The generated SQL query
    sql: str
    
    # Human-readable explanation of the query
    explanation: Optional[str] = None
    
    # Confidence score (0.0 to 1.0)
    confidence: float = 0.0
    
    # The original question
    question: Optional[str] = None
    
    # Execution results (if executed)
    rows: Optional[list[dict[str, Any]]] = None
    row_count: Optional[int] = None
    
    # Validation status
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    
    # Execution error (if any)
    execution_error: Optional[str] = None
    
    # Timing information
    generation_time_ms: Optional[float] = None
    execution_time_ms: Optional[float] = None
    
    # Parsed intent (for debugging)
    intent: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "sql": self.sql,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "question": self.question,
            "rows": self.rows,
            "row_count": self.row_count,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "execution_error": self.execution_error,
            "generation_time_ms": self.generation_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "intent": self.intent,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryResult":
        """Create a QueryResult from a dictionary."""
        return cls(
            sql=data.get("sql", ""),
            explanation=data.get("explanation"),
            confidence=data.get("confidence", 0.0),
            question=data.get("question"),
            rows=data.get("rows"),
            row_count=data.get("row_count"),
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
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")
    
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
