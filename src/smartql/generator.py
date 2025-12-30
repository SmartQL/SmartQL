"""
Query generator - converts natural language to SQL using LLM and schema context.
Includes schema caching, self-consistency, and validation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from smartql.exceptions import LLMError
from smartql.llm import LLMProvider
from smartql.result import QueryResult
from smartql.schema import Schema
from smartql.security import SecurityValidator


@dataclass
class GenerationStats:
    """Statistics about query generation."""

    generation_time_ms: float
    tokens_used: int | None = None
    model: str | None = None
    cache_hit: bool = False
    consistency_samples: int = 1


class SchemaContextCache:
    """
    Caches compiled schema context to avoid regenerating on every request.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: dict[str, str] = {}
        self._access_times: dict[str, float] = {}

    def get(self, schema_hash: str) -> str | None:
        """Get cached context by schema hash."""
        if schema_hash in self._cache:
            self._access_times[schema_hash] = time.time()
            return self._cache[schema_hash]
        return None

    def set(self, schema_hash: str, context: str) -> None:
        """Cache schema context."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        self._cache[schema_hash] = context
        self._access_times[schema_hash] = time.time()

    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
        oldest = min(self._access_times, key=self._access_times.get)
        del self._cache[oldest]
        del self._access_times[oldest]

    def invalidate(self, schema_hash: str | None = None) -> None:
        """Invalidate cache entries."""
        if schema_hash:
            self._cache.pop(schema_hash, None)
            self._access_times.pop(schema_hash, None)
        else:
            self._cache.clear()
            self._access_times.clear()


class QueryGenerator:
    """
    Generates SQL queries from natural language using an LLM
    and the semantic layer schema.

    Features:
    - Schema context caching for faster repeated queries
    - Self-consistency mode for higher accuracy
    - Integration with security validator
    - Query explanation generation
    - Auto-introspection when no entities defined
    """

    _context_cache = SchemaContextCache()

    def __init__(
        self,
        schema: Schema,
        llm: LLMProvider | None = None,
        security: SecurityValidator | None = None,
        database: Any | None = None,
    ):
        self.schema = schema
        self.llm = llm
        self.security = security
        self.database = database
        self._introspected_schema: dict | None = None
        self._schema_hash = self._compute_schema_hash()

    def _compute_schema_hash(self) -> str:
        """Compute hash of schema for caching."""
        schema_str = str(self.schema.to_dict())
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def generate(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        use_consistency: bool = False,
        consistency_samples: int = 3,
        validate: bool = True,
    ) -> QueryResult:
        """
        Generate a SQL query from a natural language question.

        Args:
            question: Natural language question
            context: Additional context (e.g., user_id for tenant filtering)
            use_consistency: Use self-consistency (multiple samples)
            consistency_samples: Number of samples for consistency
            validate: Validate generated query against security rules

        Returns:
            QueryResult containing the generated SQL and metadata
        """
        if not self.llm:
            raise LLMError("No LLM provider configured")

        start_time = time.time()

        schema_context = self._get_schema_context()
        examples = self.schema.prompts.get("examples", [])
        custom_system_prompt = self.schema.prompts.get("system")

        try:
            if use_consistency:
                result = self.llm.generate_sql_with_consistency(
                    question=question,
                    schema_context=schema_context,
                    examples=examples,
                    num_samples=consistency_samples,
                )
            else:
                result = self.llm.generate_sql(
                    question=question,
                    schema_context=schema_context,
                    examples=examples,
                    custom_system_prompt=custom_system_prompt,
                )
        except Exception as e:
            raise LLMError(f"Failed to generate SQL: {e}")

        generation_time = (time.time() - start_time) * 1000

        query_result = QueryResult(
            sql=result.get("sql", ""),
            explanation=result.get("explanation"),
            confidence=result.get("confidence", 0.0),
            question=question,
            generation_time_ms=generation_time,
            intent={
                "tables_used": result.get("tables_used", []),
                "reasoning": result.get("reasoning"),
            },
        )

        if self.security and context:
            query_result.sql = self.security.apply_required_filters(
                query_result.sql,
                context,
            )

        query_result.sql = self._post_process_sql(query_result.sql)

        if validate:
            self._validate_result(query_result)

        return query_result

    def _get_schema_context(self) -> str:
        """Get schema context, using cache if available."""
        cached = self._context_cache.get(self._schema_hash)
        if cached:
            return cached

        if self.schema.entities:
            schema_context = self.schema.to_prompt_context()
        else:
            schema_context = self._get_introspected_context()

        if self.security:
            security_context = self.security.get_context_string()
            schema_context = f"{schema_context}\n\n{security_context}"

        if self.schema.prompts.get("context"):
            schema_context = f"{schema_context}\n\n{self.schema.prompts['context']}"

        self._context_cache.set(self._schema_hash, schema_context)
        return schema_context

    def _get_introspected_context(self) -> str:
        """Build schema context from database introspection."""
        if not self.database:
            return "DATABASE SCHEMA:\nNo schema information available."

        if self._introspected_schema is None:
            self._introspected_schema = self.database.introspect()

        lines = ["DATABASE SCHEMA:", ""]
        tables = self._introspected_schema.get("tables", {})

        for table_name, table_info in tables.items():
            lines.append(f"Table: {table_name}")
            lines.append("  Columns:")

            columns = table_info if isinstance(table_info, list) else table_info.get("columns", [])
            for col in columns:
                col_info = f"    - {col['name']} ({col['type']})"
                if col.get("primary_key"):
                    col_info += " [PRIMARY KEY]"
                if col.get("nullable") is False:
                    col_info += " [NOT NULL]"
                lines.append(col_info)
            lines.append("")

        return "\n".join(lines)

    def _post_process_sql(self, sql: str) -> str:
        """Post-process generated SQL for safety and correctness."""
        sql = sql.strip().rstrip(";")

        if self.security:
            sql = self.security.enforce_limit(sql)

        return sql

    def _validate_result(self, result: QueryResult) -> None:
        """Validate generated query against security rules."""
        if not self.security:
            result.is_valid = True
            return

        errors = self.security.validate_query(result.sql)
        result.validation_errors = errors
        result.is_valid = len(errors) == 0

    def explain_query(self, sql: str) -> str:
        """Generate a human-readable explanation of a SQL query."""
        if not self.llm:
            return "No LLM configured for explanations"

        prompt = f"""Explain this SQL query in simple, non-technical terms:

```sql
{sql}
```

Provide a brief explanation covering:
1. What data is being retrieved
2. What filters or conditions are applied
3. How tables are connected (if multiple)
4. Any aggregations or groupings"""

        system_prompt = (
            "You are a SQL expert. Explain queries clearly and concisely "
            "for non-technical users. Avoid jargon."
        )

        return self.llm.generate(prompt, system_prompt=system_prompt)

    def suggest_improvements(self, sql: str) -> list[str]:
        """Suggest improvements for a SQL query."""
        if not self.llm:
            return []

        prompt = f"""Analyze this SQL query and suggest improvements:

```sql
{sql}
```

Consider:
- Performance optimizations
- Readability improvements
- Best practices

Return suggestions as a JSON array of strings."""

        try:
            response = self.llm.generate(
                prompt,
                system_prompt="You are a SQL optimization expert. Return only valid JSON.",
            )
            import json

            return json.loads(response)
        except Exception:
            return []

    def decompose_complex_question(
        self,
        question: str,
    ) -> list[dict[str, str]]:
        """
        Decompose a complex question into simpler sub-questions.
        Useful for multi-step queries.
        """
        if not self.llm:
            return [{"question": question, "dependency": None}]

        schema_context = self._get_schema_context()

        prompt = f"""Given this database schema:
{schema_context}

Decompose this complex question into simpler sub-questions that can be answered
with individual SQL queries:

Question: {question}

Return a JSON array where each item has:
- "question": the sub-question
- "dependency": index of question this depends on (null if independent)

Example:
[
    {{"question": "Get all active users", "dependency": null}},
    {{"question": "Get total orders for those users", "dependency": 0}}
]"""

        try:
            response = self.llm.generate(
                prompt,
                system_prompt="You are a SQL expert. Return only valid JSON.",
            )
            import json
            import re

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return [{"question": question, "dependency": None}]

    def invalidate_cache(self) -> None:
        """Invalidate the schema context cache."""
        self._context_cache.invalidate(self._schema_hash)

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all cached schema contexts."""
        cls._context_cache.invalidate()
