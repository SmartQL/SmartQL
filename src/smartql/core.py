"""
Core SmartQL class - the main entry point for natural language to SQL conversion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from smartql.cache import CacheBackend, create_cache
from smartql.database import DatabaseConnector, QueryPlan, create_connector
from smartql.exceptions import SmartQLError, ValidationError
from smartql.generator import QueryGenerator
from smartql.llm import LLMProvider, create_llm_provider
from smartql.result import QueryResult
from smartql.schema import Schema
from smartql.security import SecurityValidator


class SmartQL:
    """
    Main class for converting natural language to SQL queries.

    Features:
    - Natural language to SQL conversion using LLMs
    - YAML-based semantic layer for business context
    - Security validation with sqlglot AST parsing
    - Query plan analysis with EXPLAIN
    - Streaming support for large responses
    - Self-consistency mode for higher accuracy
    - Schema context caching

    Example:
        >>> sql = SmartQL.from_yaml("smartql.yml")
        >>> result = sql.ask("Show me all active users with orders")
        >>> print(result.sql)
    """

    def __init__(
        self,
        schema: Schema,
        database: DatabaseConnector | None = None,
        llm: LLMProvider | None = None,
        cache: CacheBackend | None = None,
    ):
        """
        Initialize SmartQL with a schema and optional components.

        Args:
            schema: The parsed Schema object containing semantic layer definition
            database: Database connector for executing queries
            llm: LLM provider for generating SQL from natural language
            cache: Cache backend for caching results
        """
        self.schema = schema
        self.database = database
        self.llm = llm
        self.cache = cache

        db_type = schema.database.get("type", "postgresql") if schema.database else "postgresql"
        self.security = SecurityValidator(schema.security, dialect=db_type)

        self.generator = QueryGenerator(
            schema=schema,
            llm=llm,
            security=self.security,
            database=database,
        )

        has_validation = hasattr(schema, "validation") and schema.validation
        self._validate_queries = (
            schema.validation.get("explain_queries", False) if has_validation else False
        )
        self._reject_expensive = (
            schema.validation.get("reject_expensive", True) if has_validation else True
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        env_file: str | Path | None = None,
    ) -> SmartQL:
        """
        Create a SmartQL instance from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file
            env_file: Optional path to .env file for environment variables

        Returns:
            Configured SmartQL instance
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        schema = Schema.from_yaml(path)

        database = None
        if schema.database:
            database = create_connector(schema.database)

        llm = None
        if schema.llm:
            llm = create_llm_provider(schema.llm)

        cache = None
        if schema.cache and schema.cache.get("enabled"):
            cache = create_cache(schema.cache)

        return cls(
            schema=schema,
            database=database,
            llm=llm,
            cache=cache,
        )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SmartQL:
        """
        Create a SmartQL instance from a dictionary configuration.
        """
        schema = Schema.from_dict(config)

        database = None
        if schema.database:
            database = create_connector(schema.database)

        llm = None
        if schema.llm:
            llm = create_llm_provider(schema.llm)

        cache = None
        if schema.cache and schema.cache.get("enabled"):
            cache = create_cache(schema.cache)

        return cls(
            schema=schema,
            database=database,
            llm=llm,
            cache=cache,
        )

    def ask(
        self,
        question: str,
        *,
        execute: bool = False,
        validate_only: bool = False,
        context: dict[str, Any] | None = None,
        use_consistency: bool = False,
        consistency_samples: int = 3,
        explain_before_execute: bool = False,
    ) -> QueryResult:
        """
        Convert a natural language question to SQL and optionally execute it.

        Args:
            question: Natural language question
            execute: Whether to execute the query and include results
            validate_only: Only validate the query, don't execute
            context: Additional context (e.g., tenant_id for filtering)
            use_consistency: Use self-consistency (multiple samples, pick best)
            consistency_samples: Number of samples for self-consistency
            explain_before_execute: Run EXPLAIN before execution to check plan

        Returns:
            QueryResult containing the SQL, explanation, and optionally results
        """
        if not self.llm:
            raise SmartQLError("No LLM provider configured. Add 'llm' section to your YAML.")

        cache_key = self._cache_key(question, context)
        if self.cache and not execute:
            cached = self.cache.get(cache_key)
            if cached:
                return QueryResult.from_dict(cached)

        result = self.generator.generate(
            question,
            context=context,
            use_consistency=use_consistency,
            consistency_samples=consistency_samples,
        )

        validation_errors = self.security.validate_query(result.sql)
        if validation_errors:
            result.validation_errors = validation_errors
            result.is_valid = False
            if validate_only:
                return result
            raise ValidationError(f"Query validation failed: {validation_errors}")

        result.is_valid = True

        if execute and self.database and not validate_only:
            if explain_before_execute or self._validate_queries:
                plan = self.database.explain(result.sql)
                result.intent = result.intent or {}
                result.intent["query_plan"] = {
                    "estimated_cost": plan.estimated_cost,
                    "estimated_rows": plan.estimated_rows,
                    "plan_type": plan.plan_type,
                    "warnings": plan.warnings,
                    "suggestions": plan.suggestions,
                }

                if not plan.is_acceptable and self._reject_expensive:
                    result.execution_error = (
                        f"Query rejected by plan analysis: {'; '.join(plan.warnings)}"
                    )
                    return result

            try:
                rows = self.database.execute(result.sql, params=context)
                result.rows = rows
                result.row_count = len(rows)
            except Exception as e:
                result.execution_error = str(e)

        if self.cache and not execute:
            self.cache.set(cache_key, result.to_dict())

        return result

    def stream_ask(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """
        Stream the SQL generation process.

        Yields SQL as it's generated, useful for long responses.

        Args:
            question: Natural language question
            context: Additional context

        Yields:
            Chunks of the generated SQL
        """
        if not self.llm:
            raise SmartQLError("No LLM provider configured.")

        schema_context = self.generator._get_schema_context()
        examples = self.schema.prompts.get("examples", [])

        prompt = self.llm._build_sql_prompt(question, schema_context, examples)
        system_prompt = self.llm._sql_system_prompt()

        yield from self.llm.stream(prompt, system_prompt=system_prompt)

    async def aask(
        self,
        question: str,
        *,
        execute: bool = False,
        context: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Async version of ask().

        Args:
            question: Natural language question
            execute: Whether to execute the query
            context: Additional context

        Returns:
            QueryResult
        """
        if not self.llm:
            raise SmartQLError("No LLM provider configured.")

        schema_context = self.generator._get_schema_context()
        examples = self.schema.prompts.get("examples", [])

        prompt = self.llm._build_sql_prompt(question, schema_context, examples)
        system_prompt = self.llm._sql_system_prompt()

        response = await self.llm.agenerate(prompt, system_prompt=system_prompt)
        result_dict = self.llm._parse_sql_response(response)

        result = QueryResult(
            sql=result_dict.get("sql", ""),
            explanation=result_dict.get("explanation"),
            confidence=result_dict.get("confidence", 0.0),
            question=question,
        )

        validation_errors = self.security.validate_query(result.sql)
        result.validation_errors = validation_errors
        result.is_valid = len(validation_errors) == 0

        if execute and self.database and result.is_valid:
            try:
                rows = self.database.execute(result.sql, params=context)
                result.rows = rows
                result.row_count = len(rows)
            except Exception as e:
                result.execution_error = str(e)

        return result

    async def astream_ask(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming version of ask().

        Args:
            question: Natural language question
            context: Additional context

        Yields:
            Chunks of the generated SQL
        """
        if not self.llm:
            raise SmartQLError("No LLM provider configured.")

        schema_context = self.generator._get_schema_context()
        examples = self.schema.prompts.get("examples", [])

        prompt = self.llm._build_sql_prompt(question, schema_context, examples)
        system_prompt = self.llm._sql_system_prompt()

        async for chunk in self.llm.astream(prompt, system_prompt=system_prompt):
            yield chunk

    def explain_query(self, sql: str) -> QueryPlan:
        """
        Get the execution plan for a SQL query.

        Args:
            sql: SQL query to analyze

        Returns:
            QueryPlan with cost estimates and suggestions
        """
        if not self.database:
            raise SmartQLError("No database configured.")

        return self.database.explain(sql)

    def suggest_indexes(self, sql: str) -> list[str]:
        """
        Suggest indexes that could improve query performance.

        Args:
            sql: SQL query to analyze

        Returns:
            List of CREATE INDEX statements
        """
        if not self.database:
            raise SmartQLError("No database configured.")

        return self.database.suggest_indexes(sql)

    def explain(self, question: str) -> str:
        """
        Get a human-readable explanation of what SQL would be generated.
        """
        result = self.ask(question)
        return result.explanation or f"This generates: {result.sql}"

    def validate(self, sql: str) -> list[str]:
        """
        Validate a SQL query against security rules.
        """
        return self.security.validate_query(sql)

    def execute(
        self,
        sql: str,
        *,
        validate: bool = True,
        explain_first: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Execute a SQL query directly.

        Args:
            sql: SQL query to execute
            validate: Whether to validate against security rules
            explain_first: Run EXPLAIN before execution

        Returns:
            List of result rows as dictionaries
        """
        if not self.database:
            raise SmartQLError("No database configured.")

        if validate:
            errors = self.security.validate_query(sql)
            if errors:
                raise ValidationError(f"Query validation failed: {errors}")

        if explain_first:
            plan = self.database.explain(sql)
            if not plan.is_acceptable and self._reject_expensive:
                raise ValidationError(f"Query rejected: {'; '.join(plan.warnings)}")

        return self.database.execute(sql)

    def introspect(self) -> dict[str, Any]:
        """
        Introspect the database schema.
        """
        if not self.database:
            raise SmartQLError("No database configured.")

        return self.database.introspect()

    def generate_yaml(self, output_path: str | Path | None = None) -> str:
        """
        Generate YAML configuration from database introspection.
        """
        if not self.database:
            raise SmartQLError("No database configured.")

        schema_info = self.database.introspect()
        yaml_content = self._generate_yaml_from_schema(schema_info)

        if output_path:
            Path(output_path).write_text(yaml_content)

        return yaml_content

    def _cache_key(self, question: str, context: dict | None = None) -> str:
        """Generate a cache key for a question."""
        import hashlib

        key_parts = [question]
        if context:
            key_parts.append(str(sorted(context.items())))
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]

    def _generate_yaml_from_schema(self, schema_info: dict) -> str:
        """Generate YAML from introspected schema."""
        import yaml

        entities = {}
        tables_data = schema_info.get("tables", {})

        for table_name, table_info in tables_data.items():
            if isinstance(table_info, dict):
                columns_list = table_info.get("columns", table_info)
            else:
                columns_list = table_info
            if isinstance(columns_list, dict):
                columns_list = columns_list.get("columns", [])

            entity = {
                "table": table_name,
                "description": f"TODO: Add description for {table_name}",
                "columns": {},
            }

            for col in columns_list:
                entity["columns"][col["name"]] = {
                    "type": self._map_db_type(col["type"]),
                    "description": f"TODO: Describe {col['name']}",
                }
                if col.get("primary_key"):
                    entity["columns"][col["name"]]["primary"] = True
                if col.get("nullable") is False:
                    entity["columns"][col["name"]]["nullable"] = False

            entities[table_name] = entity

        config = {
            "version": "1.0",
            "database": {
                "type": schema_info.get("dialect", "postgresql"),
                "connection": {
                    "url": "${DATABASE_URL}",
                },
            },
            "semantic_layer": {
                "entities": entities,
                "relationships": [],
                "business_rules": [],
            },
            "security": {
                "mode": "read_only",
                "max_rows": 1000,
            },
            "llm": {
                "provider": "openai",
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-4o",
                },
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _map_db_type(self, db_type: str) -> str:
        """Map database type to SmartQL type."""
        db_type = db_type.lower()
        if "int" in db_type:
            return "integer"
        if "char" in db_type or "text" in db_type:
            return "string"
        if "bool" in db_type:
            return "boolean"
        if any(t in db_type for t in ("decimal", "numeric", "float", "double")):
            return "decimal"
        if "timestamp" in db_type or "datetime" in db_type:
            return "datetime"
        if "date" in db_type:
            return "date"
        if "time" in db_type:
            return "time"
        if "json" in db_type:
            return "json"
        if "uuid" in db_type:
            return "uuid"
        return "string"

    def __repr__(self) -> str:
        entities = len(self.schema.entities) if self.schema.entities else 0
        llm_name = self.llm.__class__.__name__ if self.llm else None
        return f"<SmartQL entities={entities} llm={llm_name}>"
