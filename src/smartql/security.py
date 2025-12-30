"""
Security validator using sqlglot for robust SQL parsing and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


@dataclass
class SecurityConfig:
    """Security configuration for query validation."""

    mode: str = "read_only"
    allowed_tables: set[str] = field(default_factory=set)
    blocked_tables: set[str] = field(default_factory=set)
    blocked_columns: set[str] = field(default_factory=set)
    filter_only_columns: set[str] = field(default_factory=set)
    required_filters: dict[str, dict] = field(default_factory=dict)
    max_rows: int = 1000
    default_limit: int = 100
    timeout_seconds: int = 30
    max_join_depth: int = 4
    blocked_operations: list[str] = field(default_factory=list)
    max_complexity: int = 100

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SecurityConfig:
        """Create config from dictionary."""
        return cls(
            mode=config.get("mode", "read_only"),
            allowed_tables=set(config.get("allowed_tables", [])),
            blocked_tables=set(config.get("blocked_tables", [])),
            blocked_columns=set(config.get("blocked_columns", [])),
            filter_only_columns=set(config.get("filter_only_columns", [])),
            required_filters=config.get("required_filters", {}),
            max_rows=config.get("max_rows", 1000),
            default_limit=config.get("default_limit", 100),
            timeout_seconds=config.get("timeout_seconds", 30),
            max_join_depth=config.get("max_join_depth", 4),
            blocked_operations=config.get("block_operations", []),
            max_complexity=config.get("max_complexity", 100),
        )


class SQLAnalyzer:
    """
    Analyzes SQL queries using sqlglot AST parsing.
    Provides detailed information about tables, columns, joins, and operations.
    """

    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def parse(self, sql: str) -> exp.Expression | None:
        """Parse SQL and return AST."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except ParseError:
            return None

    def get_statement_type(self, sql: str) -> str | None:
        """Get the type of SQL statement (SELECT, INSERT, UPDATE, etc.)."""
        ast = self.parse(sql)
        if ast is None:
            return None

        if isinstance(ast, exp.Select):
            return "SELECT"
        elif isinstance(ast, exp.Insert):
            return "INSERT"
        elif isinstance(ast, exp.Update):
            return "UPDATE"
        elif isinstance(ast, exp.Delete):
            return "DELETE"
        elif isinstance(ast, exp.Drop):
            return "DROP"
        elif isinstance(ast, exp.Create):
            return "CREATE"
        elif isinstance(ast, exp.Alter):
            return "ALTER"
        elif isinstance(ast, exp.Truncate):
            return "TRUNCATE"
        elif isinstance(ast, exp.Union):
            return "SELECT"
        else:
            return ast.__class__.__name__.upper()

    def get_tables(self, sql: str) -> set[str]:
        """Extract all table names from the query."""
        ast = self.parse(sql)
        if ast is None:
            return set()

        tables = set()
        for table in ast.find_all(exp.Table):
            table_name = table.name
            if table_name:
                tables.add(table_name.lower())
        return tables

    def get_columns(self, sql: str) -> set[str]:
        """Extract all column references from the query."""
        ast = self.parse(sql)
        if ast is None:
            return set()

        columns = set()
        for col in ast.find_all(exp.Column):
            col_name = col.name
            table = col.table
            if table:
                columns.add(f"{table.lower()}.{col_name.lower()}")
            else:
                columns.add(col_name.lower())
        return columns

    def get_qualified_columns(self, sql: str) -> set[str]:
        """Extract columns with their table qualifications."""
        ast = self.parse(sql)
        if ast is None:
            return set()

        columns = set()
        for col in ast.find_all(exp.Column):
            if col.table:
                columns.add(f"{col.table.lower()}.{col.name.lower()}")
        return columns

    def count_joins(self, sql: str) -> int:
        """Count the number of JOIN operations."""
        ast = self.parse(sql)
        if ast is None:
            return 0
        return len(list(ast.find_all(exp.Join)))

    def has_subquery(self, sql: str) -> bool:
        """Check if query contains subqueries."""
        ast = self.parse(sql)
        if ast is None:
            return False
        subqueries = list(ast.find_all(exp.Subquery))
        return len(subqueries) > 0

    def has_union(self, sql: str) -> bool:
        """Check if query contains UNION operations."""
        ast = self.parse(sql)
        if ast is None:
            return False
        return isinstance(ast, exp.Union) or len(list(ast.find_all(exp.Union))) > 0

    def get_limit(self, sql: str) -> int | None:
        """Get the LIMIT value if present."""
        ast = self.parse(sql)
        if ast is None:
            return None

        limit = ast.find(exp.Limit)
        if limit and limit.expression:
            try:
                return int(limit.expression.this)
            except (ValueError, AttributeError):
                return None
        return None

    def get_functions(self, sql: str) -> set[str]:
        """Extract all function calls from the query."""
        ast = self.parse(sql)
        if ast is None:
            return set()

        functions = set()
        for func in ast.find_all(exp.Func):
            functions.add(func.sql_name().upper())
        return functions

    def estimate_complexity(self, sql: str) -> int:
        """
        Estimate query complexity based on:
        - Number of tables
        - Number of joins
        - Subqueries
        - Aggregations
        - UNION operations
        """
        ast = self.parse(sql)
        if ast is None:
            return 0

        score = 0
        score += len(self.get_tables(sql)) * 5
        score += self.count_joins(sql) * 10
        score += 20 if self.has_subquery(sql) else 0
        score += 15 if self.has_union(sql) else 0
        score += len(list(ast.find_all(exp.AggFunc))) * 5
        score += 10 if ast.find(exp.Group) else 0
        score += 5 if ast.find(exp.Order) else 0
        score += 10 if ast.find(exp.Having) else 0

        return score


class SecurityValidator:
    """
    Validates SQL queries against security rules using AST-based analysis.
    """

    def __init__(self, config: dict[str, Any], dialect: str = "postgres"):
        self.config = SecurityConfig.from_dict(config)
        self.analyzer = SQLAnalyzer(dialect)
        self.dialect = dialect

    def validate_query(self, sql: str) -> list[str]:
        """
        Validate a SQL query against all security rules.
        Returns list of error messages (empty if valid).
        """
        errors = []

        ast = self.analyzer.parse(sql)
        if ast is None:
            errors.append("Failed to parse SQL query - syntax error")
            return errors

        errors.extend(self._check_statement_type(sql))
        errors.extend(self._check_tables(sql))
        errors.extend(self._check_columns(sql))
        errors.extend(self._check_joins(sql))
        errors.extend(self._check_complexity(sql))
        errors.extend(self._check_dangerous_patterns(sql, ast))
        errors.extend(self._check_limit(sql))

        return errors

    def _check_statement_type(self, sql: str) -> list[str]:
        """Check if statement type is allowed."""
        errors = []
        stmt_type = self.analyzer.get_statement_type(sql)

        if self.config.mode == "read_only":
            if stmt_type and stmt_type not in ("SELECT",):
                errors.append(f"{stmt_type} statements not allowed in read-only mode")
        return errors

    def _check_tables(self, sql: str) -> list[str]:
        """Check table access permissions."""
        errors = []
        tables = self.analyzer.get_tables(sql)

        if self.config.allowed_tables:
            allowed_lower = {t.lower() for t in self.config.allowed_tables}
            for table in tables:
                if table not in allowed_lower:
                    errors.append(f"Table '{table}' not in allowlist")

        if self.config.blocked_tables:
            blocked_lower = {t.lower() for t in self.config.blocked_tables}
            for table in tables:
                if table in blocked_lower:
                    errors.append(f"Access to table '{table}' is blocked")

        return errors

    def _check_columns(self, sql: str) -> list[str]:
        """Check column access permissions."""
        errors = []
        columns = self.analyzer.get_qualified_columns(sql)

        if self.config.blocked_columns:
            blocked_lower = {c.lower() for c in self.config.blocked_columns}
            for col in columns:
                if col in blocked_lower:
                    errors.append(f"Access to column '{col}' is blocked")

        return errors

    def _check_joins(self, sql: str) -> list[str]:
        """Check JOIN depth limits."""
        errors = []
        join_count = self.analyzer.count_joins(sql)

        if join_count > self.config.max_join_depth:
            errors.append(
                f"Too many JOINs ({join_count}). Maximum allowed: {self.config.max_join_depth}"
            )

        return errors

    def _check_complexity(self, sql: str) -> list[str]:
        """Check query complexity limits."""
        errors = []
        complexity = self.analyzer.estimate_complexity(sql)

        if complexity > self.config.max_complexity:
            errors.append(
                f"Query too complex (score: {complexity}). Maximum: {self.config.max_complexity}"
            )

        return errors

    def _check_dangerous_patterns(
        self,
        sql: str,
        ast: exp.Expression,
    ) -> list[str]:
        """Check for dangerous SQL patterns."""
        errors = []

        dangerous_funcs = {"EXEC", "EXECUTE", "LOAD_FILE", "INTO_OUTFILE"}
        functions = self.analyzer.get_functions(sql)
        for func in functions:
            if func in dangerous_funcs:
                errors.append(f"Dangerous function '{func}' not allowed")

        if ";" in sql:
            parts = [p.strip() for p in sql.split(";") if p.strip()]
            if len(parts) > 1:
                errors.append("Multiple statements not allowed")

        if "--" in sql or "/*" in sql:
            errors.append("SQL comments not allowed (potential injection)")

        return errors

    def _check_limit(self, sql: str) -> list[str]:
        """Check LIMIT clause compliance."""
        errors = []
        limit = self.analyzer.get_limit(sql)

        if limit is not None and limit > self.config.max_rows:
            errors.append(f"LIMIT {limit} exceeds maximum allowed rows ({self.config.max_rows})")

        return errors

    def enforce_limit(self, sql: str) -> str:
        """
        Add or modify LIMIT clause to enforce max_rows.
        """
        current_limit = self.analyzer.get_limit(sql)

        if current_limit is None or current_limit > self.config.max_rows:
            ast = self.analyzer.parse(sql)
            if ast is None:
                return sql

            try:
                if isinstance(ast, exp.Select):
                    ast = ast.limit(self.config.max_rows)
                    return ast.sql(dialect=self.dialect)
            except Exception:
                pass

            if current_limit is None:
                return f"{sql.rstrip().rstrip(';')} LIMIT {self.config.max_rows}"

        return sql

    def apply_required_filters(
        self,
        sql: str,
        context: dict[str, Any],
    ) -> str:
        """
        Apply required filters (e.g., tenant_id) to a SQL query.
        Uses AST manipulation for safe modification.
        """
        if not self.config.required_filters:
            return sql

        ast = self.analyzer.parse(sql)
        if ast is None:
            return sql

        tables = self.analyzer.get_tables(sql)

        for table_name, filter_config in self.config.required_filters.items():
            if table_name.lower() not in tables:
                continue

            column = (
                filter_config if isinstance(filter_config, str) else filter_config.get("column")
            )
            if not column:
                continue

            filter_value = context.get(column)
            if filter_value is None:
                continue

            try:
                condition = sqlglot.parse_one(
                    f"{table_name}.{column} = '{filter_value}'",
                    dialect=self.dialect,
                )

                if isinstance(ast, exp.Select):
                    if ast.args.get("where"):
                        ast = ast.where(condition, append=True)
                    else:
                        ast = ast.where(condition)

                return ast.sql(dialect=self.dialect)
            except Exception:
                continue

        return sql

    def sanitize_identifier(self, identifier: str) -> str:
        """Sanitize a SQL identifier using sqlglot."""
        try:
            ident = sqlglot.exp.to_identifier(identifier)
            return ident.sql(dialect=self.dialect)
        except Exception:
            clean = "".join(c for c in identifier if c.isalnum() or c == "_")
            return clean

    def sanitize_value(self, value: Any) -> str:
        """Sanitize a value for safe inclusion in SQL."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)

        str_value = str(value)
        str_value = str_value.replace("'", "''")
        return f"'{str_value}'"

    def get_context_string(self) -> str:
        """Generate context string for LLM prompts describing security constraints."""
        lines = ["## SECURITY CONSTRAINTS"]

        if self.config.mode == "read_only":
            lines.append("- Only SELECT queries allowed (no INSERT/UPDATE/DELETE)")

        if self.config.allowed_tables:
            tables = ", ".join(sorted(self.config.allowed_tables))
            lines.append(f"- Allowed tables: {tables}")

        if self.config.blocked_tables:
            tables = ", ".join(sorted(self.config.blocked_tables))
            lines.append(f"- Blocked tables: {tables}")

        if self.config.blocked_columns:
            cols = ", ".join(sorted(self.config.blocked_columns))
            lines.append(f"- Blocked columns: {cols}")

        lines.append(f"- Maximum rows: {self.config.max_rows}")
        lines.append(f"- Maximum JOINs: {self.config.max_join_depth}")

        if self.config.required_filters:
            for table, fconfig in self.config.required_filters.items():
                col = fconfig if isinstance(fconfig, str) else fconfig.get("column")
                lines.append(f"- Required filter on {table}: {col}")

        return "\n".join(lines)

    def is_read_only(self, sql: str) -> bool:
        """Check if query is read-only (SELECT only)."""
        stmt_type = self.analyzer.get_statement_type(sql)
        return stmt_type == "SELECT"

    def get_tables_from_query(self, sql: str) -> set[str]:
        """Get all tables referenced in the query."""
        return self.analyzer.get_tables(sql)
