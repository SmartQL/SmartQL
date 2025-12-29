"""
Database connectors with query plan analysis and EXPLAIN validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from smartql.exceptions import DatabaseError


@dataclass
class QueryPlan:
    """Represents a query execution plan."""
    raw_plan: list[dict[str, Any]]
    estimated_cost: float
    estimated_rows: int
    plan_type: str
    warnings: list[str]
    suggestions: list[str]
    is_acceptable: bool

    @classmethod
    def from_explain(
        cls,
        explain_result: list[dict[str, Any]],
        dialect: str,
        cost_threshold: float = 10000.0,
        row_threshold: int = 100000,
    ) -> "QueryPlan":
        """Parse EXPLAIN output into a QueryPlan."""
        warnings = []
        suggestions = []
        estimated_cost = 0.0
        estimated_rows = 0
        plan_type = "unknown"

        if dialect in ("postgresql", "postgres"):
            return cls._parse_postgres_plan(
                explain_result, cost_threshold, row_threshold
            )
        elif dialect == "mysql":
            return cls._parse_mysql_plan(
                explain_result, cost_threshold, row_threshold
            )
        else:
            return cls(
                raw_plan=explain_result,
                estimated_cost=0.0,
                estimated_rows=0,
                plan_type="unknown",
                warnings=[],
                suggestions=[],
                is_acceptable=True,
            )

    @classmethod
    def _parse_postgres_plan(
        cls,
        explain_result: list[dict[str, Any]],
        cost_threshold: float,
        row_threshold: int,
    ) -> "QueryPlan":
        """Parse PostgreSQL EXPLAIN output."""
        warnings = []
        suggestions = []
        estimated_cost = 0.0
        estimated_rows = 0
        plan_type = "unknown"

        if explain_result and len(explain_result) > 0:
            plan = explain_result[0]

            if "QUERY PLAN" in plan:
                plan_text = plan["QUERY PLAN"]
                if "cost=" in plan_text:
                    import re
                    cost_match = re.search(r'cost=[\d.]+\.\.([\d.]+)', plan_text)
                    if cost_match:
                        estimated_cost = float(cost_match.group(1))

                    rows_match = re.search(r'rows=(\d+)', plan_text)
                    if rows_match:
                        estimated_rows = int(rows_match.group(1))

                if "Seq Scan" in plan_text:
                    plan_type = "sequential_scan"
                    if estimated_rows > 10000:
                        warnings.append("Sequential scan on large table")
                        suggestions.append("Consider adding an index")
                elif "Index Scan" in plan_text or "Index Only Scan" in plan_text:
                    plan_type = "index_scan"
                elif "Nested Loop" in plan_text:
                    plan_type = "nested_loop"
                    if estimated_rows > 100000:
                        warnings.append("Nested loop with high row count")
                elif "Hash Join" in plan_text:
                    plan_type = "hash_join"
                elif "Merge Join" in plan_text:
                    plan_type = "merge_join"

        is_acceptable = (
            estimated_cost < cost_threshold
            and estimated_rows < row_threshold
            and "Sequential scan on large table" not in warnings
        )

        return cls(
            raw_plan=explain_result,
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            plan_type=plan_type,
            warnings=warnings,
            suggestions=suggestions,
            is_acceptable=is_acceptable,
        )

    @classmethod
    def _parse_mysql_plan(
        cls,
        explain_result: list[dict[str, Any]],
        cost_threshold: float,
        row_threshold: int,
    ) -> "QueryPlan":
        """Parse MySQL EXPLAIN output."""
        warnings = []
        suggestions = []
        estimated_cost = 0.0
        estimated_rows = 0
        plan_type = "unknown"

        for row in explain_result:
            rows = row.get("rows", 0) or 0
            estimated_rows += int(rows)

            select_type = row.get("type", "")
            if select_type == "ALL":
                plan_type = "full_table_scan"
                warnings.append(f"Full table scan on {row.get('table', 'unknown')}")
                suggestions.append("Consider adding an index")
            elif select_type in ("index", "range"):
                plan_type = "index_scan"
            elif select_type == "ref":
                plan_type = "ref_scan"

            extra = row.get("Extra", "") or ""
            if "Using filesort" in extra:
                warnings.append("Query requires filesort")
                suggestions.append("Consider adding an index for ORDER BY columns")
            if "Using temporary" in extra:
                warnings.append("Query requires temporary table")

        is_acceptable = (
            estimated_rows < row_threshold
            and "Full table scan" not in " ".join(warnings)
        )

        return cls(
            raw_plan=explain_result,
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            plan_type=plan_type,
            warnings=warnings,
            suggestions=suggestions,
            is_acceptable=is_acceptable,
        )


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    @abstractmethod
    def execute(self, sql: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dictionaries."""
        pass

    @abstractmethod
    def introspect(self) -> dict[str, Any]:
        """Introspect the database schema."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def explain(self, sql: str) -> QueryPlan:
        """Get the execution plan for a query."""
        pass

    @abstractmethod
    def validate_query(self, sql: str) -> tuple[bool, list[str]]:
        """Validate a query without executing it."""
        pass


class SQLAlchemyConnector(DatabaseConnector):
    """Database connector using SQLAlchemy with query plan analysis."""

    def __init__(self, config: dict[str, Any]):
        try:
            from sqlalchemy import create_engine, text, inspect
            from sqlalchemy.pool import QueuePool
        except ImportError:
            raise DatabaseError(
                "SQLAlchemy is required. Install with: pip install sqlalchemy"
            )

        self._text = text
        self._inspect = inspect

        connection = config.get("connection", {})
        if "url" in connection:
            url = connection["url"]
        else:
            url = self._build_url(config)

        pool_config = connection.get("pool", {})

        try:
            self.engine = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=pool_config.get("min_connections", 2),
                max_overflow=(
                    pool_config.get("max_connections", 10)
                    - pool_config.get("min_connections", 2)
                ),
                pool_timeout=pool_config.get("idle_timeout", 300),
                echo=False,
            )
            self.dialect = self.engine.dialect.name
        except Exception as e:
            raise DatabaseError(f"Failed to create database engine: {e}")

        self._cost_threshold = config.get("query_cost_threshold", 10000.0)
        self._row_threshold = config.get("query_row_threshold", 100000)

    def _build_url(self, config: dict[str, Any]) -> str:
        """Build a database URL from configuration."""
        db_type = config.get("type", "postgresql")
        connection = config.get("connection", {})

        driver_map = {
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "mysql": "mysql+pymysql",
            "sqlite": "sqlite",
            "sqlserver": "mssql+pyodbc",
            "mssql": "mssql+pyodbc",
        }

        driver = driver_map.get(db_type, "postgresql")

        if db_type == "sqlite":
            database = connection.get("database", ":memory:")
            return f"sqlite:///{database}"

        host = connection.get("host", "localhost")
        port = connection.get("port", 5432 if "postgres" in db_type else 3306)
        database = connection.get("database", "")
        user = connection.get("user", "")
        password = connection.get("password", "")

        if password:
            return f"{driver}://{user}:{password}@{host}:{port}/{database}"
        elif user:
            return f"{driver}://{user}@{host}:{port}/{database}"
        else:
            return f"{driver}://{host}:{port}/{database}"

    def execute(
        self,
        sql: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        try:
            with self.engine.connect() as conn:
                if timeout and self.dialect in ("postgresql", "postgres"):
                    conn.execute(self._text(f"SET statement_timeout = {timeout * 1000}"))

                result = conn.execute(self._text(sql), params or {})

                if result.returns_rows:
                    columns = result.keys()
                    rows = []
                    for row in result:
                        rows.append(dict(zip(columns, row)))
                    return rows

                conn.commit()
                return []

        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}")

    def explain(self, sql: str, analyze: bool = False) -> QueryPlan:
        """
        Get the execution plan for a query.

        Args:
            sql: The SQL query to explain
            analyze: If True, actually execute the query for real statistics

        Returns:
            QueryPlan with cost estimates and warnings
        """
        try:
            if self.dialect in ("postgresql", "postgres"):
                explain_sql = f"EXPLAIN {'ANALYZE ' if analyze else ''}{sql}"
            elif self.dialect == "mysql":
                explain_sql = f"EXPLAIN {sql}"
            elif self.dialect == "sqlite":
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            else:
                explain_sql = f"EXPLAIN {sql}"

            with self.engine.connect() as conn:
                result = conn.execute(self._text(explain_sql))
                rows = [dict(zip(result.keys(), row)) for row in result]

            return QueryPlan.from_explain(
                rows,
                self.dialect,
                self._cost_threshold,
                self._row_threshold,
            )

        except Exception as e:
            return QueryPlan(
                raw_plan=[],
                estimated_cost=0.0,
                estimated_rows=0,
                plan_type="error",
                warnings=[f"Failed to get execution plan: {e}"],
                suggestions=[],
                is_acceptable=False,
            )

    def validate_query(self, sql: str) -> tuple[bool, list[str]]:
        """
        Validate a query without executing it.
        Uses EXPLAIN to check for syntax errors and get plan info.
        """
        errors = []

        try:
            plan = self.explain(sql, analyze=False)

            if plan.plan_type == "error":
                return False, plan.warnings

            if not plan.is_acceptable:
                errors.extend(plan.warnings)
                return False, errors

            return True, plan.warnings

        except Exception as e:
            return False, [f"Validation failed: {e}"]

    def execute_with_validation(
        self,
        sql: str,
        params: Optional[dict] = None,
        timeout: Optional[int] = None,
        reject_expensive: bool = True,
    ) -> tuple[list[dict[str, Any]], QueryPlan]:
        """
        Execute a query with pre-validation using EXPLAIN.

        Args:
            sql: SQL query to execute
            params: Query parameters
            timeout: Query timeout in seconds
            reject_expensive: If True, reject queries over cost threshold

        Returns:
            Tuple of (results, query_plan)
        """
        plan = self.explain(sql, analyze=False)

        if reject_expensive and not plan.is_acceptable:
            raise DatabaseError(
                f"Query rejected: {'; '.join(plan.warnings)}. "
                f"Suggestions: {'; '.join(plan.suggestions)}"
            )

        results = self.execute(sql, params, timeout)
        return results, plan

    def introspect(self) -> dict[str, Any]:
        """Introspect the database schema."""
        try:
            inspector = self._inspect(self.engine)

            tables = {}
            for table_name in inspector.get_table_names():
                columns = []
                for col in inspector.get_columns(table_name):
                    columns.append({
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "default": col.get("default"),
                        "primary_key": False,
                    })

                pk_constraint = inspector.get_pk_constraint(table_name)
                pk_columns = (
                    pk_constraint.get("constrained_columns", [])
                    if pk_constraint
                    else []
                )
                for col in columns:
                    if col["name"] in pk_columns:
                        col["primary_key"] = True

                indexes = []
                for idx in inspector.get_indexes(table_name):
                    indexes.append({
                        "name": idx.get("name"),
                        "columns": idx.get("column_names", []),
                        "unique": idx.get("unique", False),
                    })

                tables[table_name] = {
                    "columns": columns,
                    "indexes": indexes,
                }

            foreign_keys = {}
            for table_name in inspector.get_table_names():
                fks = []
                for fk in inspector.get_foreign_keys(table_name):
                    fks.append({
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                    })
                if fks:
                    foreign_keys[table_name] = fks

            return {
                "dialect": self.dialect,
                "tables": tables,
                "foreign_keys": foreign_keys,
            }

        except Exception as e:
            raise DatabaseError(f"Schema introspection failed: {e}")

    def suggest_indexes(self, sql: str) -> list[str]:
        """
        Analyze a query and suggest indexes that could improve performance.
        """
        suggestions = []

        try:
            import sqlglot
            from sqlglot import exp

            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            if ast is None:
                return suggestions

            where_columns = []
            for col in ast.find_all(exp.Column):
                parent = col.parent
                if isinstance(
                    parent,
                    (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.In, exp.Like),
                ):
                    table = col.table or "unknown"
                    where_columns.append((table, col.name))

            join_columns = []
            for join in ast.find_all(exp.Join):
                on_clause = join.args.get("on")
                if on_clause:
                    for col in on_clause.find_all(exp.Column):
                        table = col.table or "unknown"
                        join_columns.append((table, col.name))

            order_columns = []
            order = ast.find(exp.Order)
            if order:
                for col in order.find_all(exp.Column):
                    table = col.table or "unknown"
                    order_columns.append((table, col.name))

            seen = set()
            for table, col in where_columns + join_columns:
                key = f"{table}.{col}"
                if key not in seen:
                    suggestions.append(
                        f"CREATE INDEX idx_{table}_{col} ON {table}({col})"
                    )
                    seen.add(key)

            if order_columns and len(order_columns) <= 3:
                cols = ", ".join(col for _, col in order_columns)
                table = order_columns[0][0]
                suggestions.append(
                    f"CREATE INDEX idx_{table}_order ON {table}({cols})"
                )

        except Exception:
            pass

        return suggestions

    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.engine.connect() as conn:
                conn.execute(self._text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()


class MockConnector(DatabaseConnector):
    """Mock database connector for testing."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._mock_data: dict[str, list[dict]] = {}
        self._mock_schema: dict[str, Any] = {}
        self.dialect = "postgresql"

    def set_mock_data(self, table: str, data: list[dict[str, Any]]) -> None:
        """Set mock data for a table."""
        self._mock_data[table] = data

    def set_mock_schema(self, schema: dict[str, Any]) -> None:
        """Set mock schema."""
        self._mock_schema = schema

    def execute(self, sql: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
        """Execute a mock query."""
        sql_upper = sql.upper()
        if "SELECT" in sql_upper:
            for table_name, data in self._mock_data.items():
                if table_name.upper() in sql_upper:
                    return data
        return []

    def explain(self, sql: str, analyze: bool = False) -> QueryPlan:
        """Return mock query plan."""
        return QueryPlan(
            raw_plan=[{"QUERY PLAN": "Mock plan"}],
            estimated_cost=100.0,
            estimated_rows=100,
            plan_type="mock",
            warnings=[],
            suggestions=[],
            is_acceptable=True,
        )

    def validate_query(self, sql: str) -> tuple[bool, list[str]]:
        """Always validates for mock."""
        return True, []

    def introspect(self) -> dict[str, Any]:
        """Return mock schema."""
        return self._mock_schema

    def test_connection(self) -> bool:
        """Always returns True for mock."""
        return True

    def close(self) -> None:
        """No-op for mock."""
        pass


def create_connector(config: dict[str, Any]) -> DatabaseConnector:
    """Factory function to create the appropriate database connector."""
    db_type = config.get("type", "postgresql").lower()

    if db_type == "mock" or config.get("mock", False):
        return MockConnector(config)

    return SQLAlchemyConnector(config)
