"""
Security validator using sqlglot for robust SQL parsing and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from smartql.exceptions import SecurityError

# Statement node types that mutate data or schema. Read-only mode rejects any
# of these appearing anywhere in the tree, not just at the root, so a write
# smuggled inside a CTE (WITH d AS (DELETE ...) SELECT ...) is caught too.
_WRITE_NODES = tuple(
    node
    for node in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Merge", None),
    )
    if node is not None
)
_DDL_NODES = tuple(
    node
    for node in (
        getattr(exp, "Drop", None),
        getattr(exp, "Create", None),
        getattr(exp, "Alter", None),
        getattr(exp, "Truncate", None),
    )
    if node is not None
)


@dataclass(frozen=True)
class ThroughFilter:
    """
    Scoping borrowed from another table.

    Used for tables that carry no tenant column of their own and are only ever
    reachable through a parent (a refund through its transaction, a budget period
    through its budget). The parent's own required filter is what does the
    scoping; this just says which column links the two.
    """

    column: str  # Local column linking to the parent
    references: str  # "<table>.<column>" on the parent

    @property
    def ref_table(self) -> str:
        return self.references.split(".", 1)[0]

    @property
    def ref_column(self) -> str:
        parts = self.references.split(".", 1)
        return parts[1] if len(parts) == 2 and parts[1] else "id"


@dataclass(frozen=True)
class RequiredFilter:
    """
    One table's tenant-scoping rule, normalised from its config entry.

    The three forms compose: a table may bind a column to a context parameter,
    pin columns to constants from config, and borrow scoping from a parent, in
    any combination.
    """

    table: str
    column: str | None = None
    param: str | None = None
    constants: dict[str, Any] = field(default_factory=dict)
    through: ThroughFilter | None = None

    @property
    def param_name(self) -> str | None:
        """Context key the column binds to; defaults to the column's own name."""
        return self.param or self.column

    @property
    def is_empty(self) -> bool:
        return not (self.column or self.constants or self.through)

    @classmethod
    def from_config(cls, table: str, config: Any) -> RequiredFilter:
        """Parse a required_filters entry. The shorthand string form is a column."""
        if isinstance(config, str):
            return cls(table=table, column=config)
        if not isinstance(config, dict):
            return cls(table=table)

        through = None
        through_config = config.get("through")
        if isinstance(through_config, dict) and through_config.get("column"):
            through = ThroughFilter(
                column=through_config["column"],
                references=through_config.get("references", ""),
            )

        constants = config.get("constants") or {}

        return cls(
            table=table,
            column=config.get("column"),
            param=config.get("param"),
            constants=dict(constants) if isinstance(constants, dict) else {},
            through=through,
        )


@dataclass
class SecurityConfig:
    """Security configuration for query validation."""

    mode: str = "read_only"
    allowed_tables: set[str] = field(default_factory=set)
    blocked_tables: set[str] = field(default_factory=set)
    blocked_columns: set[str] = field(default_factory=set)
    filter_only_columns: set[str] = field(default_factory=set)
    required_filters: dict[str, dict] = field(default_factory=dict)
    permission_key: str = "role"
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
            permission_key=config.get("permission_key", "role"),
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

    def alias_map(self, ast: exp.Expression) -> dict[str, str]:
        """Map every table alias and bare name to its real table name (all lowercased)."""
        mapping: dict[str, str] = {}
        for tbl in ast.find_all(exp.Table):
            name = (tbl.name or "").lower()
            if not name:
                continue
            mapping[name] = name
            alias = tbl.alias_or_name
            if alias:
                mapping[alias.lower()] = name
        return mapping

    def has_bound_filter(self, ast: exp.Expression, table: str, column: str) -> bool:
        """
        True if the query constrains <table>.<column> with an equality against a
        bound parameter (e.g. ``t.user_id = :user_id``).

        Requiring a placeholder, not an arbitrary expression, is deliberate: it
        prevents bypasses like comparing the tenant column to another table's
        column or to a literal. An unqualified column only resolves when a single
        base table is in scope.
        """
        return self._has_equality(
            ast,
            table,
            column,
            lambda other: isinstance(other, (exp.Placeholder, exp.Parameter)),
        )

    def has_constant_filter(self, ast: exp.Expression, table: str, column: str, value: Any) -> bool:
        """
        True if the query pins ``<table>.<column>`` to the given literal value.

        Constants come from trusted configuration (a morph type, a status), never
        from request data, so a literal is the correct and only accepted form.
        """
        expected = str(value)
        return self._has_equality(
            ast,
            table,
            column,
            lambda other: isinstance(other, exp.Literal) and str(other.this) == expected,
        )

    def has_scoped_subquery(
        self,
        ast: exp.Expression,
        table: str,
        column: str,
        ref_table: str,
    ) -> bool:
        """
        True if ``<table>.<column>`` is restricted to a subquery reading the
        parent table. The parent's own required filter is enforced separately:
        pulling it into the query makes it a referenced table, so the ordinary
        per-table check covers it.
        """
        table = table.lower()
        column = column.lower()
        ref_table = ref_table.lower()

        for in_expr in ast.find_all(exp.In):
            col_side = in_expr.this
            if not isinstance(col_side, exp.Column):
                continue
            if col_side.name.lower() != column:
                continue
            if self._resolve_column_table(ast, col_side) != table:
                continue
            query = in_expr.args.get("query")
            if query is None:
                continue
            for tbl in query.find_all(exp.Table):
                if (tbl.name or "").lower() == ref_table:
                    return True
        return False

    def _resolve_column_table(self, ast: exp.Expression, col: exp.Column) -> str | None:
        """
        Real table a column reference belongs to. An unqualified column only
        resolves when a single base table is in scope; anything ambiguous is
        deliberately left unresolved so checks fail closed.
        """
        amap = self.alias_map(ast)
        qualifier = (col.table or "").lower()
        if qualifier:
            return amap.get(qualifier)
        base_tables = set(amap.values())
        if len(base_tables) == 1:
            return next(iter(base_tables))
        return None

    def _has_equality(
        self,
        ast: exp.Expression,
        table: str,
        column: str,
        accepts: Any,
    ) -> bool:
        """Find an equality on ``<table>.<column>`` whose other side ``accepts``."""
        table = table.lower()
        column = column.lower()

        for eq in ast.find_all(exp.EQ):
            for col_side, other in ((eq.left, eq.right), (eq.right, eq.left)):
                if not isinstance(col_side, exp.Column):
                    continue
                if col_side.name.lower() != column:
                    continue
                if self._resolve_column_table(ast, col_side) != table:
                    continue
                if accepts(other):
                    return True
        return False

    def scope_alias_for_table(self, select: exp.Select, table: str) -> str | None:
        """
        Return the alias (or bare name) under which ``table`` is directly
        referenced in this SELECT's own scope, or None if it is not a direct
        source here. Tables belonging to nested sub-selects are ignored so each
        scope is filtered with its own correct alias.
        """
        table = table.lower()
        for tbl in select.find_all(exp.Table):
            if (tbl.name or "").lower() != table:
                continue
            if tbl.find_ancestor(exp.Select) is select:
                return tbl.alias_or_name
        return None

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
        self._filters = {
            table.lower(): RequiredFilter.from_config(table.lower(), fconfig)
            for table, fconfig in self.config.required_filters.items()
        }

    def _required_filter(self, table: str) -> RequiredFilter | None:
        """Parsed scoping rule for a table, or None if it has none configured."""
        return self._filters.get(table.lower())

    def _filter_bypassed(self, filter_config: Any, context: dict[str, Any] | None) -> bool:
        """
        True if the caller's runtime role exempts them from a table's required
        filter, letting trusted roles (e.g. admins, service accounts) run
        system-wide queries.

        The role is read from the context under ``permission_key`` and must be set
        by the application from the authenticated session, never from user input.
        A missing or unrecognised role is never exempt (fail safe).
        """
        if not isinstance(filter_config, dict):
            return False
        bypass_roles = filter_config.get("bypass_roles")
        if not bypass_roles:
            return False
        if not context:
            return False
        role_value = context.get(self.config.permission_key)
        if role_value is None:
            return False
        roles = role_value if isinstance(role_value, (list, tuple, set)) else [role_value]
        return any(role in bypass_roles for role in roles)

    def validate_query(
        self,
        sql: str,
        context: dict[str, Any] | None = None,
        complexity_source: str | None = None,
    ) -> list[str]:
        """
        Validate a SQL query against all security rules.
        Returns list of error messages (empty if valid).

        ``context`` carries the caller's runtime identity and role; it gates
        per-table required-filter enforcement so exempt roles can be allowed
        through.

        ``complexity_source`` is the query to judge complexity against, when
        that differs from the query being validated. Tenant scoping adds
        subqueries the caller never asked for; charging those against the
        complexity budget would mean the limit tightens every time a table gains
        a filter, rejecting questions that were answerable the day before.
        Defaults to ``sql``, so a directly validated query is unaffected. Table
        and join limits still apply to the real query, which is what actually
        executes.
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
        errors.extend(self._check_complexity(complexity_source or sql))
        errors.extend(self._check_dangerous_patterns(sql, ast))
        errors.extend(self._check_limit(sql))
        errors.extend(self._check_required_filters(sql, context))

        return errors

    def _check_statement_type(self, sql: str) -> list[str]:
        """
        In read-only mode, reject any statement that is not a pure SELECT,
        including data-modifying statements hidden inside CTEs or subqueries
        (e.g. ``WITH d AS (DELETE FROM t RETURNING id) SELECT ...``), which parse
        with a SELECT at the root.
        """
        errors: list[str] = []

        if self.config.mode != "read_only":
            return errors

        ast = self.analyzer.parse(sql)
        if ast is None:
            return errors

        stmt_type = self.analyzer.get_statement_type(sql)
        if stmt_type and stmt_type != "SELECT":
            errors.append(f"{stmt_type} statements not allowed in read-only mode")
            return errors

        nested_write = ast.find(*_WRITE_NODES, *_DDL_NODES)
        if nested_write is not None:
            kind = nested_write.__class__.__name__.upper()
            errors.append(f"{kind} statements not allowed in read-only mode")

        return errors

    def _check_required_filters(self, sql: str, context: dict[str, Any] | None = None) -> list[str]:
        """
        Enforce that every configured required-filter table is constrained by its
        tenant column bound to a parameter. This is the fail-closed backstop
        behind apply_required_filters: a query that reads a scoped table without
        the bound filter is rejected, so tenant isolation never depends on the
        LLM choosing to add the filter.

        Tables whose filter is bypassed for the caller's role (see
        _filter_bypassed) are not enforced, allowing admins or service roles to
        run system-wide queries.
        """
        errors: list[str] = []
        if not self.config.required_filters:
            return errors

        ast = self.analyzer.parse(sql)
        if ast is None:
            return errors

        referenced = self.analyzer.get_tables(sql)
        for table_name, filter_config in self.config.required_filters.items():
            if table_name.lower() not in referenced:
                continue
            if self._filter_bypassed(filter_config, context):
                continue
            rfilter = self._required_filter(table_name)
            if rfilter is None or rfilter.is_empty:
                continue
            errors.extend(self._filter_errors(ast, rfilter))

        return errors

    def _filter_errors(self, ast: exp.Expression, rfilter: RequiredFilter) -> list[str]:
        """Report every part of a table's scoping rule the query fails to satisfy."""
        errors: list[str] = []
        table = rfilter.table

        if rfilter.column and not self.analyzer.has_bound_filter(ast, table, rfilter.column):
            errors.append(
                f"Required filter on '{table}.{rfilter.column}' is missing "
                "or not bound to a parameter"
            )

        for name, value in rfilter.constants.items():
            if not self.analyzer.has_constant_filter(ast, table, name, value):
                errors.append(f"Required filter on '{table}.{name}' is missing its fixed value")

        through = rfilter.through
        if through and not self.analyzer.has_scoped_subquery(
            ast, table, through.column, through.ref_table
        ):
            errors.append(
                f"Required filter on '{table}.{through.column}' must restrict rows "
                f"to the caller's '{through.ref_table}'"
            )

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
        Inject the configured tenant filters into a query so it can only read the
        caller's own rows. For each required table referenced by the query, an
        equality on its tenant column bound to a parameter (``alias.col = :col``)
        is ANDed into every SELECT scope that references the table, using the
        alias actually in scope.

        The filter value is bound as a parameter, never interpolated, so it
        cannot alter the query structure. The bound value must be supplied at
        execution time under the column's name.

        Tables whose filter is bypassed for the caller's role are left unscoped,
        so an admin or service role can query system-wide.

        Fails closed: if a required table is referenced but its filter cannot be
        applied, or its value is absent from the context, a SecurityError is
        raised rather than returning an under-filtered query.
        """
        if not self.config.required_filters:
            return sql

        ast = self.analyzer.parse(sql)
        if ast is None:
            raise SecurityError("Cannot apply required filters: query failed to parse")

        referenced = self.analyzer.get_tables(sql)

        for table_name, filter_config in self.config.required_filters.items():
            if table_name.lower() not in referenced:
                continue

            if self._filter_bypassed(filter_config, context):
                continue

            rfilter = self._required_filter(table_name)
            if rfilter is None or rfilter.is_empty:
                continue

            for param in self._required_params(rfilter):
                if context.get(param) is None:
                    raise SecurityError(
                        f"Required filter '{table_name}' has no value for '{param}' "
                        "in the request context"
                    )

            if not self._inject_filter(ast, rfilter):
                raise SecurityError(f"Could not apply required filter on '{table_name}'")

        return ast.sql(dialect=self.dialect)

    def _required_params(self, rfilter: RequiredFilter, seen: frozenset | None = None) -> set[str]:
        """
        Context keys a scoping rule needs, following through-chains so a borrowed
        filter's parameter is demanded up front rather than failing mid-injection.
        """
        seen = seen or frozenset()
        if rfilter.table in seen:
            return set()
        seen = seen | {rfilter.table}

        params = set()
        if rfilter.param_name:
            params.add(rfilter.param_name)
        if rfilter.through:
            parent = self._required_filter(rfilter.through.ref_table)
            if parent is not None:
                params |= self._required_params(parent, seen)
        return params

    def _inject_filter(self, ast: exp.Expression, rfilter: RequiredFilter) -> bool:
        """
        AND a table's scoping conditions into every SELECT scope that references
        it directly. Returns True if it was applied at least once (i.e. the table
        really is a source somewhere).
        """
        applied = False
        # Materialised because through-scoping adds SELECT nodes as we go.
        for select in list(ast.find_all(exp.Select)):
            alias = self.analyzer.scope_alias_for_table(select, rfilter.table)
            if alias is None:
                continue
            for condition in self._filter_conditions(alias, rfilter):
                select.where(condition, append=True, copy=False)
            applied = True
        return applied

    def _filter_conditions(
        self,
        alias: str,
        rfilter: RequiredFilter,
        seen: frozenset | None = None,
    ) -> list[exp.Expression]:
        """Build the conditions that scope one table's rows to the caller."""
        seen = seen or frozenset()
        conditions: list[exp.Expression] = []

        if rfilter.column:
            conditions.append(
                exp.condition(
                    f"{alias}.{rfilter.column} = :{rfilter.param_name}",
                    dialect=self.dialect,
                )
            )

        for name, value in rfilter.constants.items():
            conditions.append(
                exp.EQ(
                    this=exp.column(name, table=alias),
                    expression=self._literal(value),
                )
            )

        if rfilter.through:
            conditions.append(self._through_condition(alias, rfilter, seen))

        return conditions

    def _through_condition(
        self,
        alias: str,
        rfilter: RequiredFilter,
        seen: frozenset,
    ) -> exp.Expression:
        """
        Restrict a table to rows whose parent the caller can read:
        ``<alias>.<column> IN (SELECT <key> FROM <parent> WHERE <parent scoping>)``.

        The parent's rule is resolved recursively, so a chain (period states ->
        budgets -> owner) lands the real tenant condition at the end of it.
        """
        through = rfilter.through
        ref_table = through.ref_table

        if ref_table in seen or ref_table == rfilter.table:
            raise SecurityError(
                f"Required filter on '{rfilter.table}' has a circular 'through' chain "
                f"via '{ref_table}'"
            )

        parent = self._required_filter(ref_table)
        if parent is None or parent.is_empty:
            raise SecurityError(
                f"Required filter on '{rfilter.table}' scopes through '{ref_table}', "
                "which has no required filter of its own"
            )

        inner = exp.select(exp.column(through.ref_column, table=ref_table)).from_(ref_table)
        for condition in self._filter_conditions(ref_table, parent, seen | {rfilter.table}):
            inner = inner.where(condition)

        return exp.In(
            this=exp.column(through.column, table=alias),
            query=exp.Subquery(this=inner),
        )

    def _literal(self, value: Any) -> exp.Expression:
        """
        Turn a configured constant into a literal node. Building the node rather
        than formatting SQL text leaves escaping to the dialect's generator, so a
        value like a backslash-separated class name survives intact.
        """
        if value is None:
            return exp.Null()
        if isinstance(value, bool):
            return exp.Boolean(this=value)
        if isinstance(value, (int, float)):
            return exp.Literal.number(value)
        return exp.Literal.string(str(value))

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

        for table, rfilter in self._filters.items():
            if rfilter.is_empty:
                continue
            parts = []
            if rfilter.column:
                parts.append(f"{rfilter.column} = :{rfilter.param_name}")
            parts.extend(f"{name} = {value!r}" for name, value in rfilter.constants.items())
            if rfilter.through:
                parts.append(
                    f"{rfilter.through.column} must belong to the caller's "
                    f"{rfilter.through.ref_table}"
                )
            lines.append(f"- Required filter on {table}: {'; '.join(parts)}")

        return "\n".join(lines)

    def is_read_only(self, sql: str) -> bool:
        """Check if query is read-only (SELECT only)."""
        stmt_type = self.analyzer.get_statement_type(sql)
        return stmt_type == "SELECT"

    def get_tables_from_query(self, sql: str) -> set[str]:
        """Get all tables referenced in the query."""
        return self.analyzer.get_tables(sql)
