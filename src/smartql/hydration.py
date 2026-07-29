"""
Deterministic post-processing of result rows.

Two passes, both driven purely by schema metadata:

1. Foreign keys whose target entity declares a ``label_column`` get a sibling
   key carrying the human-readable label (``wallet_id`` -> ``wallet_label``), so
   a caller never has to show a raw id.
2. Columns marked ``hidden`` are dropped from the output, keeping internal
   plumbing (tenant ids, soft-delete flags) out of answers while still allowing
   the query planner to filter on them.

Neither pass changes the SQL the model wrote. Labels are fetched with a separate
scoped query, so hydration can never widen what the caller is allowed to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smartql.exceptions import SecurityError
from smartql.schema import Entity, Schema


@dataclass(frozen=True)
class LabelSource:
    """Where the label for one foreign-key column comes from."""

    column: str  # Column in the result rows (e.g. wallet_id)
    target_table: str  # Table holding the label (e.g. wallets)
    key_column: str  # Column in the target matching the FK value (e.g. id)
    label_column: str  # Column carrying the human-readable label (e.g. name)

    @property
    def output_key(self) -> str:
        """Sibling key the label is attached under."""
        base = self.column[:-3] if self.column.endswith("_id") else self.column
        return f"{base}_label"


class ResultHydrator:
    """Attaches FK labels to result rows and strips hidden columns."""

    def __init__(self, schema: Schema, database: Any = None, security: Any = None):
        self.schema = schema
        self.database = database
        self.security = security

    @property
    def enabled(self) -> bool:
        """True only when the schema actually declares hydration metadata."""
        for entity in self.schema.entities.values():
            if entity.label_column:
                return True
            if any(col.hidden for col in entity.columns.values()):
                return True
        return False

    def hydrate(
        self,
        rows: list[dict[str, Any]],
        tables: set[str],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return a new list of rows with labels attached and hidden columns removed.

        ``tables`` are the tables the query read; result columns are resolved
        against those entities only, so a column name is never matched against an
        unrelated table that happens to share it.
        """
        if not rows or not self.enabled:
            return rows

        entities = self._entities_for(tables)
        hidden = self._hidden_columns(entities)
        sources = self._label_sources(rows[0].keys(), entities)

        labels: dict[str, dict[Any, Any]] = {}
        for source in sources:
            values = {row[source.column] for row in rows if row.get(source.column) is not None}
            if not values:
                continue
            fetched = self._fetch_labels(source, values, context)
            if fetched:
                labels[source.column] = fetched

        hydrated = []
        for row in rows:
            new_row = {k: v for k, v in row.items() if k not in hidden}
            for source in sources:
                mapping = labels.get(source.column)
                if not mapping:
                    continue
                value = row.get(source.column)
                if value in mapping:
                    new_row[source.output_key] = mapping[value]
            hydrated.append(new_row)

        return hydrated

    def _entities_for(self, tables: set[str]) -> list[Entity]:
        """Entities backing the tables a query read."""
        lowered = {t.lower() for t in tables}
        return [e for e in self.schema.entities.values() if e.table.lower() in lowered]

    def _hidden_columns(self, entities: list[Entity]) -> set[str]:
        return {col.name for entity in entities for col in entity.columns.values() if col.hidden}

    def _label_sources(self, keys: Any, entities: list[Entity]) -> list[LabelSource]:
        """
        Work out which result columns can be labelled.

        A column is only labelled when every entity in scope agrees on where it
        points; an ambiguous name (two tables in the query defining the same
        column against different targets) is left alone rather than guessed at.
        """
        sources: list[LabelSource] = []

        for key in keys:
            candidates = set()
            for entity in entities:
                col = entity.columns.get(key)
                if col is None or not col.references:
                    continue
                target = self.schema.resolve_reference(col.references)
                if target is None:
                    continue
                target_entity, key_column = target
                if not target_entity.label_column:
                    continue
                candidates.add((target_entity.table, key_column, target_entity.label_column))

            if len(candidates) != 1:
                continue

            target_table, key_column, label_column = candidates.pop()
            sources.append(
                LabelSource(
                    column=key,
                    target_table=target_table,
                    key_column=key_column,
                    label_column=label_column,
                )
            )

        return sources

    def _fetch_labels(
        self,
        source: LabelSource,
        values: set[Any],
        context: dict[str, Any] | None,
    ) -> dict[Any, Any]:
        """
        Batch-fetch labels for one foreign key.

        The lookup goes through the same required-filter injection as any other
        query, so a label can only be read from a row the caller could have read
        directly. If scoping cannot be applied, the labels are skipped rather
        than fetched unscoped.
        """
        if self.database is None:
            return {}

        ordered = sorted(values, key=str)
        placeholders = ", ".join(f":_hydrate_{i}" for i in range(len(ordered)))
        sql = (
            f"SELECT {source.key_column}, {source.label_column} "
            f"FROM {source.target_table} "
            f"WHERE {source.key_column} IN ({placeholders})"
        )

        params: dict[str, Any] = dict(context or {})
        for i, value in enumerate(ordered):
            params[f"_hydrate_{i}"] = value

        if self.security is not None:
            try:
                sql = self.security.apply_required_filters(sql, params)
            except SecurityError:
                return {}

        try:
            rows = self.database.execute(sql, params=params)
        except Exception:
            return {}

        return {
            row[source.key_column]: row[source.label_column]
            for row in rows
            if source.key_column in row and source.label_column in row
        }
