"""Tests for result hydration: foreign-key labels and hidden-column stripping."""

from smartql.hydration import ResultHydrator
from smartql.schema import Schema
from smartql.security import SecurityValidator

CONFIG = {
    "semantic_layer": {
        "entities": {
            "transactions": {
                "table": "transactions",
                "columns": {
                    "id": {"type": "integer", "primary": True},
                    "amount": {"type": "decimal"},
                    "user_id": {"type": "integer", "hidden": True},
                    "wallet_id": {"type": "integer", "references": "wallets.id"},
                    "party_id": {"type": "integer", "references": "parties.id"},
                },
            },
            "wallets": {
                "table": "wallets",
                "label_column": "name",
                "columns": {
                    "id": {"type": "integer", "primary": True},
                    "name": {"type": "string"},
                    "user_id": {"type": "integer", "hidden": True},
                },
            },
            "parties": {
                "table": "parties",
                "columns": {
                    "id": {"type": "integer", "primary": True},
                    "name": {"type": "string"},
                },
            },
        }
    },
    "security": {
        "mode": "read_only",
        "required_filters": {
            "transactions": {"column": "user_id"},
            "wallets": {"column": "user_id"},
        },
    },
}


class FakeDatabase:
    """Records the queries hydration issues and replays canned rows."""

    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))
        if self.fail:
            raise RuntimeError("boom")
        return self.rows


def build(rows=None, fail=False):
    schema = Schema.from_dict(CONFIG)
    db = FakeDatabase(rows=rows, fail=fail)
    security = SecurityValidator(CONFIG["security"], dialect="mysql")
    return ResultHydrator(schema, database=db, security=security), db


class TestLabelAttachment:
    def test_attaches_label_for_a_referenced_entity(self):
        hydrator, _ = build(rows=[{"id": 3, "name": "Cash"}])
        out = hydrator.hydrate(
            [{"amount": 10, "wallet_id": 3}],
            tables={"transactions"},
            context={"user_id": 7},
        )
        assert out == [{"amount": 10, "wallet_id": 3, "wallet_label": "Cash"}]

    def test_batches_one_query_per_foreign_key(self):
        hydrator, db = build(rows=[{"id": 3, "name": "Cash"}, {"id": 4, "name": "Bank"}])
        rows = [{"wallet_id": 3}, {"wallet_id": 4}, {"wallet_id": 3}]
        out = hydrator.hydrate(rows, tables={"transactions"}, context={"user_id": 7})
        assert len(db.calls) == 1
        assert [r["wallet_label"] for r in out] == ["Cash", "Bank", "Cash"]

    def test_target_without_a_label_column_is_left_alone(self):
        hydrator, db = build()
        out = hydrator.hydrate([{"party_id": 9}], tables={"transactions"}, context={"user_id": 7})
        assert out == [{"party_id": 9}]
        assert db.calls == []

    def test_null_foreign_keys_are_skipped(self):
        hydrator, db = build()
        out = hydrator.hydrate(
            [{"wallet_id": None}], tables={"transactions"}, context={"user_id": 7}
        )
        assert out == [{"wallet_id": None}]
        assert db.calls == []

    def test_unmatched_id_gets_no_label(self):
        hydrator, _ = build(rows=[])
        out = hydrator.hydrate([{"wallet_id": 3}], tables={"transactions"}, context={"user_id": 7})
        assert out == [{"wallet_id": 3}]

    def test_aggregate_columns_are_untouched(self):
        hydrator, db = build()
        rows = [{"total": 500, "month": "2026-01"}]
        assert hydrator.hydrate(rows, tables={"transactions"}, context={}) == rows
        assert db.calls == []

    def test_column_not_belonging_to_a_queried_table_is_ignored(self):
        hydrator, db = build(rows=[{"id": 3, "name": "Cash"}])
        # wallet_id is defined on transactions, which this query never read.
        out = hydrator.hydrate([{"wallet_id": 3}], tables={"parties"}, context={})
        assert out == [{"wallet_id": 3}]
        assert db.calls == []


class TestHiddenColumns:
    def test_hidden_columns_are_stripped(self):
        hydrator, _ = build()
        out = hydrator.hydrate(
            [{"amount": 10, "user_id": 7}], tables={"transactions"}, context={"user_id": 7}
        )
        assert out == [{"amount": 10}]

    def test_visible_columns_survive(self):
        hydrator, _ = build()
        rows = [{"id": 1, "amount": 10}]
        assert hydrator.hydrate(rows, tables={"transactions"}, context={}) == rows

    def test_original_rows_are_not_mutated(self):
        hydrator, _ = build()
        rows = [{"amount": 10, "user_id": 7}]
        hydrator.hydrate(rows, tables={"transactions"}, context={"user_id": 7})
        assert rows == [{"amount": 10, "user_id": 7}]


class TestTenantScopedLookups:
    def test_lookup_query_carries_the_tenant_filter(self):
        hydrator, db = build(rows=[{"id": 3, "name": "Cash"}])
        hydrator.hydrate([{"wallet_id": 3}], tables={"transactions"}, context={"user_id": 7})
        sql, params = db.calls[0]
        assert "wallets.user_id = :user_id" in sql
        assert params["user_id"] == 7

    def test_ids_are_bound_not_interpolated(self):
        hydrator, db = build(rows=[{"id": 3, "name": "Cash"}])
        hydrator.hydrate([{"wallet_id": 3}], tables={"transactions"}, context={"user_id": 7})
        sql, params = db.calls[0]
        assert ":_hydrate_0" in sql
        assert params["_hydrate_0"] == 3

    def test_missing_tenant_context_skips_the_lookup(self):
        hydrator, db = build(rows=[{"id": 3, "name": "Cash"}])
        out = hydrator.hydrate([{"wallet_id": 3}], tables={"transactions"}, context={})
        assert out == [{"wallet_id": 3}]
        assert db.calls == []

    def test_a_failing_lookup_leaves_results_intact(self):
        hydrator, _ = build(fail=True)
        out = hydrator.hydrate(
            [{"amount": 10, "wallet_id": 3}], tables={"transactions"}, context={"user_id": 7}
        )
        assert out == [{"amount": 10, "wallet_id": 3}]


class TestDisabledWithoutMetadata:
    def test_no_metadata_is_a_no_op(self):
        schema = Schema.from_dict(
            {
                "semantic_layer": {
                    "entities": {
                        "transactions": {
                            "table": "transactions",
                            "columns": {"amount": {"type": "decimal"}},
                        }
                    }
                }
            }
        )
        db = FakeDatabase()
        hydrator = ResultHydrator(schema, database=db)
        rows = [{"amount": 10}]
        assert hydrator.hydrate(rows, tables={"transactions"}, context={}) is rows
        assert db.calls == []
