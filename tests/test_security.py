"""Tests for SQL security validation: read-only enforcement and tenant scoping."""

import pytest

from smartql.exceptions import SecurityError
from smartql.security import SecurityValidator


class TestReadOnlyEnforcement:
    """Read-only mode must reject any write, including ones hidden in CTEs."""

    def setup_method(self):
        self.v = SecurityValidator({"mode": "read_only"}, dialect="postgres")

    def _rejected(self, sql):
        return any("read-only" in e for e in self.v.validate_query(sql))

    def test_plain_select_allowed(self):
        assert self.v.validate_query("SELECT name FROM customers LIMIT 10") == []

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM customers",
            "UPDATE customers SET active = false",
            "INSERT INTO customers (name) VALUES ('x')",
        ],
    )
    def test_top_level_writes_rejected(self, sql):
        assert self._rejected(sql)

    def test_cte_wrapped_delete_rejected(self):
        # Parses with a SELECT at the root; the DELETE is nested in the CTE.
        sql = "WITH d AS (DELETE FROM customers RETURNING id) SELECT * FROM customers"
        assert self._rejected(sql)

    def test_cte_wrapped_update_rejected(self):
        sql = "WITH d AS (UPDATE customers SET admin = true RETURNING id) SELECT * FROM customers"
        assert self._rejected(sql)

    def test_read_only_cte_allowed(self):
        sql = "WITH c AS (SELECT id FROM customers) SELECT * FROM c LIMIT 5"
        assert self.v.validate_query(sql) == []


class TestRequiredFilterEnforcement:
    """Queries touching a scoped table must be bound to the tenant parameter."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "required_filters": {
                "transactions": {"column": "user_id"},
                "wallets": {"column": "user_id"},
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")

    def _missing(self, sql):
        return any("Required filter" in e for e in self.v.validate_query(sql))

    def test_unscoped_query_rejected(self):
        assert self._missing("SELECT amount FROM transactions")

    def test_bound_filter_unqualified_allowed(self):
        assert not self._missing("SELECT amount FROM transactions WHERE user_id = :user_id")

    def test_bound_filter_aliased_allowed(self):
        assert not self._missing("SELECT t.amount FROM transactions t WHERE t.user_id = :user_id")

    def test_literal_value_rejected(self):
        # A literal is not a bound parameter and could be any user's id.
        assert self._missing("SELECT amount FROM transactions WHERE user_id = 5")

    def test_join_column_bypass_rejected(self):
        sql = (
            "SELECT t.amount FROM transactions t JOIN wallets w ON t.wallet_id = w.id "
            "WHERE t.user_id = w.user_id"
        )
        assert self._missing(sql)

    def test_join_partially_scoped_rejected(self):
        sql = (
            "SELECT t.amount FROM transactions t JOIN wallets w ON t.wallet_id = w.id "
            "WHERE t.user_id = :user_id"
        )
        assert self._missing(sql)

    def test_join_fully_scoped_allowed(self):
        sql = (
            "SELECT t.amount FROM transactions t JOIN wallets w ON t.wallet_id = w.id "
            "WHERE t.user_id = :user_id AND w.user_id = :user_id"
        )
        assert not self._missing(sql)

    def test_unreferenced_table_not_required(self):
        assert not self._missing("SELECT name FROM categories")


class TestApplyRequiredFilters:
    """Filter injection must scope every reference, bind a parameter, and fail closed."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "required_filters": {
                "transactions": {"column": "user_id"},
                "wallets": {"column": "user_id"},
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")
        self.ctx = {"user_id": 7}

    def _rf_ok(self, sql):
        return not any("Required filter" in e for e in self.v.validate_query(sql))

    def test_injects_filter_on_bare_table(self):
        out = self.v.apply_required_filters("SELECT amount FROM transactions", self.ctx)
        assert ":user_id" in out
        assert self._rf_ok(out)

    def test_injects_using_alias(self):
        out = self.v.apply_required_filters("SELECT t.amount FROM transactions t", self.ctx)
        assert self._rf_ok(out)

    def test_injects_on_all_required_tables_in_join(self):
        sql = "SELECT t.amount FROM transactions t JOIN wallets w ON t.wallet_id = w.id"
        out = self.v.apply_required_filters(sql, self.ctx)
        assert self._rf_ok(out)

    def test_idor_query_gets_scoped(self):
        # User asks for everyone's data; injection forces it back to the caller.
        out = self.v.apply_required_filters("SELECT user_id, amount FROM transactions", self.ctx)
        assert ":user_id" in out
        assert self._rf_ok(out)

    def test_missing_context_fails_closed(self):
        with pytest.raises(SecurityError):
            self.v.apply_required_filters("SELECT amount FROM transactions", {})

    def test_subquery_scope_filtered(self):
        sql = "SELECT * FROM (SELECT amount, user_id FROM transactions) x"
        out = self.v.apply_required_filters(sql, self.ctx)
        assert self._rf_ok(out)

    def test_noop_without_required_filters(self):
        v = SecurityValidator({"mode": "read_only"}, dialect="mysql")
        sql = "SELECT amount FROM transactions"
        assert v.apply_required_filters(sql, {}) == sql


class TestRoleBypass:
    """A trusted role (e.g. admin) may run system-wide queries; everyone else is scoped."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "permission_key": "role",
            "required_filters": {
                "transactions": {"column": "user_id", "bypass_roles": ["admin"]},
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")
        self.sql = "SELECT amount FROM transactions"

    def _rf_error(self, context):
        return any("Required filter" in e for e in self.v.validate_query(self.sql, context=context))

    def test_non_admin_is_enforced(self):
        assert self._rf_error({"user_id": 7})

    def test_missing_context_is_enforced(self):
        assert self._rf_error(None)

    def test_unknown_role_is_enforced(self):
        assert self._rf_error({"role": "viewer"})

    def test_admin_is_bypassed(self):
        assert not self._rf_error({"user_id": 1, "role": "admin"})

    def test_role_list_is_bypassed(self):
        assert not self._rf_error({"role": ["admin", "auditor"]})

    def test_admin_injection_leaves_query_unscoped(self):
        out = self.v.apply_required_filters(self.sql, {"role": "admin"})
        assert ":user_id" not in out

    def test_non_admin_injection_scopes_query(self):
        out = self.v.apply_required_filters(self.sql, {"user_id": 7})
        assert ":user_id" in out
