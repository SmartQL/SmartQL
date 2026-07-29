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


class TestParamRenaming:
    """A tenant column may bind to a context key that is not its own name."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "required_filters": {
                "budgets": {"column": "owner_id", "param": "user_id"},
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")

    def test_binds_the_named_param(self):
        out = self.v.apply_required_filters("SELECT amount FROM budgets", {"user_id": 7})
        assert "budgets.owner_id = :user_id" in out

    def test_context_checked_under_param_not_column(self):
        with pytest.raises(SecurityError):
            self.v.apply_required_filters("SELECT amount FROM budgets", {"owner_id": 7})

    def test_column_defaults_to_its_own_name(self):
        v = SecurityValidator(
            {"mode": "read_only", "required_filters": {"wallets": {"column": "user_id"}}},
            dialect="mysql",
        )
        out = v.apply_required_filters("SELECT id FROM wallets", {"user_id": 7})
        assert "wallets.user_id = :user_id" in out


class TestConstantFilters:
    """Morph-owned tables pin a type column to a value fixed in configuration."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "required_filters": {
                "budgets": {
                    "column": "owner_id",
                    "param": "user_id",
                    "constants": {"owner_type": "App\\Models\\User"},
                },
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")
        self.ctx = {"user_id": 7}

    def _errors(self, sql):
        return [e for e in self.v.validate_query(sql, context=self.ctx) if "Required filter" in e]

    def test_injects_the_constant(self):
        out = self.v.apply_required_filters("SELECT amount FROM budgets", self.ctx)
        assert self._errors(out) == []

    def test_owner_id_alone_is_rejected(self):
        # Without the type pin, another owner type's row with the same id leaks.
        sql = "SELECT amount FROM budgets WHERE budgets.owner_id = :user_id"
        assert any("owner_type" in e for e in self._errors(sql))

    def test_wrong_constant_value_rejected(self):
        sql = (
            "SELECT amount FROM budgets WHERE budgets.owner_id = :user_id "
            "AND budgets.owner_type = 'App\\\\Models\\\\Group'"
        )
        assert any("owner_type" in e for e in self._errors(sql))

    def test_constant_survives_a_postgres_round_trip(self):
        v = SecurityValidator(
            {
                "mode": "read_only",
                "required_filters": {
                    "budgets": {
                        "column": "owner_id",
                        "param": "user_id",
                        "constants": {"owner_type": "App\\Models\\User"},
                    }
                },
            },
            dialect="postgres",
        )
        out = v.apply_required_filters("SELECT amount FROM budgets", self.ctx)
        assert "App\\Models\\User" in out
        assert [e for e in v.validate_query(out, context=self.ctx) if "Required filter" in e] == []


class TestThroughScoping:
    """Tables with no tenant column of their own borrow their parent's scoping."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "required_filters": {
                "transactions": {"column": "user_id"},
                "budgets": {"column": "owner_id", "param": "user_id"},
                "refunds": {
                    "through": {
                        "column": "refund_transaction_id",
                        "references": "transactions.id",
                    }
                },
                "budget_period_states": {
                    "through": {"column": "budget_id", "references": "budgets.id"}
                },
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")
        self.ctx = {"user_id": 7}

    def _errors(self, sql):
        return [e for e in self.v.validate_query(sql, context=self.ctx) if "Required filter" in e]

    def test_unscoped_refund_query_rejected(self):
        assert self._errors("SELECT amount FROM refunds") != []

    def test_injection_scopes_through_the_parent(self):
        out = self.v.apply_required_filters("SELECT amount FROM refunds r", self.ctx)
        assert "r.refund_transaction_id IN (SELECT transactions.id FROM transactions" in out
        assert self._errors(out) == []

    def test_subquery_on_an_unscoped_table_rejected(self):
        # An IN against some other table is not the parent's scoping.
        sql = (
            "SELECT amount FROM refunds r WHERE r.refund_transaction_id IN "
            "(SELECT id FROM categories)"
        )
        assert self._errors(sql) != []

    def test_parent_filter_is_still_enforced_inside_the_subquery(self):
        # Pulling transactions in makes it a referenced table; its own rule applies.
        sql = (
            "SELECT amount FROM refunds r WHERE r.refund_transaction_id IN "
            "(SELECT id FROM transactions)"
        )
        assert any("transactions.user_id" in e for e in self._errors(sql))

    def test_chained_through_resolves_to_the_tenant(self):
        out = self.v.apply_required_filters("SELECT * FROM budget_period_states s", self.ctx)
        assert "budgets.owner_id = :user_id" in out
        assert self._errors(out) == []

    def test_missing_context_fails_closed(self):
        with pytest.raises(SecurityError):
            self.v.apply_required_filters("SELECT amount FROM refunds", {})

    def test_join_scopes_both_sides(self):
        sql = "SELECT t.amount FROM transactions t JOIN refunds r ON r.refund_transaction_id = t.id"
        out = self.v.apply_required_filters(sql, self.ctx)
        assert self._errors(out) == []


class TestComplexitySource:
    """Size limits judge the caller's query, not the scoping injected into it."""

    def setup_method(self):
        cfg = {
            "mode": "read_only",
            "max_complexity": 25,
            "required_filters": {
                "transactions": {"column": "user_id"},
                "refunds": {
                    "through": {
                        "column": "refund_transaction_id",
                        "references": "transactions.id",
                    }
                },
            },
        }
        self.v = SecurityValidator(cfg, dialect="mysql")
        self.ctx = {"user_id": 7}

    def test_injected_subquery_is_not_charged_to_the_caller(self):
        sql = "SELECT amount FROM refunds"
        scoped = self.v.apply_required_filters(sql, self.ctx)

        # The scoped query is over budget on its own...
        assert self.v.analyzer.estimate_complexity(scoped) > 25
        assert any("too complex" in e for e in self.v.validate_query(scoped, context=self.ctx))

        # ...but not when judged against what the caller actually asked for.
        errors = self.v.validate_query(scoped, context=self.ctx, complexity_source=sql)
        assert [e for e in errors if "too complex" in e] == []

    def test_join_limits_still_apply_to_the_real_query(self):
        # Only complexity is exempted. Joins are counted on what executes.
        v = SecurityValidator(
            {
                "mode": "read_only",
                "max_join_depth": 1,
                "required_filters": {"transactions": {"column": "user_id"}},
            },
            dialect="mysql",
        )
        sql = (
            "SELECT t.amount FROM transactions t "
            "JOIN wallets w ON t.wallet_id = w.id "
            "JOIN parties p ON t.party_id = p.id"
        )

        assert any("Too many JOINs" in e for e in v.validate_query(sql, complexity_source=sql))

    def test_a_genuinely_complex_query_is_still_rejected(self):
        sql = (
            "SELECT c.name, SUM(t.amount) FROM transactions t "
            "JOIN categorizables cz ON t.id = cz.categorizable_id "
            "JOIN categories c ON cz.category_id = c.id "
            "GROUP BY c.id ORDER BY SUM(t.amount) HAVING SUM(t.amount) > 0"
        )
        errors = self.v.validate_query(sql, context=self.ctx, complexity_source=sql)

        assert any("too complex" in e for e in errors)

    def test_tenant_scoping_is_still_checked_on_the_real_query(self):
        # The caller's own query has no filter; passing it as the complexity
        # source must not smuggle it past the required-filter check.
        sql = "SELECT amount FROM transactions"
        errors = self.v.validate_query(sql, context=self.ctx, complexity_source=sql)

        assert any("Required filter" in e for e in errors)

    def test_defaults_to_the_query_under_validation(self):
        sql = "SELECT amount FROM refunds"
        scoped = self.v.apply_required_filters(sql, self.ctx)

        assert any("too complex" in e for e in self.v.validate_query(scoped, context=self.ctx))


class TestThroughMisconfiguration:
    """A through-chain that cannot reach a tenant filter must fail closed."""

    def test_parent_without_a_filter_is_rejected(self):
        v = SecurityValidator(
            {
                "mode": "read_only",
                "required_filters": {
                    "refunds": {
                        "through": {
                            "column": "refund_transaction_id",
                            "references": "transactions.id",
                        }
                    }
                },
            },
            dialect="mysql",
        )
        with pytest.raises(SecurityError):
            v.apply_required_filters("SELECT amount FROM refunds", {"user_id": 7})

    def test_self_reference_is_rejected(self):
        v = SecurityValidator(
            {
                "mode": "read_only",
                "required_filters": {
                    "refunds": {"through": {"column": "parent_id", "references": "refunds.id"}}
                },
            },
            dialect="mysql",
        )
        with pytest.raises(SecurityError):
            v.apply_required_filters("SELECT amount FROM refunds", {"user_id": 7})

    def test_cycle_between_two_tables_is_rejected(self):
        v = SecurityValidator(
            {
                "mode": "read_only",
                "required_filters": {
                    "a": {"through": {"column": "b_id", "references": "b.id"}},
                    "b": {"through": {"column": "a_id", "references": "a.id"}},
                },
            },
            dialect="mysql",
        )
        with pytest.raises(SecurityError):
            v.apply_required_filters("SELECT id FROM a", {"user_id": 7})
