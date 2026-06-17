"""Tests for the validation-repair loop in SmartQL.ask()."""

import pytest

from smartql.core import SmartQL
from smartql.exceptions import ValidationError
from smartql.result import QueryResult
from smartql.schema import Schema


def _smartql() -> SmartQL:
    schema = Schema.from_dict(
        {
            "security": {"allowed_tables": ["transactions", "categories"]},
            "validation": {"repair_attempts": 1},
        }
    )
    return SmartQL(schema=schema, database=None, llm=object(), cache=None)


def test_ask_repairs_invalid_query(monkeypatch):
    sql = _smartql()

    bad = QueryResult(sql="SELECT * FROM nope", question="q")
    good = QueryResult(sql="SELECT id FROM transactions", question="q")
    generated = iter([bad, good])
    feedbacks: list[str | None] = []

    def fake_generate(question, context=None, feedback=None, **kwargs):
        feedbacks.append(feedback)
        return next(generated)

    errors = iter([["Table 'nope' not in allowlist"], []])
    monkeypatch.setattr(sql.generator, "generate", fake_generate)
    monkeypatch.setattr(sql, "_validation_errors", lambda result, context: next(errors))

    result = sql.ask("show my transactions")

    # First attempt has no feedback; the repair attempt is fed the errors.
    assert [f is None for f in feedbacks] == [True, False]
    assert "not in allowlist" in feedbacks[1]
    assert result.sql == "SELECT id FROM transactions"
    assert result.is_valid


def test_ask_raises_when_repair_is_exhausted(monkeypatch):
    sql = _smartql()
    bad = QueryResult(sql="SELECT * FROM nope", question="q")

    monkeypatch.setattr(
        sql.generator, "generate", lambda question, context=None, feedback=None, **kw: bad
    )
    monkeypatch.setattr(
        sql, "_validation_errors", lambda result, context: ["Table 'nope' not in allowlist"]
    )

    with pytest.raises(ValidationError):
        sql.ask("show my transactions")


def test_repair_feedback_lists_allowed_tables():
    sql = _smartql()
    msg = sql._repair_feedback(["Table 'nope' not in allowlist"])
    assert "not in allowlist" in msg
    assert "transactions" in msg
    assert "categories" in msg
