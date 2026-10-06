from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from litellm.types.utils import Usage

from smartql import server
from smartql.exceptions import LLMError
from smartql.llm import LLMConfig, LLMProvider
from smartql.usage import request_usage


def response(input_tokens, output_tokens):
    return SimpleNamespace(
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
    )


def test_http_usage_includes_all_completed_calls_and_resets_between_requests():
    llm = LLMProvider(LLMConfig(provider="openai", model="test", api_key="placeholder"))

    def ask(**kwargs):
        llm.generate("query")
        llm.generate("repair")
        return SimpleNamespace(
            sql="SELECT 1",
            explanation=None,
            confidence=None,
            rows=None,
            llm_format_hint=None,
            validation_errors=[],
            cached=False,
        )

    qw = SimpleNamespace(ask=ask, llm=llm)
    with (
        patch.dict(server._schemas, {"usage-test": qw}),
        patch.dict("os.environ", {"SMARTQL_API_KEY": "test-key"}),
        patch("smartql.llm.completion", side_effect=[response(10, 2), response(20, 3)] * 2),
    ):
        client = TestClient(server.app)
        headers = {"X-Schema-ID": "usage-test", "X-API-Key": "test-key"}
        for _ in range(2):
            result = client.post("/ask", json={"question": "question"}, headers=headers)
            assert result.status_code == 200
            assert result.json()["usage"] == {
                "input_tokens": 30,
                "output_tokens": 5,
                "total_tokens": 35,
                "complete": True,
            }
    assert request_usage.get() is None


def test_http_failure_returns_usage_of_completed_calls():
    llm = LLMProvider(LLMConfig(provider="openai", model="test", api_key="placeholder"))

    def ask(**kwargs):
        llm.generate("query")
        raise LLMError("repair failed")

    with (
        patch.dict(server._schemas, {"usage-test": SimpleNamespace(ask=ask)}),
        patch.dict("os.environ", {"SMARTQL_API_KEY": "test-key"}),
        patch("smartql.llm.completion", return_value=response(10, 2)),
    ):
        result = TestClient(server.app).post(
            "/ask",
            json={"question": "question"},
            headers={"X-Schema-ID": "usage-test", "X-API-Key": "test-key"},
        )
    assert result.status_code == 502
    assert result.json()["detail"]["usage"]["total_tokens"] == 12
    assert request_usage.get() is None


def test_missing_usage_is_marked_incomplete():
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "complete": True}
    token = request_usage.set(totals)
    try:
        llm = LLMProvider(LLMConfig(provider="openai", model="test", api_key="placeholder"))
        with patch(
            "smartql.llm.completion",
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
            ),
        ):
            llm.generate("query")
        assert totals["complete"] is False
    finally:
        request_usage.reset(token)
