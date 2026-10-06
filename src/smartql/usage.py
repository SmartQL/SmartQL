from contextvars import ContextVar
from typing import Any

request_usage: ContextVar[dict[str, Any] | None] = ContextVar("request_usage", default=None)


def record(response: Any) -> None:
    totals = request_usage.get()
    if totals is None:
        return
    usage = getattr(response, "usage", None)
    values = [
        getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    if any(type(value) is not int or value < 0 for value in values):
        totals["complete"] = False
        return
    for key, value in zip(("input_tokens", "output_tokens", "total_tokens"), values):
        totals[key] += value
