from __future__ import annotations

import os
import uuid
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

import httpx

from smartql.exceptions import LLMError

owner_grant: ContextVar[str | None] = ContextVar("owner_grant", default=None)


def proxy_url() -> str | None:
    return os.getenv("SMARTQL_MODEL_PROXY_URL") or None


def complete(**kwargs: Any) -> Any:
    grant = owner_grant.get()
    url = proxy_url()
    if not url or not grant:
        raise LLMError("An authenticated owner grant is required for model calls.")
    messages = kwargs["messages"]
    payload = {
        "call_id": str(uuid.uuid4()),
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),
        "max_tokens": min(int(kwargs.get("max_tokens") or 2000), 4096),
    }
    response = None
    for attempt in range(2):
        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {grant}"},
                timeout=float(kwargs.get("timeout", 120)) + 30,
                follow_redirects=False,
            )
            break
        except httpx.TransportError:
            if attempt:
                raise LLMError("The model gateway could not be reached.") from None
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else 502
        raise LLMError(f"Model gateway rejected the call ({status}).")
    data = response.json()
    if data.get("success") is not True or not isinstance(data.get("data", {}).get("text"), str):
        raise LLMError("The model gateway returned an invalid response.")
    text = data["data"]["text"]
    if kwargs.get("stream"):
        return iter(
            [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])]
        )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
