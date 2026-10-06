import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from smartql import model_proxy
from smartql.exceptions import LLMError


class ModelProxyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"SMARTQL_MODEL_PROXY_URL": "http://app/model"})
        self.environment.start()
        self.grant = model_proxy.owner_grant.set("opaque-owner-grant")

    def tearDown(self):
        model_proxy.owner_grant.reset(self.grant)
        self.environment.stop()

    def test_owner_grant_required_before_transport(self):
        token = model_proxy.owner_grant.set(None)
        try:
            with patch.object(httpx, "post") as transport:
                with self.assertRaises(LLMError):
                    model_proxy.complete(messages=[])
                transport.assert_not_called()
        finally:
            model_proxy.owner_grant.reset(token)

    def test_bounded_call_and_owned_response_contract(self):
        response = SimpleNamespace(
            status_code=200, json=lambda: {"success": True, "data": {"text": "SELECT 1"}}
        )
        with patch.object(httpx, "post", return_value=response) as transport:
            result = model_proxy.complete(
                messages=[{"role": "user", "content": "question"}], max_tokens=8000
            )
        self.assertEqual("SELECT 1", result.choices[0].message.content)
        self.assertEqual(4096, transport.call_args.kwargs["json"]["max_tokens"])
        self.assertEqual(
            "Bearer opaque-owner-grant", transport.call_args.kwargs["headers"]["Authorization"]
        )

    def test_transport_retry_preserves_call_identity_and_input(self):
        response = SimpleNamespace(
            status_code=200, json=lambda: {"success": True, "data": {"text": "answer"}}
        )
        with patch.object(
            httpx, "post", side_effect=[httpx.ReadTimeout("timeout"), response]
        ) as transport:
            model_proxy.complete(messages=[{"role": "user", "content": "question"}])
        first, second = transport.call_args_list
        self.assertEqual(first.kwargs["json"], second.kwargs["json"])

    def test_denied_call_is_not_retried(self):
        with patch.object(
            httpx, "post", return_value=SimpleNamespace(status_code=402)
        ) as transport:
            with self.assertRaises(LLMError):
                model_proxy.complete(messages=[{"role": "user", "content": "question"}])
            self.assertEqual(1, transport.call_count)


if __name__ == "__main__":
    unittest.main()
