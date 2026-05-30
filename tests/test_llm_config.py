"""Tests for LLMConfig parsing and credential resolution."""

import os
from unittest import mock

import pytest

from smartql.exceptions import LLMError
from smartql.llm import LLMConfig, LLMProvider


class TestNumericCoercion:
    """Numeric config values arrive as strings from ${VAR} interpolation."""

    def test_string_retries_coerced_to_int(self):
        """retries="3" must become int 3, not stay a string (litellm compares it)."""
        cfg = LLMConfig.from_dict(
            {"provider": "gemini", "gemini": {"api_key": "k"}, "retries": "3"}
        )
        assert cfg.num_retries == 3
        assert isinstance(cfg.num_retries, int)

    def test_string_numeric_provider_fields_coerced(self):
        cfg = LLMConfig.from_dict(
            {
                "provider": "openai",
                "openai": {
                    "api_key": "k",
                    "temperature": "0.7",
                    "max_tokens": "1500",
                    "timeout": "90",
                },
            }
        )
        assert cfg.temperature == 0.7 and isinstance(cfg.temperature, float)
        assert cfg.max_tokens == 1500 and isinstance(cfg.max_tokens, int)
        assert cfg.timeout == 90.0 and isinstance(cfg.timeout, float)

    def test_empty_string_falls_back_to_default(self):
        """Unset ${VAR} interpolates to "" and must not crash on int("")."""
        cfg = LLMConfig.from_dict({"provider": "openai", "openai": {"api_key": "k"}, "retries": ""})
        assert cfg.num_retries == 3

    def test_invalid_value_falls_back_to_default(self):
        cfg = LLMConfig.from_dict(
            {"provider": "openai", "openai": {"api_key": "k", "max_tokens": "abc"}}
        )
        assert cfg.max_tokens == 2000


class TestDefaultModel:
    """Unset model must fall back to a live, provider-correct default."""

    def test_gemini_default_is_live_model(self):
        cfg = LLMConfig.from_dict({"provider": "gemini", "gemini": {"api_key": "k"}})
        assert cfg.model == "gemini/gemini-2.5-flash"

    def test_google_default_is_not_dead_gemini_pro(self):
        cfg = LLMConfig.from_dict({"provider": "google", "google": {"api_key": "k"}})
        assert "gemini-pro" not in cfg.model

    def test_groq_default_is_groq_model_not_openai(self):
        cfg = LLMConfig.from_dict({"provider": "groq", "groq": {"api_key": "k"}})
        assert cfg.model.startswith("groq/")


class TestModelPrefix:
    """OpenAI-compatible providers need their litellm prefix."""

    @pytest.mark.parametrize(
        "provider,model,expected",
        [
            ("deepseek", "deepseek-chat", "deepseek/deepseek-chat"),
            ("moonshot", "moonshot-v1-8k", "moonshot/moonshot-v1-8k"),
            ("dashscope", "qwen-plus", "dashscope/qwen-plus"),
            ("xai", "grok-beta", "xai/grok-beta"),
        ],
    )
    def test_prefix_applied(self, provider, model, expected):
        cfg = LLMConfig.from_dict(
            {"provider": provider, provider: {"api_key": "k", "model": model}}
        )
        assert cfg.model == expected


class TestCredentialResolution:
    """Credentials resolve from provider-native env vars, not a generic key."""

    def test_api_key_from_provider_env_var(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "from-env"}, clear=False):
            cfg = LLMConfig.from_dict(
                {"provider": "deepseek", "deepseek": {"model": "deepseek-chat"}}
            )
            assert cfg.api_key == "from-env"

    def test_explicit_key_wins_over_env(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=False):
            cfg = LLMConfig.from_dict({"provider": "gemini", "gemini": {"api_key": "explicit"}})
            assert cfg.api_key == "explicit"

    def test_missing_key_raises_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig.from_dict(
                {"provider": "gemini", "gemini": {"model": "gemini-2.5-flash"}}
            )
            with pytest.raises(LLMError, match="No API key configured for provider 'gemini'"):
                LLMProvider(cfg)

    def test_keyless_provider_needs_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig.from_dict({"provider": "ollama", "ollama": {"model": "llama3"}})
            # Must not raise despite no api key.
            LLMProvider(cfg)
