"""Tests for environment variable interpolation."""

import os
from unittest import mock

from smartql.schema import _interpolate_env_vars


class TestInterpolateEnvVars:
    """Tests for environment variable interpolation."""

    def test_simple_variable(self):
        """Test simple ${VAR} interpolation."""
        with mock.patch.dict(os.environ, {"MY_VAR": "hello"}):
            result = _interpolate_env_vars("${MY_VAR}")
            assert result == "hello"

    def test_variable_with_default_uses_env(self):
        """Test ${VAR:-default} uses env value when set."""
        with mock.patch.dict(os.environ, {"MY_VAR": "from_env"}):
            result = _interpolate_env_vars("${MY_VAR:-default_value}")
            assert result == "from_env"

    def test_variable_with_default_uses_default(self):
        """Test ${VAR:-default} uses default when env not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "UNSET_VAR" in os.environ:
                del os.environ["UNSET_VAR"]
            result = _interpolate_env_vars("${UNSET_VAR:-default_value}")
            assert result == "default_value"

    def test_missing_variable_returns_empty(self):
        """Test missing variable without default returns empty string."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "MISSING_VAR" in os.environ:
                del os.environ["MISSING_VAR"]
            result = _interpolate_env_vars("${MISSING_VAR}")
            assert result == ""

    def test_mixed_text_and_variable(self):
        """Test text mixed with variable."""
        with mock.patch.dict(os.environ, {"DB_HOST": "localhost"}):
            result = _interpolate_env_vars("mysql://${DB_HOST}:3306/db")
            assert result == "mysql://localhost:3306/db"

    def test_multiple_variables(self):
        """Test multiple variables in one string."""
        with mock.patch.dict(os.environ, {"USER": "admin", "PASS": "secret"}):
            result = _interpolate_env_vars("${USER}:${PASS}")
            assert result == "admin:secret"

    def test_default_with_special_chars(self):
        """Test default value containing special characters."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "UNSET" in os.environ:
                del os.environ["UNSET"]
            result = _interpolate_env_vars("${UNSET:-http://localhost:8080}")
            assert result == "http://localhost:8080"

    def test_dict_interpolation(self):
        """Test interpolation in dictionary values."""
        with mock.patch.dict(os.environ, {"HOST": "db.example.com", "PORT": "5432"}):
            config = {
                "host": "${HOST}",
                "port": "${PORT}",
                "static": "unchanged",
            }
            result = _interpolate_env_vars(config)
            assert result == {
                "host": "db.example.com",
                "port": "5432",
                "static": "unchanged",
            }

    def test_nested_dict_interpolation(self):
        """Test interpolation in nested dictionaries."""
        with mock.patch.dict(os.environ, {"SECRET": "mysecret"}):
            config = {
                "database": {
                    "password": "${SECRET}",
                }
            }
            result = _interpolate_env_vars(config)
            assert result["database"]["password"] == "mysecret"

    def test_list_interpolation(self):
        """Test interpolation in lists."""
        with mock.patch.dict(os.environ, {"VAL1": "a", "VAL2": "b"}):
            config = ["${VAL1}", "${VAL2}", "static"]
            result = _interpolate_env_vars(config)
            assert result == ["a", "b", "static"]

    def test_non_string_passthrough(self):
        """Test that non-string values pass through unchanged."""
        assert _interpolate_env_vars(42) == 42
        assert _interpolate_env_vars(3.14) == 3.14
        assert _interpolate_env_vars(True) is True
        assert _interpolate_env_vars(None) is None

    def test_empty_default(self):
        """Test ${VAR:-} with empty default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "UNSET" in os.environ:
                del os.environ["UNSET"]
            result = _interpolate_env_vars("${UNSET:-}")
            assert result == ""

    def test_default_with_colon(self):
        """Test default value containing colons."""
        with mock.patch.dict(os.environ, {}, clear=True):
            if "URL" in os.environ:
                del os.environ["URL"]
            result = _interpolate_env_vars("${URL:-mysql://user:pass@host:3306/db}")
            assert result == "mysql://user:pass@host:3306/db"
