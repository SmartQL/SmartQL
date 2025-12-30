"""Tests for database module."""

from unittest import mock

from sqlalchemy import text

from smartql.database import SQLAlchemyConnector


def create_mock_connector():
    """Create a mock SQLAlchemyConnector with necessary attributes."""
    connector = SQLAlchemyConnector.__new__(SQLAlchemyConnector)
    connector.dialect = "mysql"
    connector._text = text
    return connector


class TestSQLAlchemyConnectorExecute:
    """Tests for SQLAlchemyConnector.execute method."""

    def test_execute_passes_params_to_sqlalchemy(self):
        """Test that params are passed to SQLAlchemy execute."""
        connector = create_mock_connector()

        mock_engine = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["id", "name"]
        mock_result.__iter__ = lambda self: iter([(1, "test")])

        mock_engine.connect.return_value.__enter__ = mock.MagicMock(
            return_value=mock_connection
        )
        mock_engine.connect.return_value.__exit__ = mock.MagicMock(return_value=False)
        mock_connection.execute.return_value = mock_result

        connector.engine = mock_engine

        sql = "SELECT * FROM users WHERE user_id = :user_id"
        params = {"user_id": 123}

        connector.execute(sql, params=params)

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert call_args[0][1] == params

    def test_execute_with_none_params(self):
        """Test execute with None params passes empty dict."""
        connector = create_mock_connector()

        mock_engine = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["id"]
        mock_result.__iter__ = lambda self: iter([])

        mock_engine.connect.return_value.__enter__ = mock.MagicMock(
            return_value=mock_connection
        )
        mock_engine.connect.return_value.__exit__ = mock.MagicMock(return_value=False)
        mock_connection.execute.return_value = mock_result

        connector.engine = mock_engine

        sql = "SELECT * FROM users"
        connector.execute(sql, params=None)

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args
        assert call_args[0][1] == {}

    def test_execute_returns_rows_as_dicts(self):
        """Test that execute returns rows as list of dicts."""
        connector = create_mock_connector()

        mock_engine = mock.MagicMock()
        mock_connection = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.returns_rows = True
        mock_result.keys.return_value = ["id", "name", "balance"]
        mock_result.__iter__ = lambda self: iter([
            (1, "Alice", 100.50),
            (2, "Bob", 200.75),
        ])

        mock_engine.connect.return_value.__enter__ = mock.MagicMock(
            return_value=mock_connection
        )
        mock_engine.connect.return_value.__exit__ = mock.MagicMock(return_value=False)
        mock_connection.execute.return_value = mock_result

        connector.engine = mock_engine

        result = connector.execute("SELECT * FROM users")

        assert result == [
            {"id": 1, "name": "Alice", "balance": 100.50},
            {"id": 2, "name": "Bob", "balance": 200.75},
        ]
