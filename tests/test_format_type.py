"""Tests for format type detection."""

from smartql.result import FormatType, QueryResult, detect_format_type


class TestDetectFormatType:
    """Tests for detect_format_type function."""

    def test_empty_rows_returns_scalar(self):
        assert detect_format_type(None) == FormatType.SCALAR
        assert detect_format_type([]) == FormatType.SCALAR

    def test_single_value_returns_scalar(self):
        rows = [{"total": 5000}]
        assert detect_format_type(rows) == FormatType.SCALAR

    def test_single_row_two_cols_returns_pair(self):
        rows = [{"name": "Rent", "amount": 5000}]
        assert detect_format_type(rows) == FormatType.PAIR

    def test_single_row_three_plus_cols_returns_record(self):
        rows = [{"id": 1, "name": "Rent", "amount": 5000}]
        assert detect_format_type(rows) == FormatType.RECORD

        rows = [{"id": 1, "name": "Rent", "amount": 5000, "date": "2025-01-01"}]
        assert detect_format_type(rows) == FormatType.RECORD

    def test_multiple_rows_one_col_returns_list(self):
        rows = [{"name": "Rent"}, {"name": "Food"}, {"name": "Transport"}]
        assert detect_format_type(rows) == FormatType.LIST

    def test_multiple_rows_two_cols_returns_table_by_default(self):
        """M×2 defaults to TABLE; use llm_hint for PAIR_LIST."""
        rows = [
            {"name": "Rent", "amount": 5000},
            {"name": "Food", "amount": 2000},
        ]
        assert detect_format_type(rows) == FormatType.TABLE

    def test_multiple_rows_two_cols_with_llm_hint_returns_pair_list(self):
        """M×2 with llm_hint='pair_list' returns PAIR_LIST."""
        rows = [
            {"name": "Rent", "amount": 5000},
            {"name": "Food", "amount": 2000},
        ]
        assert detect_format_type(rows, llm_hint="pair_list") == FormatType.PAIR_LIST

    def test_multiple_rows_three_plus_cols_returns_table(self):
        rows = [
            {"id": 1, "name": "Rent", "amount": 5000},
            {"id": 2, "name": "Food", "amount": 2000},
        ]
        assert detect_format_type(rows) == FormatType.TABLE


class TestQueryResultFormatType:
    """Tests for QueryResult format type handling."""

    def test_compute_format_type_sets_field(self):
        result = QueryResult(
            sql="SELECT SUM(amount) FROM transactions",
            rows=[{"total": 5000}],
        )
        result.compute_format_type()
        assert result.format_type == "scalar"

    def test_compute_format_type_with_llm_hint_pair_list(self):
        """LLM hint takes priority over structural detection."""
        result = QueryResult(
            sql="SELECT name, SUM(amount) FROM categories GROUP BY name",
            rows=[
                {"name": "Rent", "total": 5000},
                {"name": "Food", "total": 2000},
            ],
            llm_format_hint="pair_list",
        )
        result.compute_format_type()
        assert result.format_type == "pair_list"

    def test_compute_format_type_m2_defaults_to_table(self):
        """M×2 without hint defaults to table."""
        result = QueryResult(
            sql="SELECT first_name, last_name FROM users",
            rows=[
                {"first_name": "John", "last_name": "Doe"},
                {"first_name": "Jane", "last_name": "Smith"},
            ],
        )
        result.compute_format_type()
        assert result.format_type == "table"

    def test_format_hint_overrides_structural_detection(self):
        """User format_hint takes priority over structural detection."""
        result = QueryResult(
            sql="SELECT * FROM transactions",
            rows=[{"id": 1, "name": "Test", "amount": 100}],
            format_hint="table",
        )
        result.compute_format_type()

        data = result.to_dict()
        assert data["format_hint"] == "table"
        assert data["format_type"] == "table"

    def test_format_hint_preserved_without_override(self):
        """Without format_hint, structural detection is used."""
        result = QueryResult(
            sql="SELECT * FROM transactions",
            rows=[{"id": 1, "name": "Test", "amount": 100}],
        )
        result.compute_format_type()

        data = result.to_dict()
        assert data["format_hint"] is None
        assert data["format_type"] == "record"

    def test_from_dict_preserves_format_fields(self):
        data = {
            "sql": "SELECT * FROM test",
            "rows": [{"a": 1}],
            "format_hint": "scalar",
            "format_type": "scalar",
        }
        result = QueryResult.from_dict(data)
        assert result.format_hint == "scalar"
        assert result.format_type == "scalar"


class TestFormatTypeEnum:
    """Tests for FormatType enum."""

    def test_enum_values(self):
        assert FormatType.SCALAR.value == "scalar"
        assert FormatType.PAIR.value == "pair"
        assert FormatType.RECORD.value == "record"
        assert FormatType.LIST.value == "list"
        assert FormatType.PAIR_LIST.value == "pair_list"
        assert FormatType.TABLE.value == "table"
        assert FormatType.RAW.value == "raw"

    def test_enum_is_string(self):
        assert isinstance(FormatType.SCALAR, str)
        assert FormatType.SCALAR == "scalar"

    def test_raw_format_via_hint(self):
        """RAW format can be requested via format_hint."""
        result = QueryResult(
            sql="SELECT * FROM test",
            rows=[{"a": 1, "b": 2}],
            format_hint="raw",
        )
        result.compute_format_type()
        assert result.format_type == "raw"


class TestHumanResponse:
    """Tests for human_response field."""

    def test_human_response_in_to_dict(self):
        result = QueryResult(
            sql="SELECT SUM(amount) FROM transactions",
            rows=[{"total": 5000}],
            human_response="You spent $5,000.00 last month.",
        )
        data = result.to_dict()
        assert data["human_response"] == "You spent $5,000.00 last month."

    def test_human_response_from_dict(self):
        data = {
            "sql": "SELECT * FROM test",
            "human_response": "Found 3 items.",
        }
        result = QueryResult.from_dict(data)
        assert result.human_response == "Found 3 items."
