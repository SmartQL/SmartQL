"""
SmartQL - Natural Language to SQL, Database First.

A language-agnostic library that converts natural language questions
into SQL queries using a YAML-based semantic layer.

Features:
- LiteLLM integration for 100+ model support
- sqlglot-based SQL parsing for robust security
- Query plan analysis with EXPLAIN
- Streaming and async support
- Self-consistency mode for higher accuracy
"""

from smartql.core import SmartQL
from smartql.database import QueryPlan, create_connector
from smartql.exceptions import (
    DatabaseError,
    LLMError,
    SchemaError,
    SecurityError,
    SmartQLError,
    ValidationError,
)
from smartql.generator import QueryGenerator
from smartql.llm import LLMConfig, LLMProvider, create_llm_provider
from smartql.result import QueryResult
from smartql.schema import BusinessRule, Column, Entity, Relationship, Schema
from smartql.security import SecurityValidator, SQLAnalyzer

__version__ = "1.0.0"
__all__ = [
    # Core
    "SmartQL",
    "QueryResult",
    "QueryPlan",
    # Schema
    "Schema",
    "Entity",
    "Column",
    "Relationship",
    "BusinessRule",
    # LLM
    "LLMProvider",
    "LLMConfig",
    "create_llm_provider",
    # Database
    "create_connector",
    # Security
    "SecurityValidator",
    "SQLAnalyzer",
    # Generator
    "QueryGenerator",
    # Exceptions
    "SmartQLError",
    "SchemaError",
    "SecurityError",
    "LLMError",
    "DatabaseError",
    "ValidationError",
]


def run_server(**kwargs):
    """Run the SmartQL HTTP API server."""
    from smartql.server import run_server as _run

    return _run(**kwargs)
