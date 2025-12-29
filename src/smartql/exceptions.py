"""
Custom exceptions for SmartQL.
"""


class SmartQLError(Exception):
    """Base exception for SmartQL errors."""
    pass


class SchemaError(SmartQLError):
    """Error related to schema parsing or validation."""
    pass


class SecurityError(SmartQLError):
    """Error related to security validation."""
    pass


class LLMError(SmartQLError):
    """Error related to LLM provider."""
    pass


class DatabaseError(SmartQLError):
    """Error related to database connection or queries."""
    pass


class ValidationError(SmartQLError):
    """Error related to query validation."""
    pass


class ConfigurationError(SmartQLError):
    """Error related to configuration."""
    pass


class CacheError(SmartQLError):
    """Error related to caching."""
    pass
