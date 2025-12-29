"""
Schema classes for parsing and representing the YAML semantic layer configuration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from smartql.exceptions import SchemaError


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in configuration values."""
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name, "")
            return env_value
        
        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


@dataclass
class Column:
    """Represents a column in an entity."""
    
    name: str
    type: str
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    primary: bool = False
    unique: bool = False
    nullable: bool = True
    default: Any = None
    searchable: bool = False
    hidden: bool = False
    values: Optional[list[str]] = None  # For enum types
    references: Optional[str] = None  # Foreign key reference
    
    @classmethod
    def from_dict(cls, name: str, data: Union[dict, str]) -> "Column":
        """Create a Column from a dictionary or shorthand string."""
        if isinstance(data, str):
            # Shorthand: just the type
            return cls(name=name, type=data)
        
        return cls(
            name=name,
            type=data.get("type", "string"),
            description=data.get("description"),
            aliases=data.get("aliases", []),
            primary=data.get("primary", False),
            unique=data.get("unique", False),
            nullable=data.get("nullable", True),
            default=data.get("default"),
            searchable=data.get("searchable", False),
            hidden=data.get("hidden", False),
            values=data.get("values"),
            references=data.get("references"),
        )


@dataclass
class Entity:
    """Represents a semantic entity (maps to a database table)."""
    
    name: str  # Semantic name
    table: str  # Actual table name
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    columns: dict[str, Column] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Entity":
        """Create an Entity from a dictionary."""
        columns = {}
        for col_name, col_data in data.get("columns", {}).items():
            columns[col_name] = Column.from_dict(col_name, col_data)
        
        return cls(
            name=name,
            table=data.get("table", name),
            description=data.get("description"),
            aliases=data.get("aliases", []),
            columns=columns,
        )
    
    def get_primary_key(self) -> Optional[Column]:
        """Get the primary key column."""
        for col in self.columns.values():
            if col.primary:
                return col
        return None
    
    def get_column_by_alias(self, alias: str) -> Optional[Column]:
        """Find a column by name or alias."""
        alias_lower = alias.lower()
        for col in self.columns.values():
            if col.name.lower() == alias_lower:
                return col
            if alias_lower in [a.lower() for a in col.aliases]:
                return col
        return None


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    
    name: str
    type: str  # one_to_one, one_to_many, many_to_one, many_to_many
    from_entity: str
    to_entity: str
    foreign_key: Optional[str] = None
    pivot_table: Optional[str] = None  # For many_to_many
    pivot_from: Optional[str] = None
    pivot_to: Optional[str] = None
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create a Relationship from a dictionary."""
        return cls(
            name=data.get("name", ""),
            type=data.get("type", "one_to_many"),
            from_entity=data.get("from", ""),
            to_entity=data.get("to", ""),
            foreign_key=data.get("foreign_key"),
            pivot_table=data.get("pivot_table"),
            pivot_from=data.get("pivot_from"),
            pivot_to=data.get("pivot_to"),
            description=data.get("description"),
        )


@dataclass
class BusinessRule:
    """Represents a named business rule/condition."""
    
    name: str
    definition: str  # SQL WHERE clause fragment
    description: Optional[str] = None
    applies_to: list[str] = field(default_factory=list)  # Entity names, or ["*"] for all
    aliases: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "BusinessRule":
        """Create a BusinessRule from a dictionary."""
        applies_to = data.get("applies_to", ["*"])
        if isinstance(applies_to, str):
            applies_to = [applies_to]
        
        return cls(
            name=data.get("name", ""),
            definition=data.get("definition", ""),
            description=data.get("description"),
            applies_to=applies_to,
            aliases=data.get("aliases", []),
            parameters=data.get("parameters", {}),
        )
    
    def applies_to_entity(self, entity_name: str) -> bool:
        """Check if this rule applies to a given entity."""
        return "*" in self.applies_to or entity_name in self.applies_to


@dataclass
class Aggregation:
    """Represents a named aggregation."""
    
    name: str
    entity: str
    expression: str
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Aggregation":
        """Create an Aggregation from a dictionary."""
        return cls(
            name=data.get("name", ""),
            entity=data.get("entity", ""),
            expression=data.get("expression", ""),
            description=data.get("description"),
            aliases=data.get("aliases", []),
        )


@dataclass
class Metric:
    """Represents a calculated metric."""
    
    name: str
    description: Optional[str] = None
    formula: str = ""
    entities: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Metric":
        """Create a Metric from a dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description"),
            formula=data.get("formula", ""),
            entities=data.get("entities", []),
            filters=data.get("filters", []),
            aliases=data.get("aliases", []),
        )


@dataclass
class Schema:
    """
    Complete schema representing the semantic layer configuration.
    """
    
    version: str = "1.0"
    database: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    business_rules: list[BusinessRule] = field(default_factory=list)
    aggregations: list[Aggregation] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    security: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    prompts: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Schema":
        """
        Load and parse a YAML configuration file.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            Parsed Schema instance
            
        Raises:
            SchemaError: If the file cannot be read or parsed
        """
        path = Path(path)
        if not path.exists():
            raise SchemaError(f"Configuration file not found: {path}")
        
        try:
            with open(path, "r") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SchemaError(f"Invalid YAML in {path}: {e}")
        
        if not raw_config:
            raise SchemaError(f"Empty configuration file: {path}")
        
        # Interpolate environment variables
        config = _interpolate_env_vars(raw_config)
        
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Schema":
        """
        Create a Schema from a dictionary configuration.
        
        Args:
            config: Dictionary containing the configuration
            
        Returns:
            Parsed Schema instance
        """
        # Parse entities
        entities = {}
        semantic_layer = config.get("semantic_layer", {})
        for name, data in semantic_layer.get("entities", {}).items():
            entities[name] = Entity.from_dict(name, data)
        
        # Parse relationships
        relationships = []
        for data in semantic_layer.get("relationships", []):
            relationships.append(Relationship.from_dict(data))
        
        # Parse business rules
        business_rules = []
        for data in semantic_layer.get("business_rules", []):
            business_rules.append(BusinessRule.from_dict(data))
        
        # Parse aggregations
        aggregations = []
        for data in semantic_layer.get("aggregations", []):
            aggregations.append(Aggregation.from_dict(data))
        
        # Parse metrics
        metrics = []
        for data in semantic_layer.get("metrics", []):
            metrics.append(Metric.from_dict(data))
        
        return cls(
            version=config.get("version", "1.0"),
            database=config.get("database", {}),
            entities=entities,
            relationships=relationships,
            business_rules=business_rules,
            aggregations=aggregations,
            metrics=metrics,
            security=config.get("security", {}),
            llm=config.get("llm", {}),
            cache=config.get("cache", {}),
            prompts=config.get("prompts", {}),
            logging=config.get("logging", {}),
        )
    
    def get_entity_by_alias(self, alias: str) -> Optional[Entity]:
        """Find an entity by name or alias."""
        alias_lower = alias.lower()
        
        # Check direct name match
        if alias_lower in self.entities:
            return self.entities[alias_lower]
        
        # Check aliases
        for entity in self.entities.values():
            if entity.name.lower() == alias_lower:
                return entity
            if alias_lower in [a.lower() for a in entity.aliases]:
                return entity
            # Also check table name
            if entity.table.lower() == alias_lower:
                return entity
        
        return None
    
    def get_rules_for_entity(self, entity_name: str) -> list[BusinessRule]:
        """Get all business rules that apply to an entity."""
        return [r for r in self.business_rules if r.applies_to_entity(entity_name)]
    
    def get_relationships_for_entity(self, entity_name: str) -> list[Relationship]:
        """Get all relationships involving an entity."""
        return [
            r for r in self.relationships
            if r.from_entity == entity_name or r.to_entity == entity_name
        ]
    
    def get_relationship(self, from_entity: str, to_entity: str) -> Optional[Relationship]:
        """Get the relationship between two entities."""
        for rel in self.relationships:
            if rel.from_entity == from_entity and rel.to_entity == to_entity:
                return rel
            if rel.from_entity == to_entity and rel.to_entity == from_entity:
                return rel
        return None
    
    def to_prompt_context(self) -> str:
        """
        Generate a context string for LLM prompts describing the schema.
        
        Returns:
            Formatted string describing all entities, columns, and rules
        """
        lines = ["DATABASE SCHEMA:"]
        lines.append("")
        
        # Entities
        for entity in self.entities.values():
            lines.append(f"Table: {entity.table}")
            if entity.description:
                lines.append(f"  Description: {entity.description}")
            if entity.aliases:
                lines.append(f"  Also known as: {', '.join(entity.aliases)}")
            lines.append("  Columns:")
            for col in entity.columns.values():
                col_info = f"    - {col.name} ({col.type})"
                if col.primary:
                    col_info += " [PRIMARY KEY]"
                if col.description:
                    col_info += f": {col.description}"
                if col.aliases:
                    col_info += f" (aliases: {', '.join(col.aliases)})"
                if col.values:
                    col_info += f" [values: {', '.join(col.values)}]"
                lines.append(col_info)
            lines.append("")
        
        # Relationships
        if self.relationships:
            lines.append("RELATIONSHIPS:")
            for rel in self.relationships:
                if rel.type == "many_to_many":
                    lines.append(
                        f"  - {rel.from_entity} <-> {rel.to_entity} "
                        f"(many-to-many via {rel.pivot_table})"
                    )
                else:
                    lines.append(
                        f"  - {rel.from_entity} -> {rel.to_entity} "
                        f"({rel.type}, FK: {rel.foreign_key})"
                    )
            lines.append("")
        
        # Business rules
        if self.business_rules:
            lines.append("BUSINESS RULES (use these terms in queries):")
            for rule in self.business_rules:
                applies = "*" if "*" in rule.applies_to else ", ".join(rule.applies_to)
                lines.append(f"  - '{rule.name}': {rule.definition}")
                if rule.description:
                    lines.append(f"    Meaning: {rule.description}")
                lines.append(f"    Applies to: {applies}")
            lines.append("")
        
        return "\n".join(lines)
    
    def validate(self) -> list[str]:
        """
        Validate the schema for consistency.

        Returns:
            List of validation error messages
        """
        errors = []

        entity_names = set(self.entities.keys())
        for rel in self.relationships:
            if rel.from_entity not in entity_names:
                errors.append(f"Relationship '{rel.name}' references unknown entity: {rel.from_entity}")
            if rel.to_entity not in entity_names:
                errors.append(f"Relationship '{rel.name}' references unknown entity: {rel.to_entity}")

        for rule in self.business_rules:
            for entity_name in rule.applies_to:
                if entity_name != "*" and entity_name not in entity_names:
                    errors.append(f"Business rule '{rule.name}' references unknown entity: {entity_name}")

        for entity in self.entities.values():
            for col in entity.columns.values():
                if col.references:
                    parts = col.references.split(".")
                    if len(parts) != 2:
                        errors.append(f"Invalid reference format in {entity.name}.{col.name}: {col.references}")
                    else:
                        ref_entity, ref_col = parts
                        if ref_entity not in entity_names:
                            errors.append(f"Reference in {entity.name}.{col.name} to unknown entity: {ref_entity}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "version": self.version,
            "database": self.database,
            "entities": {
                name: {
                    "table": entity.table,
                    "description": entity.description,
                    "aliases": entity.aliases,
                    "columns": {
                        col_name: {
                            "type": col.type,
                            "description": col.description,
                            "primary": col.primary,
                        }
                        for col_name, col in entity.columns.items()
                    }
                }
                for name, entity in self.entities.items()
            },
            "relationships": [
                {
                    "name": rel.name,
                    "type": rel.type,
                    "from": rel.from_entity,
                    "to": rel.to_entity,
                }
                for rel in self.relationships
            ],
            "security": self.security,
            "llm": self.llm,
            "cache": self.cache,
            "prompts": self.prompts,
        }
