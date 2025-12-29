"""
SmartQL HTTP API Server

FastAPI-based HTTP server that exposes SmartQL functionality as REST endpoints.
This enables language-agnostic access (PHP, Node.js, Ruby, etc.) to the SmartQL engine.

Run with:
    uvicorn smartql.server:app --reload
    # or
    smartql serve --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import yaml
import hashlib
import time
from pathlib import Path

from .core import SmartQL
from .exceptions import SmartQLError, SecurityError, LLMError, SchemaError


# =============================================================================
# Pydantic Models
# =============================================================================

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str = Field(..., description="Natural language question")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    execute: bool = Field(False, description="Whether to execute the query")
    explain: bool = Field(False, description="Include query explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Show me all active users with orders this month",
                "context": {"tenant_id": 123},
                "execute": False,
                "explain": True
            }
        }


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    sql: str
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    validation_errors: List[str] = []
    cached: bool = False


class ValidateRequest(BaseModel):
    """Request model for the /validate endpoint."""
    sql: str = Field(..., description="SQL query to validate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sql": "SELECT * FROM users WHERE status = 'active'"
            }
        }


class ValidateResponse(BaseModel):
    """Response model for the /validate endpoint."""
    valid: bool
    errors: List[str] = []


class SchemaUploadRequest(BaseModel):
    """Request model for uploading a YAML schema."""
    yaml_content: str = Field(..., description="YAML schema content")
    schema_id: Optional[str] = Field(None, description="Custom schema ID")


class SchemaResponse(BaseModel):
    """Response model for schema operations."""
    schema_id: str
    entities: List[str]
    relationships_count: int
    business_rules_count: int


class IntrospectRequest(BaseModel):
    """Request model for database introspection."""
    connection_url: str = Field(..., description="Database connection URL")
    include_views: bool = Field(False, description="Include database views")
    
    class Config:
        json_schema_extra = {
            "example": {
                "connection_url": "postgresql://user:pass@localhost/mydb",
                "include_views": False
            }
        }


class IntrospectResponse(BaseModel):
    """Response model for introspection."""
    yaml_content: str
    tables: List[str]
    columns_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    schemas_loaded: int
    uptime_seconds: float


# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="SmartQL API",
    description="Natural language to SQL translation engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SMARTQL_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_start_time = time.time()
_schemas: Dict[str, SmartQL] = {}
_default_schema_id: Optional[str] = None


# =============================================================================
# Startup & Dependencies
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load default schema on startup if configured."""
    global _default_schema_id

    default_config = os.getenv("SMARTQL_CONFIG")
    if default_config and Path(default_config).exists():
        try:
            qw = SmartQL.from_yaml(default_config)
            schema_id = "default"
            _schemas[schema_id] = qw
            _default_schema_id = schema_id
            print(f"✓ Loaded default schema from {default_config}")
        except Exception as e:
            print(f"⚠ Failed to load default schema: {e}")


def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Extract API key from header."""
    required_key = os.getenv("SMARTQL_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


def get_schema_id(x_schema_id: Optional[str] = Header(None)) -> str:
    """Get schema ID from header or use default."""
    schema_id = x_schema_id or _default_schema_id
    if not schema_id:
        raise HTTPException(
            status_code=400, 
            detail="No schema loaded. Upload a schema first or set X-Schema-ID header."
        )
    if schema_id not in _schemas:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    return schema_id


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check and server info."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        schemas_loaded=len(_schemas),
        uptime_seconds=time.time() - _start_time
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(get_api_key)])
async def ask(
    request: AskRequest,
    schema_id: str = Depends(get_schema_id)
):
    """
    Convert natural language question to SQL query.
    
    Optionally execute the query and return results.
    """
    qw = _schemas[schema_id]
    start_time = time.time()
    
    try:
        result = qw.ask(
            question=request.question,
            context=request.context,
            execute=request.execute
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return AskResponse(
            sql=result.sql,
            explanation=result.explanation if request.explain else None,
            confidence=result.confidence,
            rows=result.rows if request.execute else None,
            row_count=len(result.rows) if result.rows else None,
            execution_time_ms=execution_time,
            validation_errors=result.validation_errors,
            cached=result.cached if hasattr(result, 'cached') else False
        )
        
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")
    except SmartQLError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate", response_model=ValidateResponse, dependencies=[Depends(get_api_key)])
async def validate(
    request: ValidateRequest,
    schema_id: str = Depends(get_schema_id)
):
    """
    Validate a SQL query against security rules.
    """
    qw = _schemas[schema_id]
    
    try:
        errors = qw.validate(request.sql)
        return ValidateResponse(
            valid=len(errors) == 0,
            errors=errors
        )
    except SmartQLError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/schema", response_model=SchemaResponse, dependencies=[Depends(get_api_key)])
async def upload_schema(request: SchemaUploadRequest):
    """
    Upload a YAML schema configuration.
    
    Returns a schema ID that can be used in subsequent requests.
    """
    try:
        # Parse YAML to validate it
        config = yaml.safe_load(request.yaml_content)
        
        # Generate schema ID
        if request.schema_id:
            schema_id = request.schema_id
        else:
            schema_id = hashlib.sha256(request.yaml_content.encode()).hexdigest()[:12]
        
        # Create SmartQL instance
        qw = SmartQL.from_dict(config)
        _schemas[schema_id] = qw
        
        # Set as default if first schema
        global _default_schema_id
        if _default_schema_id is None:
            _default_schema_id = schema_id
        
        # Gather stats
        entities = list(qw.schema.entities.keys()) if qw.schema else []
        relationships = len(qw.schema.relationships) if qw.schema else 0
        rules = len(qw.schema.business_rules) if qw.schema else 0
        
        return SchemaResponse(
            schema_id=schema_id,
            entities=entities,
            relationships_count=relationships,
            business_rules_count=rules
        )
        
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=f"Schema error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema/{schema_id}", response_model=SchemaResponse, dependencies=[Depends(get_api_key)])
async def get_schema(schema_id: str):
    """Get information about a loaded schema."""
    if schema_id not in _schemas:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    
    qw = _schemas[schema_id]
    entities = list(qw.schema.entities.keys()) if qw.schema else []
    relationships = len(qw.schema.relationships) if qw.schema else 0
    rules = len(qw.schema.business_rules) if qw.schema else 0
    
    return SchemaResponse(
        schema_id=schema_id,
        entities=entities,
        relationships_count=relationships,
        business_rules_count=rules
    )


@app.delete("/schema/{schema_id}", dependencies=[Depends(get_api_key)])
async def delete_schema(schema_id: str):
    """Remove a loaded schema."""
    global _default_schema_id
    
    if schema_id not in _schemas:
        raise HTTPException(status_code=404, detail=f"Schema '{schema_id}' not found")
    
    del _schemas[schema_id]
    
    if _default_schema_id == schema_id:
        _default_schema_id = next(iter(_schemas.keys()), None)
    
    return {"message": f"Schema '{schema_id}' deleted"}


@app.get("/schemas", dependencies=[Depends(get_api_key)])
async def list_schemas():
    """List all loaded schemas."""
    schemas = []
    for schema_id, qw in _schemas.items():
        entities = list(qw.schema.entities.keys()) if qw.schema else []
        schemas.append({
            "schema_id": schema_id,
            "entities": entities,
            "is_default": schema_id == _default_schema_id
        })
    return {"schemas": schemas, "count": len(schemas)}


@app.post("/introspect", response_model=IntrospectResponse, dependencies=[Depends(get_api_key)])
async def introspect(request: IntrospectRequest):
    """
    Introspect a database and generate YAML schema.

    Connects to the database, reads table/column metadata,
    and generates a SmartQL YAML configuration.
    """
    try:
        from .database import create_connector
        
        connector = create_connector(request.connection_url)
        
        # Get table info
        tables_info = connector.introspect()
        
        # Generate YAML
        yaml_content = _generate_yaml_from_introspection(
            tables_info, 
            request.connection_url,
            request.include_views
        )
        
        tables = list(tables_info.keys())
        columns_count = sum(len(cols) for cols in tables_info.values())
        
        return IntrospectResponse(
            yaml_content=yaml_content,
            tables=tables,
            columns_count=columns_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_yaml_from_introspection(
    tables_info: Dict[str, List[Dict]], 
    connection_url: str,
    include_views: bool = False
) -> str:
    """Generate YAML schema from introspection data."""
    entities = {}
    
    for table_name, columns in tables_info.items():
        entity = {
            "table": table_name,
            "description": f"Auto-generated entity for {table_name}",
            "columns": {}
        }
        
        for col in columns:
            col_config = {
                "type": col.get("type", "string"),
                "description": col.get("comment") or f"Column {col['name']}"
            }
            
            if col.get("primary"):
                col_config["primary"] = True
            if col.get("nullable") == False:
                col_config["nullable"] = False
                
            entity["columns"][col["name"]] = col_config
        
        entities[table_name] = entity
    
    config = {
        "version": "1.0",
        "database": {
            "type": _infer_db_type(connection_url),
            "connection": {
                "url": "${DATABASE_URL}"
            }
        },
        "semantic_layer": {
            "entities": entities
        },
        "security": {
            "mode": "read_only",
            "max_rows": 1000
        },
        "llm": {
            "provider": "openai",
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "gpt-4o"
            }
        }
    }
    
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def _infer_db_type(url: str) -> str:
    """Infer database type from connection URL."""
    if url.startswith("postgresql"):
        return "postgresql"
    elif url.startswith("mysql"):
        return "mysql"
    elif url.startswith("sqlite"):
        return "sqlite"
    elif "sqlserver" in url or "mssql" in url:
        return "sqlserver"
    return "postgresql"


# =============================================================================
# CLI Integration
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    config: Optional[str] = None
):
    """Run the SmartQL API server."""
    import uvicorn

    if config:
        os.environ["SMARTQL_CONFIG"] = config

    uvicorn.run(
        "smartql.server:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
