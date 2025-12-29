# SmartQL

**Natural Language to SQL — Database First.**

SmartQL is a Python library that converts natural language questions into SQL queries using a YAML-based semantic layer. It connects directly to databases, understands your schema, and generates safe, optimized queries.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Natural Language Query                      │
│              "Show me all users who spent over $1000"           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SmartQL Core                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   LiteLLM   │  │   sqlglot   │  │    Semantic Layer       │  │
│  │  (100+ LLMs)│  │ (SQL Parser)│  │  (YAML Config)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Generated SQL                            │
│     SELECT u.* FROM users u JOIN orders o ON o.user_id = u.id   │
│     WHERE o.total >= 1000 GROUP BY u.id                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Database                                │
│              MySQL, PostgreSQL, SQLite, SQL Server              │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **100+ LLM Providers** — OpenAI, Claude, Gemini, Groq, Ollama, and more via LiteLLM
- **SQL Security** — AST-based validation with sqlglot (not regex)
- **Semantic Layer** — YAML config with business context, aliases, and rules
- **Auto-Introspection** — Discover schema from existing databases
- **Interactive CLI** — Shell mode for testing queries
- **REST API** — Optional HTTP server for language-agnostic access

## Installation

```bash
pip install smartql
```

Or with [Rye](https://rye-up.com):
```bash
rye add smartql
```

## Quick Start

### 1. Create a configuration file (`smartql.yml`)

```yaml
version: "1.0"

database:
  type: mysql
  connection:
    host: localhost
    database: myapp
    user: ${DB_USER}
    password: ${DB_PASSWORD}

llm:
  provider: groq
  groq:
    api_key: ${GROQ_API_KEY}
    model: llama-3.1-8b-instant

security:
  mode: read_only
  max_rows: 1000
```

### 2. Use the CLI

```bash
# Interactive shell
smartql shell -c smartql.yml

# Single query
smartql ask "How many users signed up this month?" -c smartql.yml

# With execution
smartql ask "Show me the top 5 customers" -c smartql.yml --execute
```

### 3. Use in Python

```python
from smartql import SmartQL

sql = SmartQL.from_yaml("smartql.yml")

# Generate SQL
result = sql.ask("Show me all active users with orders")
print(result.sql)
print(result.explanation)

# Generate and execute
result = sql.ask("Top 10 customers by revenue", execute=True)
for row in result.rows:
    print(row)
```

## Semantic Layer

The semantic layer adds business context to your database schema:

```yaml
semantic_layer:
  entities:
    customers:
      table: users
      description: "Registered customers"
      aliases: ["users", "members"]
      columns:
        id:
          type: integer
          primary: true
        email:
          type: string
          description: "Customer email"
        status:
          type: enum
          values: ["active", "inactive", "suspended"]

  relationships:
    - name: customer_orders
      type: one_to_many
      from: customers
      to: orders
      foreign_key: user_id

  business_rules:
    - name: active
      applies_to: [customers]
      definition: "status = 'active'"
      description: "Active customers"

    - name: high_value
      applies_to: [orders]
      definition: "total >= 1000"
      description: "Orders over $1000"
```

Without a semantic layer, SmartQL auto-introspects your database schema.

## Security

SmartQL uses sqlglot for AST-based SQL parsing and validation:

```yaml
security:
  mode: read_only           # Only SELECT queries allowed

  allowed_tables:           # Whitelist tables
    - users
    - orders
    - products

  blocked_columns:          # Hide sensitive data
    - users.password
    - users.api_token

  max_rows: 1000           # Prevent large result sets
  timeout_seconds: 30      # Query timeout
  max_join_depth: 4        # Limit JOIN complexity
```

## LLM Providers

SmartQL uses LiteLLM for unified access to 100+ models:

```yaml
# OpenAI
llm:
  provider: openai
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o

# Anthropic Claude
llm:
  provider: anthropic
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514

# Google Gemini
llm:
  provider: google
  google:
    api_key: ${GEMINI_API_KEY}
    model: gemini-2.0-flash

# Groq (fast & free)
llm:
  provider: groq
  groq:
    api_key: ${GROQ_API_KEY}
    model: llama-3.1-8b-instant

# Local with Ollama
llm:
  provider: ollama
  ollama:
    model: llama3:8b
    api_base: http://localhost:11434
```

## CLI Commands

```bash
# Interactive shell
smartql shell -c config.yml

# Ask a question
smartql ask "your question" -c config.yml

# Ask and execute
smartql ask "your question" -c config.yml --execute

# Validate SQL
smartql check "SELECT * FROM users" -c config.yml

# Validate config
smartql validate -c config.yml

# Introspect database
smartql introspect -c "mysql://user:pass@localhost/db" -o schema.yml

# Start HTTP API server
smartql serve -c config.yml --port 8000
```

## Shell Commands

In interactive shell mode:

```
/help     - Show help
/schema   - Display database schema
/execute  - Toggle auto-execute mode
/quit     - Exit shell
```

## Examples

See the [examples/](examples/) directory:

- `ecommerce/` — E-commerce database with products, orders, customers
- `trakli/` — Personal finance tracker with wallets and transactions

## License

MIT License
