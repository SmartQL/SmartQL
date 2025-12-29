# SmartQL YAML Schema Reference

This document defines the complete YAML schema specification for SmartQL.

## Schema Version

```yaml
version: "1.0"
```

Always specify the schema version for forward compatibility.

---

## Database Configuration

### Connection Settings

```yaml
database:
  # Required: Database type
  type: postgresql  # postgresql | mysql | sqlite | sqlserver
  
  # Connection can be a URL or individual parameters
  connection:
    # Option 1: Connection URL
    url: ${DATABASE_URL}
    
    # Option 2: Individual parameters
    host: localhost
    port: 5432
    database: myapp
    user: ${DB_USER}
    password: ${DB_PASSWORD}
    
    # Optional: SSL settings
    ssl:
      enabled: true
      ca_cert: /path/to/ca.pem
      verify: true
    
    # Optional: Connection pool
    pool:
      min_connections: 2
      max_connections: 10
      idle_timeout: 300
```

### Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
database:
  connection:
    password: ${DB_PASSWORD}
```

---

## Semantic Layer

The semantic layer maps your database schema to business concepts.

### Entities

Entities represent tables with business context:

```yaml
semantic_layer:
  entities:
    # Key is the semantic name (used in natural language)
    customers:
      # Actual table name in database
      table: users
      
      # Human-readable description (fed to LLM)
      description: "Registered customers who can make purchases"
      
      # Alternative names users might use
      aliases:
        - users
        - members
        - accounts
        - buyers
      
      # Column definitions
      columns:
        id:
          type: integer
          description: "Unique customer identifier"
          primary: true
          
        email:
          type: string
          description: "Customer's email address"
          searchable: true  # Hint for text search
          unique: true
          
        full_name:
          type: string
          description: "Customer's full name"
          aliases: ["name", "customer_name"]
          
        status:
          type: enum
          values: ["active", "inactive", "suspended", "pending"]
          description: "Account status"
          default: "pending"
          aliases: ["state", "account_status"]
          
        tier:
          type: enum
          values: ["free", "basic", "premium", "enterprise"]
          description: "Subscription tier"
          aliases: ["plan", "subscription", "level"]
          
        balance:
          type: decimal
          description: "Account balance in USD"
          aliases: ["credit", "wallet"]
          
        is_verified:
          type: boolean
          description: "Whether email is verified"
          aliases: ["verified", "email_verified"]
          
        metadata:
          type: json
          description: "Additional customer data"
          
        created_at:
          type: datetime
          description: "Registration date"
          aliases: ["registered", "signup_date", "joined", "registration_date"]
          
        updated_at:
          type: datetime
          description: "Last profile update"
```

### Column Types

| Type | Description | SQL Mapping |
|------|-------------|-------------|
| `integer` | Whole numbers | INT, BIGINT |
| `decimal` | Decimal numbers | DECIMAL, NUMERIC, FLOAT |
| `string` | Text data | VARCHAR, TEXT |
| `text` | Long text | TEXT, LONGTEXT |
| `boolean` | True/false | BOOLEAN, TINYINT(1) |
| `datetime` | Date and time | TIMESTAMP, DATETIME |
| `date` | Date only | DATE |
| `time` | Time only | TIME |
| `json` | JSON data | JSON, JSONB |
| `enum` | Enumerated values | ENUM, VARCHAR |
| `uuid` | UUID identifiers | UUID, CHAR(36) |

### Column Properties

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Column data type (required) |
| `description` | string | Human-readable description |
| `aliases` | array | Alternative names for this column |
| `primary` | boolean | Is this the primary key? |
| `unique` | boolean | Has unique constraint? |
| `nullable` | boolean | Allows NULL values? |
| `default` | any | Default value |
| `searchable` | boolean | Optimized for text search? |
| `hidden` | boolean | Exclude from query results? |
| `values` | array | Valid values (for enum type) |
| `references` | string | Foreign key reference (e.g., "users.id") |

---

## Relationships

Define how entities connect to each other:

```yaml
semantic_layer:
  relationships:
    # One-to-Many: A user has many orders
    - name: customer_orders
      type: one_to_many
      from: customers
      to: orders
      foreign_key: user_id  # Column in 'orders' table
      description: "Customers can place multiple orders"
      
    # Many-to-One: An order belongs to a user  
    - name: order_customer
      type: many_to_one
      from: orders
      to: customers
      foreign_key: user_id
      
    # One-to-One: User has one profile
    - name: customer_profile
      type: one_to_one
      from: customers
      to: profiles
      foreign_key: user_id
      
    # Many-to-Many: Products and categories
    - name: product_categories
      type: many_to_many
      from: products
      to: categories
      pivot_table: product_categories
      pivot_from: product_id
      pivot_to: category_id
      
    # Self-referential: Employee manager
    - name: employee_manager
      type: many_to_one
      from: employees
      to: employees
      foreign_key: manager_id
      description: "Employee's direct manager"
```

### Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| `one_to_one` | Single record on each side | User → Profile |
| `one_to_many` | One parent, many children | User → Orders |
| `many_to_one` | Many children, one parent | Orders → User |
| `many_to_many` | Many on both sides (via pivot) | Products ↔ Categories |

---

## Business Rules

Define named conditions that map natural language to SQL:

```yaml
semantic_layer:
  business_rules:
    # Simple condition
    - name: active
      applies_to: [customers, employees]
      definition: "status = 'active'"
      description: "Has active status"
      
    # Time-based rule
    - name: recent
      applies_to: ["*"]  # All entities with created_at
      definition: "created_at >= NOW() - INTERVAL '30 days'"
      description: "Created within the last 30 days"
      
    # Complex condition
    - name: high_value_customer
      applies_to: [customers]
      definition: |
        id IN (
          SELECT user_id FROM orders 
          GROUP BY user_id 
          HAVING SUM(total) >= 10000
        )
      description: "Customers with $10,000+ lifetime spend"
      
    # Parameterized rule
    - name: created_in_last_n_days
      applies_to: ["*"]
      definition: "created_at >= NOW() - INTERVAL '{n} days'"
      parameters:
        n:
          type: integer
          default: 30
      description: "Created within the last N days"
      
    # Compound rule
    - name: vip_customer
      applies_to: [customers]
      definition: "tier IN ('premium', 'enterprise') AND status = 'active'"
      description: "Active premium or enterprise customers"
      aliases: ["premium customer", "enterprise customer", "top customer"]
      
    # Null handling
    - name: has_phone
      applies_to: [customers]
      definition: "phone IS NOT NULL AND phone != ''"
      description: "Has a phone number on file"
      
    # Date range rules
    - name: this_month
      applies_to: ["*"]
      definition: "created_at >= DATE_TRUNC('month', CURRENT_DATE)"
      description: "Created this calendar month"
      
    - name: this_year
      applies_to: ["*"]
      definition: "created_at >= DATE_TRUNC('year', CURRENT_DATE)"
      description: "Created this calendar year"
      
    - name: last_quarter
      applies_to: ["*"]
      definition: |
        created_at >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
        AND created_at < DATE_TRUNC('quarter', CURRENT_DATE)
      description: "Created in the previous quarter"
```

### Rule Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Identifier for the rule (required) |
| `applies_to` | array | Entities this rule applies to (`["*"]` for all) |
| `definition` | string | SQL WHERE clause fragment |
| `description` | string | Human-readable description |
| `aliases` | array | Alternative names for this rule |
| `parameters` | object | Parameterized values |

---

## Aggregations

Define common aggregation patterns:

```yaml
semantic_layer:
  aggregations:
    - name: total_revenue
      entity: orders
      expression: "SUM(total)"
      description: "Sum of all order totals"
      aliases: ["revenue", "sales", "total_sales"]
      
    - name: order_count
      entity: orders
      expression: "COUNT(*)"
      description: "Number of orders"
      aliases: ["number_of_orders", "orders_count"]
      
    - name: average_order_value
      entity: orders
      expression: "AVG(total)"
      description: "Average order value"
      aliases: ["aov", "avg_order"]
      
    - name: customer_lifetime_value
      entity: customers
      expression: "(SELECT COALESCE(SUM(total), 0) FROM orders WHERE user_id = customers.id)"
      description: "Total spent by customer"
      aliases: ["ltv", "clv", "lifetime_value", "total_spent"]
```

---

## Metrics

Define calculated metrics that can be queried:

```yaml
semantic_layer:
  metrics:
    - name: conversion_rate
      description: "Percentage of users who made a purchase"
      formula: |
        CAST(COUNT(DISTINCT orders.user_id) AS FLOAT) / 
        NULLIF(COUNT(DISTINCT users.id), 0) * 100
      entities: [users, orders]
      
    - name: monthly_recurring_revenue
      description: "MRR from active subscriptions"
      formula: "SUM(subscriptions.monthly_amount)"
      filters: ["subscriptions.status = 'active'"]
      entities: [subscriptions]
      aliases: ["mrr"]
```

---

## Security Configuration

```yaml
security:
  # Query mode
  mode: read_only  # read_only | read_write
  
  # Table allowlist (if specified, only these tables can be queried)
  allowed_tables:
    - users
    - orders
    - products
    - categories
    
  # Table blocklist (these tables can never be queried)
  blocked_tables:
    - admin_logs
    - system_config
    
  # Columns that are never included in results
  blocked_columns:
    - users.password_hash
    - users.password_salt
    - users.api_token
    - users.ssn
    - payments.card_number
    - payments.cvv
    
  # Columns that can be used in WHERE but not SELECT
  filter_only_columns:
    - users.tenant_id
    
  # Required filters (for multi-tenant apps)
  required_filters:
    orders:
      column: tenant_id
      description: "All order queries must filter by tenant"
    users:
      column: tenant_id
      
  # Row limits
  max_rows: 1000
  default_limit: 100
  
  # Query timeout
  timeout_seconds: 30
  
  # Prevent expensive operations
  block_operations:
    - CROSS JOIN
    - cartesian products
    
  # Maximum JOIN depth
  max_join_depth: 4
  
  # Rate limiting
  rate_limit:
    requests_per_minute: 60
    requests_per_hour: 500
```

---

## LLM Configuration

```yaml
llm:
  # Provider selection
  provider: openai  # openai | anthropic | google | ollama | azure | custom
  
  # OpenAI configuration
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
    temperature: 0
    max_tokens: 2000
    organization: ${OPENAI_ORG_ID}  # Optional
    
  # Anthropic Claude configuration
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-20250514
    max_tokens: 2000
    
  # Google Gemini configuration
  google:
    api_key: ${GOOGLE_API_KEY}
    model: gemini-pro
    
  # Azure OpenAI configuration
  azure:
    api_key: ${AZURE_OPENAI_KEY}
    endpoint: https://your-resource.openai.azure.com
    deployment: your-deployment-name
    api_version: "2024-02-01"
    
  # Ollama (local) configuration
  ollama:
    base_url: http://localhost:11434
    model: llama3
    # Or use codellama for better SQL generation
    # model: codellama:13b
    
  # Custom endpoint (OpenAI-compatible API)
  custom:
    endpoint: https://your-api.com/v1/chat/completions
    api_key: ${CUSTOM_API_KEY}
    model: your-model-name
    headers:
      X-Custom-Header: value
```

---

## Caching

```yaml
cache:
  enabled: true
  
  # Cache backend
  backend: redis  # memory | redis | file | memcached
  
  # Time-to-live for cached queries
  ttl_seconds: 3600
  
  # Cache schema introspection results
  cache_schema: true
  schema_ttl_seconds: 86400
  
  # Redis configuration
  redis:
    url: ${REDIS_URL}
    # Or individual params
    host: localhost
    port: 6379
    db: 0
    password: ${REDIS_PASSWORD}
    
  # File cache configuration
  file:
    directory: /tmp/smartql_cache
    
  # Cache key prefix
  key_prefix: "qw:"
```

---

## Logging & Debugging

```yaml
logging:
  # Log level
  level: info  # debug | info | warning | error
  
  # Log all generated queries
  log_queries: true
  
  # Log LLM prompts and responses
  log_llm: false
  
  # Log query execution time
  log_timing: true
  
  # Output destination
  output: file  # console | file | both
  
  # Log file path
  file: /var/log/smartql/queries.log
  
  # Log format
  format: json  # json | text
```

---

## Custom Prompts

Customize the prompts sent to the LLM:

```yaml
prompts:
  # System prompt (sets context for the LLM)
  system: |
    You are a SQL expert assistant. Your job is to convert natural language 
    questions into valid SQL queries.
    
    Rules:
    - Only generate SELECT queries (no INSERT, UPDATE, DELETE)
    - Use proper JOINs, never cartesian products
    - Always use table aliases for clarity
    - Include LIMIT clause when appropriate
    - Use parameterized values for user input
    
  # Few-shot examples (helps LLM understand your domain)
  examples:
    - question: "active customers"
      sql: "SELECT * FROM users WHERE status = 'active'"
      
    - question: "orders this month"
      sql: |
        SELECT * FROM orders 
        WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
        
    - question: "top 10 customers by revenue"
      sql: |
        SELECT u.*, SUM(o.total) as total_revenue
        FROM users u
        INNER JOIN orders o ON o.user_id = u.id
        GROUP BY u.id
        ORDER BY total_revenue DESC
        LIMIT 10
        
    - question: "products with no orders"
      sql: |
        SELECT p.* FROM products p
        LEFT JOIN order_items oi ON oi.product_id = p.id
        WHERE oi.id IS NULL
        
  # Additional context (appended to every prompt)
  context: |
    Important notes about this database:
    - All monetary values are stored in cents (divide by 100 for dollars)
    - Dates are stored in UTC timezone
    - The 'deleted_at' column indicates soft-deleted records
```

---

## Validation Rules

```yaml
validation:
  # Require certain clauses
  require_where: false  # Require WHERE clause on all queries
  require_limit: true   # Require LIMIT clause
  
  # SQL syntax validation
  validate_syntax: true
  
  # Explain query before execution (EXPLAIN ANALYZE)
  explain_queries: false
  
  # Reject queries with these patterns
  blocked_patterns:
    - "DROP"
    - "TRUNCATE"
    - "DELETE FROM"
    - "UPDATE.*SET"
    - "INSERT INTO"
    - "ALTER TABLE"
    - "CREATE TABLE"
    
  # Maximum query complexity score
  max_complexity: 100
```

---

## Full Example

```yaml
version: "1.0"

database:
  type: postgresql
  connection:
    url: ${DATABASE_URL}

semantic_layer:
  entities:
    customers:
      table: users
      description: "Registered customers"
      aliases: [users, members]
      columns:
        id: { type: integer, primary: true }
        email: { type: string, searchable: true }
        name: { type: string }
        status: { type: enum, values: [active, inactive, suspended] }
        tier: { type: enum, values: [free, basic, premium] }
        created_at: { type: datetime, aliases: [registered, joined] }
        
    orders:
      table: orders
      description: "Customer orders"
      aliases: [purchases, transactions]
      columns:
        id: { type: integer, primary: true }
        user_id: { type: integer, references: users.id }
        total: { type: decimal, aliases: [amount, revenue] }
        status: { type: enum, values: [pending, paid, shipped, delivered] }
        created_at: { type: datetime, aliases: [order_date] }
        
  relationships:
    - { name: customer_orders, type: one_to_many, from: customers, to: orders, foreign_key: user_id }
    
  business_rules:
    - { name: active, definition: "status = 'active'", applies_to: [customers] }
    - { name: premium, definition: "tier = 'premium'", applies_to: [customers] }
    - { name: recent, definition: "created_at >= NOW() - INTERVAL '30 days'" }
    - { name: high_value, definition: "total >= 1000", applies_to: [orders] }

security:
  mode: read_only
  allowed_tables: [users, orders]
  blocked_columns: [users.password_hash]
  max_rows: 1000

llm:
  provider: openai
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
    temperature: 0

cache:
  enabled: true
  backend: memory
  ttl_seconds: 3600

logging:
  level: info
  log_queries: true
```

---

## Environment Variables

All configuration values support environment variable interpolation using `${VAR_NAME}` syntax.

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379/0
```

SmartQL automatically loads `.env` files from the current directory.
