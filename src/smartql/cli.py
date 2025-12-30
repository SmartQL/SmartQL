"""
Command-line interface for SmartQL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from smartql import SmartQL
from smartql.exceptions import SmartQLError
from smartql.result import FormatType, detect_format_type

console = Console()


def display_results(rows: list[dict], format_type: FormatType | None = None) -> None:
    """Display query results based on format type."""
    if not rows:
        console.print("[dim]No results[/dim]")
        return

    if format_type is None:
        format_type = detect_format_type(rows)

    if format_type == FormatType.SCALAR:
        value = list(rows[0].values())[0]
        if isinstance(value, (int, float)):
            value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        console.print(f"[bold green]{value}[/bold green]")

    elif format_type == FormatType.PAIR:
        keys = list(rows[0].keys())
        label, value = rows[0][keys[0]], rows[0][keys[1]]
        if isinstance(value, (int, float)):
            value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        console.print(f"[bold]{label}:[/bold] [green]{value}[/green]")

    elif format_type == FormatType.RECORD:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Field", style="bold")
        table.add_column("Value", style="green")
        for key, value in rows[0].items():
            if isinstance(value, float):
                value = f"{value:,.2f}"
            table.add_row(str(key), str(value) if value is not None else "-")
        console.print(Panel(table, title="Record", border_style="blue"))

    elif format_type == FormatType.LIST:
        key = list(rows[0].keys())[0]
        for row in rows[:20]:
            console.print(f"  • {row[key]}")
        if len(rows) > 20:
            console.print(f"  [dim]... and {len(rows) - 20} more[/dim]")

    elif format_type == FormatType.PAIR_LIST:
        keys = list(rows[0].keys())
        for row in rows[:20]:
            label, value = row[keys[0]], row[keys[1]]
            if isinstance(value, float):
                value = f"{value:,.2f}"
            console.print(f"  [bold]{label}:[/bold] {value}")
        if len(rows) > 20:
            console.print(f"  [dim]... and {len(rows) - 20} more[/dim]")

    elif format_type == FormatType.TABLE:
        table = Table()
        for key in rows[0].keys():
            table.add_column(str(key))
        for row in rows[:20]:
            values = []
            for v in row.values():
                if isinstance(v, float):
                    values.append(f"{v:,.2f}")
                else:
                    values.append(str(v) if v is not None else "-")
            table.add_row(*values)
        console.print(table)
        if len(rows) > 20:
            console.print(f"[dim]... and {len(rows) - 20} more rows[/dim]")

    elif format_type == FormatType.RAW:
        import json

        console.print(json.dumps(rows[:20], indent=2, default=str))
        if len(rows) > 20:
            console.print(f"[dim]... and {len(rows) - 20} more rows[/dim]")


@click.group()
@click.version_option(version="0.1.0", prog_name="smartql")
def main():
    """SmartQL - Natural Language to SQL, Database First."""
    pass


@main.command()
@click.argument("question")
@click.option(
    "-c",
    "--config",
    default="smartql.yml",
    help="Path to configuration file",
)
@click.option(
    "-e",
    "--execute",
    is_flag=True,
    help="Execute the query and show results",
)
@click.option(
    "--env-file",
    default=None,
    help="Path to .env file",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
def ask(
    question: str,
    config: str,
    execute: bool,
    env_file: str | None,
    output_json: bool,
):
    """
    Convert a natural language question to SQL.

    Example:
        smartql ask "Show me all active users with orders"
    """
    try:
        qw = SmartQL.from_yaml(config, env_file=env_file)
        result = qw.ask(question, execute=execute)

        if output_json:
            console.print(result.to_json())
            return

        # Display SQL
        console.print(
            Panel(
                Syntax(result.sql, "sql", theme="monokai", line_numbers=True),
                title="Generated SQL",
                border_style="green",
            )
        )

        # Display explanation
        if result.explanation:
            console.print(f"\n[dim]Explanation:[/dim] {result.explanation}")

        # Display confidence
        if result.confidence:
            if result.confidence > 0.8:
                confidence_color = "green"
            elif result.confidence > 0.5:
                confidence_color = "yellow"
            else:
                confidence_color = "red"
            console.print(f"[dim]Confidence:[/dim] [{confidence_color}]{result.confidence:.0%}[/]")

        # Display results if executed
        if execute and result.rows is not None:
            result.compute_format_type()
            console.print(f"\n[dim]Results ({result.row_count} rows) - {result.format_type}:[/dim]")
            display_results(result.rows)

        if result.execution_error:
            console.print(f"\n[red]Execution Error:[/red] {result.execution_error}")

    except SmartQLError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found:[/red] {config}")
        sys.exit(1)


@main.command()
@click.option(
    "--connection",
    "-c",
    required=True,
    help="Database connection URL",
)
@click.option(
    "--output",
    "-o",
    default="smartql.yml",
    help="Output file path",
)
def introspect(connection: str, output: str):
    """
    Introspect a database and generate a YAML configuration file.

    Example:
        smartql introspect -c "postgresql://user:pass@localhost/db" -o schema.yml
    """
    try:
        from smartql.database import SQLAlchemyConnector

        connector = SQLAlchemyConnector({"connection": {"url": connection}})

        if not connector.test_connection():
            console.print("[red]Failed to connect to database[/red]")
            sys.exit(1)

        console.print("[dim]Connected to database. Introspecting schema...[/dim]")

        schema_info = connector.introspect()

        from smartql.core import SmartQL
        from smartql.schema import Schema

        qw = SmartQL(
            schema=Schema(),
            database=connector,
        )

        yaml_content = qw._generate_yaml_from_schema(schema_info)

        Path(output).write_text(yaml_content)
        console.print(f"[green]Generated configuration file:[/green] {output}")

        # Show summary
        tables = schema_info.get("tables", {})
        console.print(f"\n[dim]Found {len(tables)} tables:[/dim]")
        for table_name in tables.keys():
            console.print(f"  - {table_name}")

        console.print(
            f"\n[yellow]Remember to edit {output} to add descriptions and business rules![/yellow]"
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "-c",
    "--config",
    default="smartql.yml",
    help="Path to configuration file",
)
def validate(config: str):
    """
    Validate a SmartQL configuration file.

    Example:
        smartql validate -c smartql.yml
    """
    try:
        from smartql.schema import Schema

        schema = Schema.from_yaml(config)
        errors = schema.validate()

        if errors:
            console.print("[red]Validation errors found:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            sys.exit(1)
        else:
            console.print("[green]Configuration is valid![/green]")
            console.print(f"\n[dim]Entities:[/dim] {len(schema.entities)}")
            console.print(f"[dim]Relationships:[/dim] {len(schema.relationships)}")
            console.print(f"[dim]Business Rules:[/dim] {len(schema.business_rules)}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "-c",
    "--config",
    default="smartql.yml",
    help="Path to configuration file",
)
@click.option(
    "--env-file",
    default=None,
    help="Path to .env file",
)
def shell(config: str, env_file: str | None):
    """
    Start an interactive query shell.

    Example:
        smartql shell
    """
    try:
        qw = SmartQL.from_yaml(config, env_file=env_file)

        def print_banner():
            console.clear()
            console.print(
                Panel(
                    "[bold]SmartQL Interactive Shell[/bold]\n\n"
                    "Enter natural language questions to generate SQL.\n"
                    "Commands: [dim]/help, /schema, /execute, /clear, /quit[/dim]",
                    border_style="blue",
                )
            )

        print_banner()
        execute_mode = False
        history = InMemoryHistory()

        while True:
            try:
                question = prompt("\n> ", history=history)

                if not question.strip():
                    continue

                # Handle commands
                if question.startswith("/"):
                    cmd = question.lower().strip()

                    if cmd in ("/quit", "/exit", "/q"):
                        console.print("[dim]Goodbye![/dim]")
                        break

                    elif cmd == "/help":
                        console.print("""
[bold]Commands:[/bold]
  /help     - Show this help
  /schema   - Show the database schema
  /execute  - Toggle auto-execute mode
  /clear    - Clear the screen
  /quit     - Exit the shell
                        """)

                    elif cmd == "/clear":
                        print_banner()

                    elif cmd == "/schema":
                        console.print(qw.generator._get_schema_context())

                    elif cmd == "/execute":
                        execute_mode = not execute_mode
                        status = "ON" if execute_mode else "OFF"
                        console.print(f"[dim]Auto-execute mode: {status}[/dim]")

                    else:
                        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")

                    continue

                # Generate SQL
                result = qw.ask(question, execute=execute_mode)

                console.print(Syntax(result.sql, "sql", theme="monokai"))

                if result.explanation:
                    console.print(f"[dim]{result.explanation}[/dim]")

                if execute_mode and result.rows is not None:
                    result.compute_format_type()
                    fmt = result.format_type
                    console.print(f"\n[dim]Results ({result.row_count} rows) - {fmt}:[/dim]")
                    display_results(result.rows)

            except KeyboardInterrupt:
                console.print("\n[dim]Use /quit to exit[/dim]")
            except SmartQLError as e:
                console.print(f"[red]Error:[/red] {e}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("sql")
@click.option(
    "-c",
    "--config",
    default="smartql.yml",
    help="Path to configuration file",
)
def check(sql: str, config: str):
    """
    Check if a SQL query is valid according to security rules.

    Example:
        smartql check "SELECT * FROM users"
    """
    try:
        qw = SmartQL.from_yaml(config)
        errors = qw.validate(sql)

        if errors:
            console.print("[red]Validation failed:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            sys.exit(1)
        else:
            console.print("[green]Query is valid![/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "-c",
    "--config",
    default=None,
    help="Path to configuration file to load on startup",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to listen on",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for authentication",
)
def serve(
    config: str | None,
    host: str,
    port: int,
    reload: bool,
    api_key: str | None,
):
    """
    Start the SmartQL HTTP API server.

    Example:
        smartql serve --config schema.yml --port 8000
    """
    import os

    try:
        import uvicorn
    except ImportError:
        console.print("[red]Server dependencies not installed.[/red]")
        console.print("Install with: pip install smartql[server]")
        sys.exit(1)

    if config:
        os.environ["SMARTQL_CONFIG"] = config
    if api_key:
        os.environ["SMARTQL_API_KEY"] = api_key

    console.print(
        Panel(
            f"[bold]SmartQL API Server[/bold]\n\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Config: {config or 'None (upload via API)'}\n"
            f"Reload: {'Enabled' if reload else 'Disabled'}",
            border_style="green",
        )
    )

    console.print(f"\n[dim]API docs available at: http://{host}:{port}/docs[/dim]")
    console.print(f"[dim]Health check: http://{host}:{port}/health[/dim]\n")

    try:
        uvicorn.run(
            "smartql.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")


if __name__ == "__main__":
    main()
