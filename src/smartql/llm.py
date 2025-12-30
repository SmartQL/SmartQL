"""
LLM provider using LiteLLM for unified access to 100+ models.
Supports streaming, structured outputs, retries, and fallbacks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion, completion
from pydantic import BaseModel

from smartql.exceptions import LLMError


class SQLResponse(BaseModel):
    """Structured output schema for SQL generation."""

    sql: str
    explanation: str
    confidence: float
    tables_used: list[str] = []
    reasoning: str | None = None


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: float = 120.0
    num_retries: int = 3
    fallback_models: list[str] = field(default_factory=list)
    api_key: str | None = None
    api_base: str | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> LLMConfig:
        """Create config from dictionary."""
        provider = config.get("provider", "openai")
        provider_config = config.get(provider, {})

        model = provider_config.get("model") or cls._default_model(provider)
        model_name = cls._normalize_model_name(provider, model)

        return cls(
            model=model_name,
            temperature=provider_config.get("temperature", 0.0),
            max_tokens=provider_config.get("max_tokens", 2000),
            timeout=provider_config.get("timeout", 120.0),
            num_retries=config.get("retries", 3),
            fallback_models=config.get("fallback_models", []),
            api_key=provider_config.get("api_key"),
            api_base=provider_config.get("api_base") or provider_config.get("base_url"),
        )

    @staticmethod
    def _default_model(provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-pro",
            "ollama": "llama3",
            "azure": "gpt-4o",
            "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0",
            "vertex_ai": "gemini-pro",
        }
        return defaults.get(provider, "gpt-4o")

    @staticmethod
    def _normalize_model_name(provider: str, model: str) -> str:
        """Normalize model name for LiteLLM format."""
        if "/" in model or provider == "openai":
            return model

        prefixes = {
            "anthropic": "anthropic/",
            "claude": "anthropic/",
            "google": "gemini/",
            "gemini": "gemini/",
            "ollama": "ollama/",
            "azure": "azure/",
            "bedrock": "bedrock/",
            "vertex_ai": "vertex_ai/",
            "groq": "groq/",
            "together": "together_ai/",
            "mistral": "mistral/",
            "cohere": "cohere/",
        }

        prefix = prefixes.get(provider, "")
        return f"{prefix}{model}" if prefix else model


class LLMProvider:
    """
    Unified LLM provider using LiteLLM.

    Supports 100+ models with a single interface, including:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google (Gemini)
    - Azure OpenAI
    - AWS Bedrock
    - Ollama (local)
    - Groq, Together, Mistral, Cohere, and more
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._schema_context_cache: dict[str, str] = {}

        if config.api_key:
            self._set_api_key(config.model, config.api_key)

        litellm.set_verbose = False

    def _set_api_key(self, model: str, api_key: str) -> None:
        """Set API key based on model provider."""
        import os

        if "anthropic" in model.lower():
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif "gemini" in model.lower() or "google" in model.lower():
            os.environ["GEMINI_API_KEY"] = api_key
        elif "azure" in model.lower():
            os.environ["AZURE_API_KEY"] = api_key
        elif "groq" in model.lower():
            os.environ["GROQ_API_KEY"] = api_key
        elif "ollama" not in model.lower():
            os.environ["OPENAI_API_KEY"] = api_key

    def _get_completion_kwargs(self) -> dict[str, Any]:
        """Get extra kwargs for completion calls (e.g., api_base for Ollama)."""
        kwargs = {}
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        return kwargs

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion synchronously."""
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = completion(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout,
                num_retries=self.config.num_retries,
                **self._get_completion_kwargs(),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if self.config.fallback_models:
                return self._try_fallbacks(messages, temperature, max_tokens, e)
            raise LLMError(f"LLM generation failed: {e}")

    async def agenerate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion asynchronously."""
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = await acompletion(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout,
                num_retries=self.config.num_retries,
                **self._get_completion_kwargs(),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"LLM generation failed: {e}")

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream a completion synchronously."""
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = completion(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **self._get_completion_kwargs(),
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {e}")

    async def astream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion asynchronously."""
        messages = self._build_messages(prompt, system_prompt)

        try:
            response = await acompletion(
                model=self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **self._get_completion_kwargs(),
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"LLM streaming failed: {e}")

    def generate_sql(
        self,
        question: str,
        schema_context: str,
        examples: list[dict] | None = None,
        custom_system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate SQL from natural language with structured output.
        Uses chain-of-thought reasoning for better accuracy.
        """
        system_prompt = custom_system_prompt or self._sql_system_prompt()
        prompt = self._build_sql_prompt(question, schema_context, examples)

        try:
            if self._supports_structured_output():
                return self._generate_structured_sql(prompt, system_prompt)
            else:
                return self._generate_json_sql(prompt, system_prompt)
        except Exception as e:
            raise LLMError(f"SQL generation failed: {e}")

    def generate_sql_with_consistency(
        self,
        question: str,
        schema_context: str,
        examples: list[dict] | None = None,
        num_samples: int = 3,
    ) -> dict[str, Any]:
        """
        Generate SQL with self-consistency.
        Generates multiple samples and picks the most common/best one.
        """
        results = []
        for _ in range(num_samples):
            result = self.generate_sql(question, schema_context, examples)
            results.append(result)

        return self._select_best_result(results)

    def _supports_structured_output(self) -> bool:
        """Check if model supports structured JSON output."""
        structured_models = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
        return any(m in self.config.model.lower() for m in structured_models)

    def _generate_structured_sql(
        self,
        prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Generate SQL using structured output (JSON mode)."""
        messages = self._build_messages(prompt, system_prompt)

        response = completion(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            timeout=self.config.timeout,
            num_retries=self.config.num_retries,
            **self._get_completion_kwargs(),
        )

        content = response.choices[0].message.content or "{}"
        return self._parse_sql_response(content)

    def _generate_json_sql(
        self,
        prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        """Generate SQL with JSON parsing fallback."""
        messages = self._build_messages(prompt, system_prompt)

        response = completion(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            num_retries=self.config.num_retries,
            **self._get_completion_kwargs(),
        )

        content = response.choices[0].message.content or ""
        return self._parse_sql_response(content)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build messages list for LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _sql_system_prompt(self) -> str:
        """System prompt optimized for SQL generation with chain-of-thought."""
        return """You are an expert SQL query generator.
Convert natural language to safe, accurate SQL.

## APPROACH
Use chain-of-thought reasoning:
1. UNDERSTAND: What entities and relationships are involved?
2. IDENTIFY: What columns and filters are needed?
3. PLAN: What JOINs and conditions are required?
4. GENERATE: Write the SQL query
5. VALIDATE: Check for correctness and safety

## RULES
- Only generate SELECT queries (no INSERT/UPDATE/DELETE/DROP)
- Use explicit JOINs with proper ON clauses, never implicit joins
- Use table aliases for clarity (e.g., u for users, o for orders)
- Include appropriate WHERE clauses based on the question
- Use LIMIT when the question implies a specific count
- Handle NULL values appropriately with COALESCE or IS NULL checks
- Use proper aggregate functions with GROUP BY when needed
- Qualify all column names with table aliases to avoid ambiguity

## RESULT FORMAT TYPES
Suggest the best display format for the result based on the question:
- "scalar": Single value (e.g., "How much total?", "Count of users")
- "pair": Single label-value pair (e.g., "Top category by spending")
- "record": Single entity with multiple fields (e.g., "Show user #123 details")
- "list": Simple list of items (e.g., "List all category names")
- "pair_list": Key-value pairs (e.g., "Spending by category", "Sales by region")
- "table": Full tabular data (e.g., "Show all transactions", "List users with details")
- "raw": Unformatted JSON output

## OUTPUT FORMAT
Respond with a JSON object:
{
    "reasoning": "Step-by-step thought process",
    "sql": "The SQL query",
    "explanation": "Brief explanation of what the query does",
    "confidence": 0.0-1.0,
    "tables_used": ["table1", "table2"],
    "format": "suggested format type from the list above"
}"""

    def _build_sql_prompt(
        self,
        question: str,
        schema_context: str,
        examples: list[dict] | None = None,
    ) -> str:
        """Build prompt for SQL generation."""
        parts = [schema_context]

        if examples:
            parts.append("\n## EXAMPLES")
            for ex in examples:
                parts.append(f"Question: {ex['question']}")
                parts.append(f"SQL: {ex['sql']}")
                parts.append("")

        parts.append(f"\n## QUESTION\n{question}")
        parts.append("\n## RESPONSE\nThink step-by-step, then provide the JSON response:")

        return "\n".join(parts)

    def _parse_sql_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response to extract SQL and metadata."""
        import re

        try:
            if content.strip().startswith("{"):
                parsed = json.loads(content)
                return self._normalize_parsed_response(parsed)

            json_match = re.search(r'\{[\s\S]*"sql"[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r",\s*}", "}", json_str)
                parsed = json.loads(json_str)
                return self._normalize_parsed_response(parsed)
        except json.JSONDecodeError:
            pass

        sql = content
        code_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", content)
        if code_match:
            sql = code_match.group(1)

        return {
            "sql": sql.strip(),
            "explanation": "Generated from natural language query",
            "confidence": 0.6,
            "tables_used": [],
            "reasoning": None,
            "format": None,
        }

    def _normalize_parsed_response(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Normalize parsed LLM response with defaults."""
        return {
            "sql": parsed.get("sql", ""),
            "explanation": parsed.get("explanation"),
            "confidence": parsed.get("confidence", 0.0),
            "tables_used": parsed.get("tables_used", []),
            "reasoning": parsed.get("reasoning"),
            "format": parsed.get("format"),
        }

    def _try_fallbacks(
        self,
        messages: list[dict],
        temperature: float | None,
        max_tokens: int | None,
        original_error: Exception,
    ) -> str:
        """Try fallback models if primary fails."""
        for fallback_model in self.config.fallback_models:
            try:
                response = completion(
                    model=fallback_model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    timeout=self.config.timeout,
                    **self._get_completion_kwargs(),
                )
                return response.choices[0].message.content or ""
            except Exception:
                continue

        raise LLMError(f"All models failed. Original error: {original_error}")

    def _select_best_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Select best result from multiple samples using voting."""
        if not results:
            raise LLMError("No results to select from")

        if len(results) == 1:
            return results[0]

        sql_counts: dict[str, int] = {}
        sql_to_result: dict[str, dict] = {}

        for result in results:
            sql = result.get("sql", "").strip().lower()
            sql_hash = hashlib.md5(sql.encode()).hexdigest()
            sql_counts[sql_hash] = sql_counts.get(sql_hash, 0) + 1
            if sql_hash not in sql_to_result:
                sql_to_result[sql_hash] = result

        best_hash = max(sql_counts, key=lambda k: sql_counts[k])
        return sql_to_result[best_hash]

    def get_cached_context(self, schema_context: str) -> str:
        """Get or cache schema context."""
        cache_key = hashlib.md5(schema_context.encode()).hexdigest()
        if cache_key not in self._schema_context_cache:
            self._schema_context_cache[cache_key] = schema_context
        return self._schema_context_cache[cache_key]

    def generate_human_response(
        self,
        question: str,
        rows: list[dict[str, Any]],
        row_count: int,
        format_type: str | None = None,
    ) -> str:
        """
        Generate a human-readable response summarizing query results.

        Args:
            question: The original natural language question
            rows: Query result rows (limited sample)
            row_count: Total number of rows
            format_type: The detected format type

        Returns:
            Natural language summary of the results
        """
        rows_sample = rows[:10] if rows else []
        rows_json = json.dumps(rows_sample, indent=2, default=str)

        prompt = f"""Given this question and query results, provide a natural response.

Question: {question}

Results ({row_count} total rows):
{rows_json}

Format type: {format_type or "unknown"}

Instructions:
- Respond naturally as if answering the user's question directly
- Include specific numbers/values from the results
- Format currency values nicely (e.g., $5,000.00)
- If there are many results, summarize the key findings
- Keep the response concise but informative
- Do not mention SQL, queries, or technical details
- Do not start with "Based on the results" or similar phrases
- Just answer the question directly"""

        system_prompt = (
            "You are a helpful assistant that summarizes data results in natural language. "
            "Respond conversationally as if directly answering the user's question."
        )

        try:
            return self.generate(prompt, system_prompt=system_prompt, max_tokens=500)
        except Exception:
            return self._fallback_human_response(rows, row_count, format_type)

    def _fallback_human_response(
        self,
        rows: list[dict[str, Any]],
        row_count: int,
        format_type: str | None,
    ) -> str:
        """Fallback human response when LLM fails."""
        if not rows:
            return "No results found."

        if row_count == 1 and len(rows[0]) == 1:
            value = list(rows[0].values())[0]
            if isinstance(value, (int, float)):
                return f"The result is {value:,.2f}."
            return f"The result is {value}."

        return f"Found {row_count} result{'s' if row_count != 1 else ''}."


def create_llm_provider(config: dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider from config."""
    llm_config = LLMConfig.from_dict(config)
    return LLMProvider(llm_config)
