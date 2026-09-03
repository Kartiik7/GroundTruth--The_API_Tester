"""
groundtruth/test_generator.py  —  Module 2
────────────────────────────────────────────
Uses the Groq LLM to generate structured test cases for one endpoint.

INPUT:  EndpointDefinition  (from Module 1)
        Groq API key, model name
OUTPUT: List[TestCase]       (schema-validated Python dataclasses)

Core architectural rule enforced here:
    The LLM is told it is generating PROPOSALS ONLY. Its expected_status
    is explicitly labelled as a guess. The module validates the LLM's
    JSON output against a strict schema — it does not trust free-form prose.

Retry policy:
    If the first LLM response fails JSON/schema validation, a single
    correction prompt is sent. If that also fails, GeneratorError is raised.
    There is no further retry — the failure is surfaced to the caller.

Public API
──────────
    generate_test_cases(endpoint, api_key, model) -> List[TestCase]
    GeneratorError                                  (exception class)
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

import jsonschema
from groq import Groq, GroqError

from groundtruth.models import EndpointDefinition, TestCase

# ── Exceptions ────────────────────────────────────────────────────────────────


class GeneratorError(Exception):
    """Raised when the LLM fails to produce valid test cases."""


# ── JSON schema that the LLM output must satisfy ──────────────────────────────
#
# This schema is the enforcement boundary between the LLM world and the
# typed Python world. If the LLM's JSON doesn't satisfy it, we retry once.

_TEST_CASE_ARRAY_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "maxItems": 12,
    "items": {
        "type": "object",
        "required": [
            "id",
            "category",
            "description",
            "path_params",
            "query_params",
            "headers",
            "request_body",
            "expected_status",
        ],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "category": {
                "type": "string",
                "enum": ["happy_path", "invalid_input", "not_found", "edge_case"],
            },
            "description": {"type": "string", "minLength": 1},
            "path_params": {"type": "object"},
            "query_params": {"type": "object"},
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            # request_body accepts any JSON value including null
            "request_body": {},
            "expected_status": {"type": "integer", "minimum": 100, "maximum": 599},
        },
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a test case generator for HTTP APIs. You will receive an OpenAPI
    endpoint definition in JSON format and must output test cases for it.

    OUTPUT FORMAT — CRITICAL:
    Output ONLY a raw JSON array. No markdown, no code fences, no prose.
    Your response must start with [ and end with ].

    Each element of the array must be a JSON object with EXACTLY these fields:
      "id"             – string, unique identifier e.g. "tc_001"
      "category"       – one of: "happy_path", "invalid_input", "not_found", "edge_case"
      "description"    – string, brief description of what this case tests
      "path_params"    – object, values for path template variables (e.g. {"petId": 10})
      "query_params"   – object, query string key-value pairs (can be {})
      "headers"        – object, string key-value pairs for request headers (can be {})
      "request_body"   – any JSON value or null (null for GET/DELETE)
      "expected_status"– integer, your predicted HTTP response status code

    RULES:
    1. Generate between 4 and 8 test cases.
    2. Always include at least one "happy_path" case with valid, realistic inputs.
    3. Cover multiple categories where the endpoint structure makes them sensible.
    4. For path parameters, substitute concrete values (not template placeholders).
    5. Your expected_status is a PROPOSAL ONLY — it will never be used as the
       pass/fail criterion. The OpenAPI spec is the authority on correctness.
    6. Do NOT write assertions, code, or any text outside the JSON array.
""")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _serialize_endpoint(endpoint: EndpointDefinition) -> str:
    """
    Serialize an EndpointDefinition to a JSON string for injection into the prompt.

    We serialize only the fields the LLM needs to reason about inputs —
    not the full internal representation.
    """
    payload = {
        "path": endpoint.path,
        "method": endpoint.method.upper(),
        "summary": endpoint.summary,
        "parameters": [
            {
                "name": p.name,
                "in": p.location,
                "required": p.required,
                "schema": p.schema,
            }
            for p in endpoint.parameters
        ],
        "requestBody": (
            {
                "required": endpoint.request_body.required,
                "content_type": endpoint.request_body.content_type,
                "schema": endpoint.request_body.schema,
            }
            if endpoint.request_body
            else None
        ),
        "responses": {
            code: {"description": r.description}
            for code, r in endpoint.responses.items()
        },
    }
    return json.dumps(payload, indent=2)


def _strip_markdown_fences(text: str) -> str:
    """
    Strip accidental markdown code fences from the LLM's output.

    Some models wrap JSON in ```json ... ``` even when instructed not to.
    We strip those fences defensively before JSON parsing.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()
    return stripped


def _parse_and_validate(raw: str) -> List[TestCase]:
    """
    Parse the LLM's raw string output into a validated list of TestCase objects.

    Steps:
        1. Strip accidental markdown fences
        2. json.loads() — fail fast on malformed JSON
        3. jsonschema.validate() — enforce the strict schema contract
        4. Hydrate into TestCase dataclasses

    Raises:
        GeneratorError: with a clear message on parse or validation failure.
    """
    cleaned = _strip_markdown_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise GeneratorError(
            f"LLM output is not valid JSON.\n"
            f"Parse error: {e}\n\n"
            f"Raw output (first 500 chars):\n{raw[:500]}"
        ) from e

    try:
        jsonschema.validate(instance=data, schema=_TEST_CASE_ARRAY_SCHEMA)
    except jsonschema.ValidationError as e:
        raise GeneratorError(
            f"LLM output failed schema validation.\n"
            f"Violation: {e.message}\n"
            f"Path: {list(e.absolute_path)}"
        ) from e

    return [
        TestCase(
            id=tc["id"],
            category=tc["category"],
            description=tc["description"],
            path_params=tc["path_params"],
            query_params=tc["query_params"],
            headers=tc["headers"],
            request_body=tc["request_body"],
            expected_status=tc["expected_status"],
        )
        for tc in data
    ]


def _call_groq(client: Groq, model: str, messages: list) -> str:
    """
    Make a single chat completion call to Groq and return the content string.

    Raises:
        GeneratorError: wrapping any Groq SDK error so callers don't need
                        to import from groq directly.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=2048,
        )
    except GroqError as e:
        raise GeneratorError(f"Groq API error: {e}") from e

    content = response.choices[0].message.content
    if not content:
        raise GeneratorError("Groq returned an empty response.")
    return content


# ── Public API ────────────────────────────────────────────────────────────────


def generate_test_cases(
    endpoint: EndpointDefinition,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
) -> List[TestCase]:
    """
    Generate 4–8 test cases for the given endpoint using the Groq LLM.

    The LLM's output is validated against a strict JSON schema. If the
    first response fails validation, one correction attempt is made.
    If the correction also fails, GeneratorError is raised.

    Args:
        endpoint: The parsed endpoint definition from Module 1.
        api_key:  Groq API key (from environment — never hardcoded).
        model:    Groq model identifier. Defaults to llama-3.3-70b-versatile.

    Returns:
        A list of TestCase dataclasses, schema-validated and ready for
        Module 3 (executor).

    Raises:
        GeneratorError: if the LLM fails to produce valid output in 2 attempts,
                        or if the Groq API call itself fails.
    """
    client = Groq(api_key=api_key)
    endpoint_json = _serialize_endpoint(endpoint)

    initial_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate test cases for this endpoint:\n\n"
                f"{endpoint_json}"
            ),
        },
    ]

    # ── Attempt 1 ────────────────────────────────────────────────────────────
    raw_first = _call_groq(client, model, initial_messages)
    try:
        return _parse_and_validate(raw_first)
    except GeneratorError as first_error:
        print(f"  [warn] First LLM response failed validation: {first_error}")
        print("  [warn] Sending correction prompt (attempt 2 of 2)...")

    # ── Attempt 2 (correction) ────────────────────────────────────────────────
    # We give the model its own failed output + the error so it can self-correct.
    correction_messages = initial_messages + [
        {"role": "assistant", "content": raw_first},
        {
            "role": "user",
            "content": (
                f"Your response caused this error:\n{first_error}\n\n"
                "Please fix it. Output ONLY a valid JSON array. "
                "No markdown, no prose, no code fences."
            ),
        },
    ]

    raw_second = _call_groq(client, model, correction_messages)
    try:
        return _parse_and_validate(raw_second)
    except GeneratorError as second_error:
        raise GeneratorError(
            f"LLM failed to produce valid test cases after 2 attempts.\n"
            f"First error:  {first_error}\n"
            f"Second error: {second_error}"
        ) from second_error
