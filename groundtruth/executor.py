"""
groundtruth/executor.py  —  Module 3
──────────────────────────────────────
Executes a TestCase against the real API via HTTP and records the raw result.

INPUT:  TestCase            (from Module 2)
        EndpointDefinition  (from Module 1, for path/method info)
        base_url            (CLI-provided)
        allow_mutations     (CLI flag, default False)

OUTPUT: ExecutionResult     (raw facts — no judgment, no assertions)

Key responsibilities:
  - Path parameter substitution: '/pet/{petId}' + {'petId': 42} → '/pet/42'
  - Mutation guard: skips POST/PUT/PATCH/DELETE unless allow_mutations=True
  - Hard 10-second timeout on every request
  - Parses response as JSON if possible, falls back to plain text
  - Never raises on HTTP-level errors (4xx/5xx) — those are valid responses
    that Module 4 will evaluate. Only network/timeout errors are caught.

Public API
──────────
    execute(test_case, endpoint, base_url, allow_mutations) -> ExecutionResult
"""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from groundtruth.models import EndpointDefinition, ExecutionResult, TestCase

# ── Constants ─────────────────────────────────────────────────────────────────

_MUTATION_METHODS = frozenset({"post", "put", "patch", "delete"})
_TIMEOUT_SECONDS = 10.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _substitute_path_params(template: str, params: Dict[str, Any]) -> str:
    """
    Replace {name} placeholders in a URL path template with concrete values.

    Example:
        template = '/pet/{petId}'
        params   = {'petId': 42}
        result   = '/pet/42'

    Values are coerced to str to handle integers and other scalar types.
    Unknown placeholders in params (those not in the template) are silently
    ignored. Un-filled placeholders in the template are left as-is so that
    Module 4 can detect missing required path parameters as a failure mode.
    """
    result = template
    for key, value in params.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def _parse_response_body(response: httpx.Response) -> Any:
    """
    Try to parse the response body as JSON; fall back to the raw text string.

    We never raise here — a body that doesn't parse as JSON is still a valid
    response body (e.g. a plain-text error message). Module 4 will handle
    schema validation failures.
    """
    try:
        return response.json()
    except Exception:
        return response.text


# ── Public API ────────────────────────────────────────────────────────────────


def execute(
    test_case: TestCase,
    endpoint: EndpointDefinition,
    base_url: str,
    allow_mutations: bool = False,
) -> ExecutionResult:
    """
    Execute a single test case against the live API and return raw results.

    This function is intentionally free of any correctness judgment.
    It records facts: status code, body, timing, headers. Whether those
    facts constitute a pass or fail is determined by Module 4.

    Args:
        test_case:       The test case to execute (from Module 2).
        endpoint:        The endpoint definition (from Module 1). Used to
                         determine the HTTP method and path template.
        base_url:        The running API's base URL, e.g. 'https://api.example.com'.
                         Trailing slashes are stripped.
        allow_mutations: If False (the default), any test case targeting a
                         POST, PUT, PATCH, or DELETE endpoint is skipped and
                         returned as ExecutionResult(skipped=True).

    Returns:
        ExecutionResult with actual HTTP response data, or a skipped/error
        result if the request could not be completed.
    """
    method = endpoint.method.lower()

    # ── Mutation guard ────────────────────────────────────────────────────────
    if method in _MUTATION_METHODS and not allow_mutations:
        return ExecutionResult(
            test_case_id=test_case.id,
            endpoint_path=endpoint.path,
            method=method,
            status_code=None,
            response_body=None,
            response_headers={},
            response_time_ms=0.0,
            skipped=True,
            skip_reason=(
                f"Method '{method.upper()}' is a mutation. "
                "Pass --allow-mutations to enable mutation requests."
            ),
            error=None,
        )

    # ── Build URL ─────────────────────────────────────────────────────────────
    resolved_path = _substitute_path_params(endpoint.path, test_case.path_params)
    url = base_url.rstrip("/") + resolved_path

    # ── Build request kwargs ──────────────────────────────────────────────────
    kwargs: Dict[str, Any] = {
        "params": test_case.query_params if test_case.query_params else None,
        "headers": test_case.headers if test_case.headers else {},
        "timeout": _TIMEOUT_SECONDS,
    }

    # Attach a JSON body only for methods that conventionally carry one
    if test_case.request_body is not None and method in {"post", "put", "patch"}:
        kwargs["json"] = test_case.request_body

    # ── Execute ───────────────────────────────────────────────────────────────
    start = time.perf_counter()

    try:
        with httpx.Client(follow_redirects=True) as client:
            http_method = getattr(client, method)
            response: httpx.Response = http_method(url, **kwargs)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        body = _parse_response_body(response)

        return ExecutionResult(
            test_case_id=test_case.id,
            endpoint_path=endpoint.path,
            method=method,
            status_code=response.status_code,
            response_body=body,
            response_headers=dict(response.headers),
            response_time_ms=elapsed_ms,
            skipped=False,
            skip_reason=None,
            error=None,
        )

    except httpx.TimeoutException:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return ExecutionResult(
            test_case_id=test_case.id,
            endpoint_path=endpoint.path,
            method=method,
            status_code=None,
            response_body=None,
            response_headers={},
            response_time_ms=elapsed_ms,
            skipped=False,
            skip_reason=None,
            error=f"Request timed out after {_TIMEOUT_SECONDS}s",
        )

    except httpx.RequestError as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return ExecutionResult(
            test_case_id=test_case.id,
            endpoint_path=endpoint.path,
            method=method,
            status_code=None,
            response_body=None,
            response_headers={},
            response_time_ms=elapsed_ms,
            skipped=False,
            skip_reason=None,
            error=f"{type(exc).__name__}: {exc}",
        )
