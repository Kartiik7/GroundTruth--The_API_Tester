"""
groundtruth/validator.py  —  Module 4
───────────────────────────────────────
Validates actual HTTP responses against the OpenAPI spec.

INPUT:  EndpointDefinition  (from Module 1 — the authoritative schema source)
        TestCase            (from Module 2 — carries the LLM's guess as metadata)
        ExecutionResult     (from Module 3 — the real HTTP response facts)

OUTPUT: ValidationResult    (one of five outcome labels; three are authoritative)

Core principle enforced here:
    The outcome is determined SOLELY by comparing the actual HTTP response
    against the OpenAPI spec extracted in Module 1. The LLM's
    `expected_status` field is used ONLY to populate `llm_prediction_matched`
    — an informational boolean that is never an input to the outcome decision.

Three authoritative outcomes for completed requests:

    SPEC_MATCH
        Status code is documented AND body matches its schema (or no schema
        is defined for that code, so only the status code needs to match).

    SPEC_VIOLATION
        Status code IS documented, but the response body fails jsonschema
        validation against the spec schema for that code.

    UNDOCUMENTED_BEHAVIOR
        Status code is NOT listed in the spec for this endpoint at all.
        Distinct from a failure — the spec may be incomplete. Never merged
        into pass or fail silently.

Two auxiliary outcomes for non-completable requests:

    SKIPPED
        The executor's mutation guard prevented the request from being sent.

    EXECUTION_ERROR
        A network error or timeout — no HTTP response was ever received.

Public API
──────────
    validate(endpoint, test_case, result) -> ValidationResult
"""

from __future__ import annotations

from typing import List, Optional

import jsonschema

from groundtruth.models import (
    EXECUTION_ERROR,
    SKIPPED,
    SPEC_MATCH,
    SPEC_VIOLATION,
    UNDOCUMENTED_BEHAVIOR,
    EndpointDefinition,
    ExecutionResult,
    TestCase,
    ValidationResult,
)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _collect_schema_errors(body: object, schema: dict) -> List[str]:
    """
    Run jsonschema validation and collect all error messages.

    Returns an empty list if the body is valid.
    Returns a list of short, readable error strings if it is not.

    We use Draft7Validator (the most common for OpenAPI 3.x) and collect
    all errors in one pass rather than stopping at the first, so that the
    report is as informative as possible.
    """
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(body), key=lambda e: list(e.path))
    return [
        f"[{'.'.join(str(p) for p in err.path) or '<root>'}] {err.message}"
        for err in errors
    ]


def _outcome_label_color_hint(outcome: str) -> str:
    """Return a visual prefix for the outcome label (used nowhere in logic)."""
    return {
        SPEC_MATCH: "✓",
        SPEC_VIOLATION: "✗",
        UNDOCUMENTED_BEHAVIOR: "⚠",
        SKIPPED: "⊘",
        EXECUTION_ERROR: "✗",
    }.get(outcome, "?")


# ── Public API ────────────────────────────────────────────────────────────────


def validate(
    endpoint: EndpointDefinition,
    test_case: TestCase,
    result: ExecutionResult,
) -> ValidationResult:
    """
    Determine the authoritative outcome for one executed test case.

    The decision tree:

        result.skipped?
            → SKIPPED

        result.error or result.status_code is None?
            → EXECUTION_ERROR

        actual status code in endpoint.responses?
            No  → UNDOCUMENTED_BEHAVIOR
            Yes →
                response schema defined for that code?
                    No  → SPEC_MATCH  (status matches, no body contract to check)
                    Yes →
                        body passes jsonschema?
                            Yes → SPEC_MATCH
                            No  → SPEC_VIOLATION

    The LLM's `expected_status` is read ONCE, at the end, purely to
    compute `llm_prediction_matched`. It influences nothing above.

    Args:
        endpoint:  The parsed endpoint definition (Module 1 output).
        test_case: The LLM-generated test case (Module 2 output). Only
                   `expected_status` is read here, and only for the
                   informational `llm_prediction_matched` field.
        result:    The HTTP execution result (Module 3 output).

    Returns:
        A ValidationResult with the authoritative outcome label.
    """
    # ── Pre-flight: requests that never produced an HTTP response ─────────────

    if result.skipped:
        return ValidationResult(
            test_case_id=result.test_case_id,
            outcome=SKIPPED,
            llm_prediction_matched=False,
            reason=result.skip_reason or "Request was skipped.",
            actual_status_code=None,
            llm_expected_status=test_case.expected_status,
            schema_errors=None,
        )

    if result.error or result.status_code is None:
        return ValidationResult(
            test_case_id=result.test_case_id,
            outcome=EXECUTION_ERROR,
            llm_prediction_matched=False,
            reason=result.error or "Request produced no status code.",
            actual_status_code=result.status_code,
            llm_expected_status=test_case.expected_status,
            schema_errors=None,
        )

    # ── Step 1: Is the actual status code documented in the spec? ─────────────

    actual_code = result.status_code
    code_key = str(actual_code)

    # The spec may also have a "default" response entry; check for it as a
    # fallback after the exact code lookup fails.
    response_def = endpoint.responses.get(code_key) or endpoint.responses.get("default")

    # Informational flag — computed last but kept in one place for clarity
    llm_matched = actual_code == test_case.expected_status

    if response_def is None:
        documented = sorted(endpoint.responses.keys())
        return ValidationResult(
            test_case_id=result.test_case_id,
            outcome=UNDOCUMENTED_BEHAVIOR,
            llm_prediction_matched=llm_matched,
            reason=(
                f"Status {actual_code} is not documented for this endpoint. "
                f"Documented codes: {documented}. "
                "This may indicate an undocumented error path or a spec gap."
            ),
            actual_status_code=actual_code,
            llm_expected_status=test_case.expected_status,
            schema_errors=None,
        )

    # ── Step 2: If a body schema is defined, validate the response body ────────

    if response_def.schema is None:
        # No body schema documented for this status code — only the code matters.
        return ValidationResult(
            test_case_id=result.test_case_id,
            outcome=SPEC_MATCH,
            llm_prediction_matched=llm_matched,
            reason=(
                f"Status {actual_code} matches spec. "
                f"No body schema defined for {code_key} — body not validated."
            ),
            actual_status_code=actual_code,
            llm_expected_status=test_case.expected_status,
            schema_errors=None,
        )

    # Body schema exists — run jsonschema validation.
    errors = _collect_schema_errors(result.response_body, response_def.schema)

    if not errors:
        return ValidationResult(
            test_case_id=result.test_case_id,
            outcome=SPEC_MATCH,
            llm_prediction_matched=llm_matched,
            reason=(
                f"Status {actual_code} matches spec and response body "
                f"satisfies the {code_key} schema."
            ),
            actual_status_code=actual_code,
            llm_expected_status=test_case.expected_status,
            schema_errors=None,
        )

    return ValidationResult(
        test_case_id=result.test_case_id,
        outcome=SPEC_VIOLATION,
        llm_prediction_matched=llm_matched,
        reason=(
            f"Status {actual_code} is documented but response body violates "
            f"the {code_key} schema ({len(errors)} error(s) found)."
        ),
        actual_status_code=actual_code,
        llm_expected_status=test_case.expected_status,
        schema_errors=errors,
    )
