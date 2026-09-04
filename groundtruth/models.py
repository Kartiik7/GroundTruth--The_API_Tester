"""
groundtruth/models.py
─────────────────────
Shared data contracts for all GroundTruth modules.

These dataclasses are the ONLY way modules communicate with each other.
No module imports internals from another module — they only pass these
plain data objects across boundaries.

Design note: fields are intentionally kept as plain Python types (dict,
list, str) rather than parsed sub-objects so they round-trip cleanly
through json.dumps / dataclasses.asdict without any custom serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Module 1 output ───────────────────────────────────────────────────────────


@dataclass
class ParameterDef:
    """A single parameter declared in an OpenAPI operation."""

    name: str
    """The parameter name, e.g. 'petId'."""

    location: str
    """Where the param lives: 'path', 'query', 'header', or 'cookie'."""

    required: bool
    """Whether the spec marks this param as required."""

    schema: Dict[str, Any]
    """Raw JSON Schema fragment describing the parameter's type/format."""


@dataclass
class RequestBodyDef:
    """Describes the request body for a POST/PUT/PATCH operation."""

    required: bool
    """Whether the spec marks the body as required."""

    content_type: str
    """The primary content type, e.g. 'application/json'."""

    schema: Dict[str, Any]
    """Raw JSON Schema for the body."""


@dataclass
class ResponseDef:
    """Describes one documented response status code for an operation."""

    description: str
    """Human-readable description from the spec."""

    schema: Optional[Dict[str, Any]]
    """
    Raw JSON Schema for the response body.
    None if the spec documents no body for this status code (e.g. 204).
    """


@dataclass
class EndpointDefinition:
    """
    Complete, parsed definition of a single API endpoint.

    This is the primary output of Module 1 and the primary input for
    Modules 2 and 4. It is a pure data object — no HTTP logic, no LLM
    logic lives here.
    """

    path: str
    """The path template as written in the spec, e.g. '/pets/{petId}'."""

    method: str
    """HTTP method in lowercase, e.g. 'get', 'post'."""

    parameters: List[ParameterDef]
    """All parameters (path + query + header), merged from path-level and
    operation-level declarations."""

    request_body: Optional[RequestBodyDef]
    """Request body definition, or None for methods that don't have one."""

    responses: Dict[str, ResponseDef]
    """Keyed by status code string, e.g. '200', '404', 'default'."""

    operation_id: Optional[str]
    """The operationId from the spec, if present."""

    summary: Optional[str]
    """The summary string from the spec, if present."""


# ── Module 2 output ───────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """
    A single test case proposed by the LLM.

    !! IMPORTANT ARCHITECTURAL RULE !!
    The `expected_status` field is the LLM's GUESS. It is carried as
    metadata for discrepancy reporting (Module 5) ONLY. It is never used
    as the pass/fail criterion. Module 4 (Schema Validator) is the sole
    authority on correctness.
    """

    id: str
    """Unique identifier for this test case, e.g. 'tc_001'."""

    category: str
    """One of: 'happy_path', 'invalid_input', 'not_found', 'edge_case'."""

    description: str
    """Brief human-readable description of what this case is testing."""

    path_params: Dict[str, Any]
    """Values to substitute into path template variables, e.g. {'petId': 42}."""

    query_params: Dict[str, Any]
    """Key-value pairs for the query string."""

    headers: Dict[str, str]
    """HTTP request headers to send."""

    request_body: Any
    """Request body for POST/PUT/PATCH, or None."""

    expected_status: int
    """LLM's predicted HTTP status code. NOT authoritative — see note above."""


# ── Module 3 output ───────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """
    The raw result of executing one TestCase against the real API.

    The executor records facts — status code, body, timing — without
    making any judgment about correctness. Judgment is deferred to Module 4.
    """

    test_case_id: str
    """Matches TestCase.id for correlation."""

    endpoint_path: str
    """The path template (not the resolved URL), for reference."""

    method: str
    """HTTP method used, lowercase."""

    status_code: Optional[int]
    """
    The actual HTTP status code returned.
    None if the request did not complete (timeout or network error).
    """

    response_body: Any
    """
    Parsed JSON body if the response was valid JSON, otherwise a raw string.
    None if the request did not complete or was skipped.
    """

    response_headers: Dict[str, str]
    """All response headers as a flat string dict."""

    response_time_ms: float
    """Wall-clock time from request start to response received, in milliseconds."""

    skipped: bool
    """True if the mutation guard fired and no HTTP request was made."""

    skip_reason: Optional[str]
    """Human-readable reason for skipping, or None if not skipped."""

    error: Optional[str]
    """
    Network or timeout error message.
    None on success or when skipped.
    """


# ── Module 4 output ───────────────────────────────────────────────────────────

# The three authoritative outcome labels. They are strings (not an Enum) so
# they serialise cleanly through dataclasses.asdict() / json.dumps().
SPEC_MATCH = "SPEC_MATCH"
SPEC_VIOLATION = "SPEC_VIOLATION"
UNDOCUMENTED_BEHAVIOR = "UNDOCUMENTED_BEHAVIOR"
# Two auxiliary labels for requests that never completed:
SKIPPED = "SKIPPED"
EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass
class ValidationResult:
    """
    The authoritative verdict for one test case from Module 4.

    Outcome rules (exactly one applies per completed request):

    SPEC_MATCH
        The actual status code IS documented for this endpoint, AND
        the response body satisfies the spec schema for that status code
        (or the spec defines no body schema for that code, in which case
        only the status code needs to match).

    SPEC_VIOLATION
        The actual status code IS documented, but the response body does
        NOT satisfy the spec schema for that code (missing required fields,
        wrong types, etc.). The API responded with a known code but broke
        its own contract.

    UNDOCUMENTED_BEHAVIOR
        The actual status code is NOT listed anywhere in the spec for this
        endpoint. This is reported as a distinct gap — not automatically a
        failure of the API under test, because the spec may simply be
        incomplete. It must not be silently merged into pass or fail.

    SKIPPED / EXECUTION_ERROR
        Auxiliary labels for requests that never produced an HTTP response
        (mutation guard fired, or network/timeout error). No schema
        validation is possible for these cases.

    !! CRITICAL RULE !!
    `llm_prediction_matched` is INFORMATIONAL ONLY. It records whether
    the LLM's `expected_status` guess happened to equal the actual status
    code. It is never used in the outcome decision. The outcome is
    determined solely by comparing the actual response against the
    OpenAPI spec.
    """

    test_case_id: str
    """Matches TestCase.id for correlation."""

    outcome: str
    """One of: SPEC_MATCH, SPEC_VIOLATION, UNDOCUMENTED_BEHAVIOR, SKIPPED, EXECUTION_ERROR."""

    llm_prediction_matched: bool
    """
    Whether the LLM's expected_status happened to equal the actual status code.
    Informational only — never used to determine outcome.
    """

    reason: str
    """Short human-readable explanation of why this outcome was reached."""

    actual_status_code: Optional[int]
    """The HTTP status code that was actually received, or None."""

    llm_expected_status: int
    """The LLM's guess, carried here for reporting/discrepancy display."""

    schema_errors: Optional[List[str]]
    """
    List of JSON Schema validation error messages.
    Populated only when outcome == SPEC_VIOLATION.
    None for all other outcomes.
    """


# ── Module 5 output ───────────────────────────────────────────────────────────


@dataclass
class ReportSummary:
    """
    Aggregated statistics for one full endpoint test run.

    Produced by Module 5 (report.py). Contains only derived numbers —
    the raw lists (TestCase, ExecutionResult, ValidationResult) are kept
    outside this object so it stays lightweight and serialisable.
    """

    total_cases: int
    """Total number of test cases in the run (including skipped/errored)."""

    outcome_counts: Dict[str, int]
    """
    Count per outcome label.
    Keys: SPEC_MATCH, SPEC_VIOLATION, UNDOCUMENTED_BEHAVIOR, SKIPPED, EXECUTION_ERROR.
    Missing keys mean zero for that outcome.
    """

    outcome_pct: Dict[str, float]
    """
    Percentage of total_cases for each outcome (0.0 - 100.0).
    Same keys as outcome_counts.
    """

    by_category: Dict[str, Dict[str, int]]
    """
    Cross-table: test category -> outcome label -> count.
    e.g. {'happy_path': {'SPEC_MATCH': 1, 'UNDOCUMENTED_BEHAVIOR': 1}}
    Only categories and outcomes that actually appear are included.
    """

    llm_matched_count: int
    """Number of cases where llm_prediction_matched is True."""

    llm_checkable_count: int
    """
    Denominator for LLM accuracy: cases that produced an HTTP response
    (excludes SKIPPED and EXECUTION_ERROR, where no status was returned).
    """

    llm_accuracy_pct: float
    """
    llm_matched_count / llm_checkable_count * 100, or 0.0 if none checkable.
    Informational only -- never used for pass/fail.
    """

    spec_violation_ids: List[str]
    """
    test_case_ids where outcome == SPEC_VIOLATION.
    Surfaced prominently in the console report: these are actual contract
    breaks where the API returned a documented status code but with a body
    that violated its own spec schema.
    """

