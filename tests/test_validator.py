"""
tests/test_validator.py
────────────────────────
Unit tests for Module 4 (validator.py).

All three authoritative outcome categories are tested explicitly, along
with the two auxiliary outcomes (SKIPPED, EXECUTION_ERROR). A dedicated
test reproduces the exact 500-response pattern seen in the live Petstore
run to confirm UNDOCUMENTED_BEHAVIOR is correctly triggered — not
miscategorised as SPEC_VIOLATION or SPEC_MATCH.

No network calls, no LLM calls. Pure data-over-logic tests.
"""

from __future__ import annotations

from groundtruth.models import (
    EXECUTION_ERROR,
    SKIPPED,
    SPEC_MATCH,
    SPEC_VIOLATION,
    UNDOCUMENTED_BEHAVIOR,
    EndpointDefinition,
    ExecutionResult,
    ParameterDef,
    ResponseDef,
    TestCase,
)
from groundtruth.validator import validate

# ── Shared fixtures ───────────────────────────────────────────────────────────


def _endpoint(responses: dict) -> EndpointDefinition:
    """Build a minimal EndpointDefinition with the given responses dict."""
    return EndpointDefinition(
        path="/pet/{petId}",
        method="get",
        parameters=[
            ParameterDef(name="petId", location="path", required=True, schema={"type": "integer"})
        ],
        request_body=None,
        responses=responses,
        operation_id="getPetById",
        summary="Find pet by ID",
    )


# Pet schema matching our petstore fixture (id and name are required)
_PET_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "status"],
    "properties": {
        "id":     {"type": "integer"},
        "name":   {"type": "string"},
        "status": {"type": "string", "enum": ["available", "pending", "sold"]},
    },
}

# Endpoint with a 200 body schema and schema-less 400/404 codes
_ENDPOINT_WITH_SCHEMA = _endpoint({
    "200": ResponseDef(description="Successful", schema=_PET_SCHEMA),
    "400": ResponseDef(description="Invalid ID", schema=None),
    "404": ResponseDef(description="Not found",  schema=None),
})

# Endpoint that only documents 200/400/404 — no 500
_ENDPOINT_NO_500 = _ENDPOINT_WITH_SCHEMA


def _test_case(expected_status: int = 200) -> TestCase:
    return TestCase(
        id="tc_001",
        category="happy_path",
        description="fetch a pet",
        path_params={"petId": 10},
        query_params={},
        headers={},
        request_body=None,
        expected_status=expected_status,
    )


def _result(
    status_code: int | None = 200,
    body: object = None,
    skipped: bool = False,
    skip_reason: str | None = None,
    error: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        test_case_id="tc_001",
        endpoint_path="/pet/{petId}",
        method="get",
        status_code=status_code,
        response_body=body,
        response_headers={},
        response_time_ms=123.4,
        skipped=skipped,
        skip_reason=skip_reason,
        error=error,
    )


# ── SPEC_MATCH ────────────────────────────────────────────────────────────────


class TestSpecMatch:
    def test_200_with_valid_body(self):
        """Valid body for a documented code → SPEC_MATCH."""
        body = {"id": 10, "name": "doggie", "status": "available"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.outcome == SPEC_MATCH
        assert vr.schema_errors is None
        assert vr.actual_status_code == 200

    def test_400_no_schema_only_code_checked(self):
        """Documented code with no body schema → SPEC_MATCH regardless of body content."""
        vr = validate(
            _ENDPOINT_WITH_SCHEMA,
            _test_case(400),
            _result(400, {"some": "error body"}),
        )
        assert vr.outcome == SPEC_MATCH
        assert vr.schema_errors is None
        assert "body not validated" in vr.reason

    def test_404_no_schema_null_body(self):
        """Documented 404 with null body → SPEC_MATCH."""
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(404), _result(404, None))
        assert vr.outcome == SPEC_MATCH

    def test_llm_prediction_matched_true_when_guessed_correctly(self):
        body = {"id": 1, "name": "cat", "status": "pending"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.llm_prediction_matched is True

    def test_llm_prediction_matched_false_even_on_spec_match(self):
        """LLM guessed 200 but got 404. Outcome is still SPEC_MATCH (404 is documented)."""
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(404, None))
        assert vr.outcome == SPEC_MATCH
        assert vr.llm_prediction_matched is False

    def test_spec_with_default_response_fallback(self):
        """A 'default' response entry should match any undocumented-specific code."""
        endpoint = _endpoint({
            "200": ResponseDef(description="OK", schema=None),
            "default": ResponseDef(description="Unexpected", schema=None),
        })
        vr = validate(endpoint, _test_case(503), _result(503, None))
        assert vr.outcome == SPEC_MATCH


# ── SPEC_VIOLATION ────────────────────────────────────────────────────────────


class TestSpecViolation:
    def test_200_missing_required_field(self):
        """200 is documented, body is missing a required field → SPEC_VIOLATION."""
        body = {"id": 10}  # missing 'name' and 'status'
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.outcome == SPEC_VIOLATION
        assert vr.schema_errors is not None
        assert len(vr.schema_errors) > 0

    def test_200_wrong_type_for_field(self):
        """id should be integer, not a string → SPEC_VIOLATION."""
        body = {"id": "not-an-int", "name": "doggie", "status": "available"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.outcome == SPEC_VIOLATION
        # The error message should reference the 'id' field
        assert any("id" in err for err in vr.schema_errors)

    def test_schema_errors_listed_in_result(self):
        """schema_errors should be a non-empty list, not None."""
        body = {}  # all required fields missing
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.outcome == SPEC_VIOLATION
        assert isinstance(vr.schema_errors, list)
        assert len(vr.schema_errors) >= 3  # id, name, status all missing

    def test_spec_violation_does_not_depend_on_llm_guess(self):
        """LLM guessing 500 doesn't change a SPEC_VIOLATION to something else."""
        body = {"id": "wrong-type", "name": "x", "status": "available"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(500), _result(200, body))
        assert vr.outcome == SPEC_VIOLATION
        assert vr.llm_prediction_matched is False

    def test_schema_violation_reason_mentions_error_count(self):
        body = {}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert "error" in vr.reason.lower()


# ── UNDOCUMENTED_BEHAVIOR ─────────────────────────────────────────────────────


class TestUndocumentedBehavior:
    def test_500_not_in_spec(self):
        """
        This reproduces the exact live run result from the Petstore: the API
        returned 500 for inputs like petId=0 or petId=-1, but the spec only
        documents 200/400/404. This must be UNDOCUMENTED_BEHAVIOR, not a
        failure or a pass.
        """
        body = {
            "code": 500,
            "message": "There was an error processing your request. It has been logged (ID: 0b7da17ad6968eb6)"
        }
        vr = validate(_ENDPOINT_NO_500, _test_case(200), _result(500, body))
        assert vr.outcome == UNDOCUMENTED_BEHAVIOR

    def test_undocumented_is_not_spec_violation(self):
        """500 returned → must NOT be classified as SPEC_VIOLATION."""
        vr = validate(_ENDPOINT_NO_500, _test_case(200), _result(500, {"code": 500}))
        assert vr.outcome != SPEC_VIOLATION

    def test_undocumented_is_not_spec_match(self):
        """500 returned → must NOT be classified as SPEC_MATCH."""
        vr = validate(_ENDPOINT_NO_500, _test_case(500), _result(500, {}))
        assert vr.outcome != SPEC_MATCH

    def test_reason_mentions_documented_codes(self):
        """The reason string should list what codes ARE documented."""
        vr = validate(_ENDPOINT_NO_500, _test_case(200), _result(500, {}))
        assert "200" in vr.reason or "400" in vr.reason or "404" in vr.reason

    def test_arbitrary_undocumented_code(self):
        """A 418 (never in spec) should also be UNDOCUMENTED_BEHAVIOR."""
        vr = validate(_ENDPOINT_NO_500, _test_case(200), _result(418, None))
        assert vr.outcome == UNDOCUMENTED_BEHAVIOR

    def test_llm_prediction_matched_still_works_for_undocumented(self):
        """LLM correctly predicted 500 (informational) but outcome is still UNDOCUMENTED."""
        vr = validate(_ENDPOINT_NO_500, _test_case(500), _result(500, {}))
        assert vr.outcome == UNDOCUMENTED_BEHAVIOR
        assert vr.llm_prediction_matched is True  # LLM happened to guess right


# ── SKIPPED ───────────────────────────────────────────────────────────────────


class TestSkipped:
    def test_skipped_result_gives_skipped_outcome(self):
        result = _result(
            status_code=None,
            body=None,
            skipped=True,
            skip_reason="Method 'POST' is a mutation.",
        )
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert vr.outcome == SKIPPED
        assert vr.schema_errors is None
        assert vr.actual_status_code is None

    def test_skipped_llm_prediction_matched_is_false(self):
        result = _result(skipped=True, skip_reason="mutation guard")
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert vr.llm_prediction_matched is False

    def test_skipped_reason_propagated(self):
        result = _result(skipped=True, skip_reason="Method 'DELETE' is a mutation.")
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert "mutation" in vr.reason.lower() or "DELETE" in vr.reason


# ── EXECUTION_ERROR ───────────────────────────────────────────────────────────


class TestExecutionError:
    def test_network_error_gives_execution_error_outcome(self):
        result = _result(status_code=None, error="ConnectError: connection refused")
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert vr.outcome == EXECUTION_ERROR
        assert vr.schema_errors is None

    def test_timeout_gives_execution_error(self):
        result = _result(status_code=None, error="Request timed out after 10s")
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert vr.outcome == EXECUTION_ERROR

    def test_execution_error_reason_contains_error_message(self):
        result = _result(status_code=None, error="ConnectError: connection refused")
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert "ConnectError" in vr.reason or "connection" in vr.reason.lower()

    def test_none_status_code_with_no_error_field(self):
        """status_code=None + error=None (degenerate case) → EXECUTION_ERROR."""
        result = _result(status_code=None, error=None)
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(), result)
        assert vr.outcome == EXECUTION_ERROR


# ── LLM prediction is never used in the outcome decision ─────────────────────


class TestLLMGuaranteedNotUsedInOutcome:
    """
    Explicit boundary tests confirming that `expected_status` from the LLM
    cannot alter the spec-based outcome, regardless of what it contains.
    """

    def test_llm_guesses_500_does_not_cause_spec_violation(self):
        """LLM said 500, API returned 200 with a valid body → must be SPEC_MATCH."""
        body = {"id": 1, "name": "rex", "status": "available"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(500), _result(200, body))
        assert vr.outcome == SPEC_MATCH

    def test_llm_guesses_200_does_not_mask_undocumented_behavior(self):
        """LLM said 200, API returned 503 (not in spec) → UNDOCUMENTED_BEHAVIOR."""
        vr = validate(_ENDPOINT_NO_500, _test_case(200), _result(503, {}))
        assert vr.outcome == UNDOCUMENTED_BEHAVIOR

    def test_llm_guesses_correctly_does_not_upgrade_spec_violation(self):
        """LLM guessed 200 AND got 200 BUT body is wrong → still SPEC_VIOLATION."""
        body = {"wrong": "shape"}
        vr = validate(_ENDPOINT_WITH_SCHEMA, _test_case(200), _result(200, body))
        assert vr.outcome == SPEC_VIOLATION
        assert vr.llm_prediction_matched is True  # LLM was right on status
        # But that correctness did not help — outcome is still SPEC_VIOLATION
