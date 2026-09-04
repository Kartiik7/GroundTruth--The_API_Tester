"""
tests/test_report.py
---------------------
Unit tests for Module 5 (report.py).

All tests use synthetic ValidationResult / TestCase lists -- no network,
no LLM, no real API calls. The tests exercise build_summary() exclusively
because that is the only place aggregation logic lives. Console and JSON
output functions are format-only and are tested by inspection in the CLI.

Coverage:
  - Outcome count and percentage calculation
  - By-category cross-table grouping
  - LLM accuracy computation (correct denominator: excludes SKIPPED/ERROR)
  - Spec violation ID collection
  - Edge cases: all-skipped run, zero cases, all violations
"""

from __future__ import annotations

from groundtruth.models import (
    EXECUTION_ERROR,
    SKIPPED,
    SPEC_MATCH,
    SPEC_VIOLATION,
    UNDOCUMENTED_BEHAVIOR,
    ReportSummary,
    TestCase,
    ValidationResult,
)
from groundtruth.report import build_summary


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _tc(id: str, category: str = "happy_path", expected: int = 200) -> TestCase:
    return TestCase(
        id=id,
        category=category,
        description=f"test {id}",
        path_params={},
        query_params={},
        headers={},
        request_body=None,
        expected_status=expected,
    )


def _vr(
    id: str,
    outcome: str,
    llm_matched: bool = False,
    actual: int | None = 200,
    llm_expected: int = 200,
    schema_errors: list | None = None,
) -> ValidationResult:
    return ValidationResult(
        test_case_id=id,
        outcome=outcome,
        llm_prediction_matched=llm_matched,
        reason="test reason",
        actual_status_code=actual,
        llm_expected_status=llm_expected,
        schema_errors=schema_errors,
    )


# ── Standard mixed run fixture ────────────────────────────────────────────────

# 6 test cases across 3 categories, covering all three authoritative outcomes

_TEST_CASES = [
    _tc("tc_001", "happy_path",    expected=200),
    _tc("tc_002", "happy_path",    expected=200),
    _tc("tc_003", "invalid_input", expected=400),
    _tc("tc_004", "not_found",     expected=404),
    _tc("tc_005", "edge_case",     expected=200),
    _tc("tc_006", "edge_case",     expected=500),
]

_VALIDATIONS = [
    _vr("tc_001", SPEC_MATCH,            llm_matched=True,  actual=200, llm_expected=200),
    _vr("tc_002", SPEC_VIOLATION,        llm_matched=True,  actual=200, llm_expected=200,
        schema_errors=["[name] 'name' is a required property"]),
    _vr("tc_003", SPEC_MATCH,            llm_matched=True,  actual=400, llm_expected=400),
    _vr("tc_004", UNDOCUMENTED_BEHAVIOR, llm_matched=False, actual=500, llm_expected=404),
    _vr("tc_005", UNDOCUMENTED_BEHAVIOR, llm_matched=False, actual=500, llm_expected=200),
    _vr("tc_006", UNDOCUMENTED_BEHAVIOR, llm_matched=True,  actual=500, llm_expected=500),
]


# ── Outcome counts ────────────────────────────────────────────────────────────


class TestOutcomeCounts:
    def test_total_cases(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert s.total_cases == 6

    def test_spec_match_count(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert s.outcome_counts[SPEC_MATCH] == 2

    def test_spec_violation_count(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert s.outcome_counts[SPEC_VIOLATION] == 1

    def test_undocumented_behavior_count(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert s.outcome_counts[UNDOCUMENTED_BEHAVIOR] == 3

    def test_absent_outcomes_not_in_counts(self):
        """SKIPPED and EXECUTION_ERROR should not appear when there are none."""
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert SKIPPED not in s.outcome_counts
        assert EXECUTION_ERROR not in s.outcome_counts

    def test_skipped_counted_when_present(self):
        tcs = [_tc("tc_a", "happy_path")]
        vrs = [_vr("tc_a", SKIPPED, actual=None)]
        s = build_summary(tcs, vrs)
        assert s.outcome_counts[SKIPPED] == 1
        assert s.total_cases == 1


# ── Percentages ───────────────────────────────────────────────────────────────


class TestPercentages:
    def test_percentages_sum_to_100(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        total_pct = sum(s.outcome_pct.values())
        assert abs(total_pct - 100.0) < 0.5  # rounding tolerance

    def test_spec_match_pct(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        # 2 out of 6 = 33.3%
        assert abs(s.outcome_pct[SPEC_MATCH] - 33.3) < 0.2

    def test_undocumented_pct(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        # 3 out of 6 = 50.0%
        assert abs(s.outcome_pct[UNDOCUMENTED_BEHAVIOR] - 50.0) < 0.1

    def test_zero_total_does_not_crash(self):
        s = build_summary([], [])
        assert s.total_cases == 0
        assert s.outcome_pct == {}

    def test_all_same_outcome_is_100_pct(self):
        tcs = [_tc("tc_1"), _tc("tc_2"), _tc("tc_3")]
        vrs = [
            _vr("tc_1", SPEC_MATCH),
            _vr("tc_2", SPEC_MATCH),
            _vr("tc_3", SPEC_MATCH),
        ]
        s = build_summary(tcs, vrs)
        assert s.outcome_pct[SPEC_MATCH] == 100.0


# ── By-category cross-table ───────────────────────────────────────────────────


class TestByCategory:
    def test_happy_path_has_correct_outcomes(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        hp = s.by_category["happy_path"]
        # tc_001 = SPEC_MATCH, tc_002 = SPEC_VIOLATION
        assert hp[SPEC_MATCH] == 1
        assert hp[SPEC_VIOLATION] == 1

    def test_not_found_category(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        nf = s.by_category["not_found"]
        assert nf[UNDOCUMENTED_BEHAVIOR] == 1
        assert SPEC_MATCH not in nf

    def test_edge_case_category(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        ec = s.by_category["edge_case"]
        # tc_005 = UNDOCUMENTED, tc_006 = UNDOCUMENTED
        assert ec[UNDOCUMENTED_BEHAVIOR] == 2

    def test_invalid_input_category(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        ii = s.by_category["invalid_input"]
        assert ii[SPEC_MATCH] == 1

    def test_unknown_category_handled(self):
        """A test case with no category in the fixture should not crash."""
        tcs = [TestCase(
            id="tc_x", category="mystery_category",
            description="x", path_params={}, query_params={},
            headers={}, request_body=None, expected_status=200,
        )]
        vrs = [_vr("tc_x", SPEC_MATCH)]
        s = build_summary(tcs, vrs)
        assert "mystery_category" in s.by_category


# ── LLM accuracy ─────────────────────────────────────────────────────────────


class TestLLMAccuracy:
    def test_matched_count(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        # tc_001 (matched), tc_002 (matched), tc_003 (matched), tc_006 (matched) = 4
        assert s.llm_matched_count == 4

    def test_checkable_count_excludes_skipped_and_error(self):
        tcs = [_tc("tc_a"), _tc("tc_b"), _tc("tc_c")]
        vrs = [
            _vr("tc_a", SPEC_MATCH,      llm_matched=True,  actual=200),
            _vr("tc_b", SKIPPED,         llm_matched=False, actual=None),
            _vr("tc_c", EXECUTION_ERROR, llm_matched=False, actual=None),
        ]
        s = build_summary(tcs, vrs)
        assert s.llm_checkable_count == 1  # only tc_a
        assert s.llm_matched_count == 1
        assert s.llm_accuracy_pct == 100.0

    def test_accuracy_pct_calculation(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        # 4 matched out of 6 checkable = 66.7%
        assert s.llm_checkable_count == 6
        assert s.llm_accuracy_pct == 66.7

    def test_zero_checkable_gives_zero_pct_no_crash(self):
        tcs = [_tc("tc_a")]
        vrs = [_vr("tc_a", SKIPPED, actual=None)]
        s = build_summary(tcs, vrs)
        assert s.llm_checkable_count == 0
        assert s.llm_accuracy_pct == 0.0

    def test_perfect_accuracy(self):
        tcs = [_tc("tc_1", expected=200), _tc("tc_2", expected=404)]
        vrs = [
            _vr("tc_1", SPEC_MATCH, llm_matched=True,  actual=200, llm_expected=200),
            _vr("tc_2", SPEC_MATCH, llm_matched=True,  actual=404, llm_expected=404),
        ]
        s = build_summary(tcs, vrs)
        assert s.llm_accuracy_pct == 100.0

    def test_zero_accuracy(self):
        tcs = [_tc("tc_1"), _tc("tc_2")]
        vrs = [
            _vr("tc_1", SPEC_MATCH, llm_matched=False),
            _vr("tc_2", SPEC_MATCH, llm_matched=False),
        ]
        s = build_summary(tcs, vrs)
        assert s.llm_accuracy_pct == 0.0


# ── Spec violation IDs ────────────────────────────────────────────────────────


class TestSpecViolationIds:
    def test_one_violation_collected(self):
        s = build_summary(_TEST_CASES, _VALIDATIONS)
        assert s.spec_violation_ids == ["tc_002"]

    def test_no_violations_gives_empty_list(self):
        tcs = [_tc("tc_a"), _tc("tc_b")]
        vrs = [
            _vr("tc_a", SPEC_MATCH),
            _vr("tc_b", UNDOCUMENTED_BEHAVIOR),
        ]
        s = build_summary(tcs, vrs)
        assert s.spec_violation_ids == []

    def test_multiple_violations_all_collected(self):
        tcs = [_tc("tc_1"), _tc("tc_2"), _tc("tc_3")]
        vrs = [
            _vr("tc_1", SPEC_VIOLATION, schema_errors=["err1"]),
            _vr("tc_2", SPEC_MATCH),
            _vr("tc_3", SPEC_VIOLATION, schema_errors=["err2"]),
        ]
        s = build_summary(tcs, vrs)
        assert set(s.spec_violation_ids) == {"tc_1", "tc_3"}

    def test_undocumented_behavior_not_in_violations(self):
        """UNDOCUMENTED_BEHAVIOR is not a contract break -- should NOT appear."""
        tcs = [_tc("tc_x")]
        vrs = [_vr("tc_x", UNDOCUMENTED_BEHAVIOR, actual=500)]
        s = build_summary(tcs, vrs)
        assert s.spec_violation_ids == []


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_run(self):
        s = build_summary([], [])
        assert s.total_cases == 0
        assert s.outcome_counts == {}
        assert s.by_category == {}
        assert s.llm_accuracy_pct == 0.0
        assert s.spec_violation_ids == []

    def test_all_skipped_run(self):
        tcs = [_tc(f"tc_{i}", "happy_path") for i in range(3)]
        vrs = [_vr(f"tc_{i}", SKIPPED, actual=None) for i in range(3)]
        s = build_summary(tcs, vrs)
        assert s.outcome_counts[SKIPPED] == 3
        assert s.llm_checkable_count == 0
        assert s.llm_accuracy_pct == 0.0
        assert s.spec_violation_ids == []

    def test_all_violations_run(self):
        tcs = [_tc("tc_1"), _tc("tc_2")]
        vrs = [
            _vr("tc_1", SPEC_VIOLATION, schema_errors=["e1"]),
            _vr("tc_2", SPEC_VIOLATION, schema_errors=["e2"]),
        ]
        s = build_summary(tcs, vrs)
        assert s.outcome_counts[SPEC_VIOLATION] == 2
        assert s.outcome_pct[SPEC_VIOLATION] == 100.0
        assert len(s.spec_violation_ids) == 2
