"""
groundtruth/report.py  --  Module 5
--------------------------------------
Aggregates ValidationResults into a structured report.

INPUT:  List[TestCase]          (from Module 2, for category labels)
        List[ExecutionResult]   (from Module 3, for timing/status facts)
        List[ValidationResult]  (from Module 4, the authoritative verdicts)
        EndpointDefinition      (from Module 1, for endpoint metadata)

OUTPUT: ReportSummary           (pure aggregated numbers, no logic)
SIDE EFFECTS:
        print_console_report()  -- prints to stdout
        write_json_report()     -- writes a .json file if path given

Core principle preserved:
    This module never re-evaluates correctness. It only counts and
    formats what Module 4 already decided.

Public API
----------
    build_summary(test_cases, validations)                  -> ReportSummary
    print_console_report(summary, endpoint, validations,
                         test_cases)                        -> None
    write_json_report(path, endpoint, test_cases,
                      results, validations, summary)        -> None
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from groundtruth import __version__
from groundtruth.models import (
    EXECUTION_ERROR,
    SKIPPED,
    SPEC_MATCH,
    SPEC_VIOLATION,
    UNDOCUMENTED_BEHAVIOR,
    EndpointDefinition,
    ExecutionResult,
    ReportSummary,
    TestCase,
    ValidationResult,
)

# Ordered list used for consistent display across all output surfaces
_OUTCOME_ORDER = [
    SPEC_MATCH,
    SPEC_VIOLATION,
    UNDOCUMENTED_BEHAVIOR,
    SKIPPED,
    EXECUTION_ERROR,
]

_CATEGORY_ORDER = ["happy_path", "invalid_input", "not_found", "edge_case"]


# ── Pure aggregation (no I/O) ─────────────────────────────────────────────────


def build_summary(
    test_cases: List[TestCase],
    validations: List[ValidationResult],
) -> ReportSummary:
    """
    Compute all report statistics from test cases and their validation results.

    This function is pure — no printing, no file I/O. It is the only place
    in Module 5 where numbers are computed, making it the easiest piece to
    unit-test.

    Args:
        test_cases:   The LLM-generated test cases (for category labels).
        validations:  The Module 4 verdicts (one per test case, same order).

    Returns:
        A fully populated ReportSummary.
    """
    total = len(validations)

    # ── Outcome counts ────────────────────────────────────────────────────────
    outcome_counts: Dict[str, int] = {}
    for vr in validations:
        outcome_counts[vr.outcome] = outcome_counts.get(vr.outcome, 0) + 1

    outcome_pct: Dict[str, float] = {
        label: round(count / total * 100, 1) if total else 0.0
        for label, count in outcome_counts.items()
    }

    # ── Category x Outcome cross-table ────────────────────────────────────────
    # Build a lookup so we can join validations to their test case category
    tc_category: Dict[str, str] = {tc.id: tc.category for tc in test_cases}

    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for vr in validations:
        category = tc_category.get(vr.test_case_id, "unknown")
        by_category[category][vr.outcome] += 1

    # Convert nested defaultdicts to plain dicts for clean serialisation
    by_category_plain: Dict[str, Dict[str, int]] = {
        cat: dict(outcomes) for cat, outcomes in by_category.items()
    }

    # ── LLM prediction accuracy ───────────────────────────────────────────────
    # Only count cases that actually produced an HTTP response
    checkable = [
        vr for vr in validations
        if vr.outcome not in (SKIPPED, EXECUTION_ERROR)
    ]
    matched = [vr for vr in checkable if vr.llm_prediction_matched]

    llm_checkable_count = len(checkable)
    llm_matched_count = len(matched)
    llm_accuracy_pct = (
        round(llm_matched_count / llm_checkable_count * 100, 1)
        if llm_checkable_count
        else 0.0
    )

    # ── Spec violations ───────────────────────────────────────────────────────
    spec_violation_ids = [
        vr.test_case_id for vr in validations if vr.outcome == SPEC_VIOLATION
    ]

    return ReportSummary(
        total_cases=total,
        outcome_counts=outcome_counts,
        outcome_pct=outcome_pct,
        by_category=by_category_plain,
        llm_matched_count=llm_matched_count,
        llm_checkable_count=llm_checkable_count,
        llm_accuracy_pct=llm_accuracy_pct,
        spec_violation_ids=spec_violation_ids,
    )


# ── Console report ────────────────────────────────────────────────────────────

_WIDTH = 60
_SEP = "=" * _WIDTH
_THIN = "-" * _WIDTH


def _pct_bar(count: int, total: int, width: int = 20) -> str:
    """ASCII progress bar, e.g. '[####................]  4 / 6'"""
    if total == 0:
        filled = 0
    else:
        filled = round(count / total * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}]"


def print_console_report(
    summary: ReportSummary,
    endpoint: EndpointDefinition,
    validations: List[ValidationResult],
    test_cases: List[TestCase],
) -> None:
    """
    Print a structured, human-readable summary to stdout.

    Sections:
      1. Header
      2. Outcome breakdown (counts + ASCII bar)
      3. By-category cross-table
      4. LLM prediction accuracy (marked informational)
      5. Spec violations (prominent if any, 'none detected' otherwise)
    """
    tc_by_id: Dict[str, TestCase] = {tc.id: tc for tc in test_cases}
    vr_by_id: Dict[str, ValidationResult] = {vr.test_case_id: vr for vr in validations}

    print()
    print(_SEP)
    print(f"  GROUNDTRUTH REPORT  --  {endpoint.method.upper()} {endpoint.path}")
    print(_SEP)
    print()
    print(f"  {summary.total_cases} test case(s) run")
    print()

    # ── 1. Outcome breakdown ─────────────────────────────────────────────────
    print("  OUTCOME BREAKDOWN")
    print(f"  {_THIN[:40]}")

    outcome_labels = {
        SPEC_MATCH:            "SPEC_MATCH           ",
        SPEC_VIOLATION:        "SPEC_VIOLATION       ",
        UNDOCUMENTED_BEHAVIOR: "UNDOCUMENTED_BEHAVIOR",
        SKIPPED:               "SKIPPED              ",
        EXECUTION_ERROR:       "EXECUTION_ERROR      ",
    }

    for label in _OUTCOME_ORDER:
        count = summary.outcome_counts.get(label, 0)
        pct = summary.outcome_pct.get(label, 0.0)
        bar = _pct_bar(count, summary.total_cases)
        tag = outcome_labels[label]
        print(f"  {tag}  {bar}  {count:3d}  ({pct:5.1f}%)")

    print()

    # ── 2. By-category cross-table ───────────────────────────────────────────
    print("  BY CATEGORY  x  OUTCOME")
    print(f"  {_THIN[:40]}")

    all_categories = list(dict.fromkeys(
        _CATEGORY_ORDER + [c for c in summary.by_category if c not in _CATEGORY_ORDER]
    ))

    for cat in all_categories:
        outcomes = summary.by_category.get(cat)
        if not outcomes:
            continue
        parts = []
        for label in _OUTCOME_ORDER:
            n = outcomes.get(label, 0)
            if n:
                # Short label for compact display
                short = {
                    SPEC_MATCH: "MATCH",
                    SPEC_VIOLATION: "VIOLATION",
                    UNDOCUMENTED_BEHAVIOR: "UNDOC",
                    SKIPPED: "SKIP",
                    EXECUTION_ERROR: "ERR",
                }[label]
                parts.append(f"{n} {short}")
        print(f"  {cat:<16s}  {', '.join(parts) if parts else '(no results)'}")

    print()

    # ── 3. LLM prediction accuracy ───────────────────────────────────────────
    print("  LLM PREDICTION ACCURACY  (informational -- does not affect outcomes)")
    print(f"  {_THIN[:40]}")
    if summary.llm_checkable_count:
        bar = _pct_bar(summary.llm_matched_count, summary.llm_checkable_count)
        print(
            f"  {bar}  "
            f"{summary.llm_matched_count} / {summary.llm_checkable_count} "
            f"predictions matched  ({summary.llm_accuracy_pct:.1f}%)"
        )
    else:
        print("  (no checkable cases -- all were skipped or errored)")
    print()

    # ── 4. Spec violations ───────────────────────────────────────────────────
    print("  SPEC VIOLATIONS  (contract breaks: documented code, wrong body)")
    print(f"  {_THIN[:40]}")

    if not summary.spec_violation_ids:
        print("  None detected.")
    else:
        print(f"  !! {len(summary.spec_violation_ids)} SPEC VIOLATION(S) FOUND !!")
        print()
        for tc_id in summary.spec_violation_ids:
            vr = vr_by_id.get(tc_id)
            tc = tc_by_id.get(tc_id)
            if not vr or not tc:
                continue
            print(f"  {tc_id}  [{tc.category}]  {tc.description}")
            print(f"    Status: {vr.actual_status_code}  --  {vr.reason}")
            if vr.schema_errors:
                for err in vr.schema_errors:
                    print(f"    - {err}")
            print()

    print()
    print(_SEP)
    print()


# ── JSON export ───────────────────────────────────────────────────────────────


def write_json_report(
    path: str,
    endpoint: EndpointDefinition,
    test_cases: List[TestCase],
    results: List[ExecutionResult],
    validations: List[ValidationResult],
    summary: ReportSummary,
) -> None:
    """
    Write a complete, machine-readable JSON report to `path`.

    The file contains every input and output from the run in one place:
    endpoint metadata, each TestCase, each ExecutionResult, each
    ValidationResult, and the aggregated ReportSummary. It is designed
    to be the single replayable record of a run.

    Args:
        path:        Filesystem path to write (created or overwritten).
        endpoint:    The parsed endpoint definition (Module 1 output).
        test_cases:  LLM-generated test cases (Module 2 output).
        results:     HTTP execution results (Module 3 output).
        validations: Validation verdicts (Module 4 output).
        summary:     Aggregated statistics (Module 5, from build_summary).
    """
    payload = {
        "groundtruth_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": {
            "path": endpoint.path,
            "method": endpoint.method,
            "operation_id": endpoint.operation_id,
            "summary": endpoint.summary,
        },
        "summary": dataclasses.asdict(summary),
        "test_cases": [dataclasses.asdict(tc) for tc in test_cases],
        "execution_results": [dataclasses.asdict(r) for r in results],
        "validation_results": [dataclasses.asdict(vr) for vr in validations],
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
