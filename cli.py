#!/usr/bin/env python3
"""
cli.py  —  GroundTruth entry point (Modules 1–3)
─────────────────────────────────────────────────
Usage:
    python cli.py --spec <path> --base-url <url> --endpoint <path> \\
                  [--method <method>] [--allow-mutations] [--model <groq-model>]

Example (public Petstore API):
    python cli.py \\
        --spec tests/fixtures/petstore.yaml \\
        --base-url https://petstore3.swagger.io/api/v3 \\
        --endpoint /pet/{petId} \\
        --method get

The CLI wires Modules 1 → 2 → 3 in sequence and prints:
  - A human-readable progress log to stdout
  - A final JSON dump of all raw ExecutionResult objects

GROQ_API_KEY must be set in the environment or in a .env file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from dotenv import load_dotenv

from groundtruth.executor import execute
from groundtruth.spec_parser import SpecParseError, list_endpoints, load_spec, parse_endpoint
from groundtruth.test_generator import GeneratorError, generate_test_cases

# ── ANSI colour helpers (gracefully degrade on Windows without ANSI support) ──

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(t: str) -> str:
    return _c("32", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _red(t: str) -> str:
    return _c("31", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


DIVIDER = "─" * 60


# ── Argument parsing ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="groundtruth",
        description=(
            "GroundTruth — LLM-assisted API tester.\n"
            "The LLM proposes test inputs. The OpenAPI spec decides correctness."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --spec openapi.yaml --base-url http://localhost:8000"
            " --endpoint /users/{id}\n"
            "  python cli.py --spec openapi.json --base-url https://api.example.com"
            " --endpoint /orders --method post --allow-mutations"
        ),
    )
    p.add_argument("--spec", required=True, metavar="PATH",
                   help="Path to the OpenAPI 3.x spec file (.json, .yaml, or .yml)")
    p.add_argument("--base-url", required=True, metavar="URL",
                   help="Base URL of the running API (no trailing slash needed)")
    p.add_argument("--endpoint", required=True, metavar="PATH",
                   help="Endpoint path to test, exactly as written in the spec "
                        "(e.g. /pet/{petId})")
    p.add_argument("--method", default="get", metavar="METHOD",
                   help="HTTP method (default: get)")
    p.add_argument("--allow-mutations", action="store_true",
                   help="Allow POST, PUT, PATCH, DELETE requests. "
                        "Omit this flag for safe read-only testing.")
    p.add_argument("--model", default="openai/gpt-oss-120b", metavar="MODEL",
                   help="Groq model to use for test generation "
                        "(default: openai/gpt-oss-120b)")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print(
            _red("[error]") + " GROQ_API_KEY is not set.\n"
            "  Add it to a .env file (see .env.example) or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── MODULE 1: Spec Parser ─────────────────────────────────────────────────

    print(f"\n{_bold('MODULE 1 — Spec Parser')}")
    print(DIVIDER)

    try:
        spec = load_spec(args.spec)
    except SpecParseError as e:
        print(_red(f"[error] {e}"), file=sys.stderr)
        sys.exit(1)

    all_endpoints = list_endpoints(spec)

    print(f"  Spec loaded   : {args.spec}")
    print(f"  OpenAPI       : {spec.get('openapi', '?')}")
    print(f"  Title         : {spec.get('info', {}).get('title', '(none)')}")
    print(f"  Endpoints     : {len(all_endpoints)} total")

    try:
        endpoint = parse_endpoint(spec, args.endpoint, args.method)
    except SpecParseError as e:
        print(_red(f"\n[error] {e}"), file=sys.stderr)
        print("\n  Available endpoints:", file=sys.stderr)
        for ep_path, ep_method in all_endpoints:
            print(f"    {ep_method.upper():7s} {ep_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  {_bold('Endpoint')}     : {endpoint.method.upper()} {endpoint.path}")
    if endpoint.summary:
        print(f"  Summary       : {endpoint.summary}")
    if endpoint.operation_id:
        print(f"  Operation ID  : {endpoint.operation_id}")

    print(f"\n  Parameters ({len(endpoint.parameters)}):")
    if endpoint.parameters:
        for param in endpoint.parameters:
            req_tag = _yellow("required") if param.required else _dim("optional")
            type_hint = param.schema.get("type", "?")
            print(f"    [{param.location:6s}] {param.name:20s} {req_tag}  type={type_hint}")
    else:
        print("    (none)")

    if endpoint.request_body:
        rb = endpoint.request_body
        req_tag = _yellow("required") if rb.required else _dim("optional")
        print(f"\n  Request body  : {rb.content_type} ({req_tag})")

    print(f"\n  Documented responses:")
    for code, resp in endpoint.responses.items():
        has_schema = "✓ schema" if resp.schema else "no body"
        print(f"    {code}  {resp.description[:50]:50s}  [{has_schema}]")

    # ── MODULE 2: Test Case Generator ────────────────────────────────────────

    print(f"\n{_bold('MODULE 2 — Test Case Generator (LLM)')}")
    print(DIVIDER)
    print(f"  Model         : {args.model}")
    print(f"  Calling Groq API...", end=" ", flush=True)

    try:
        test_cases = generate_test_cases(endpoint, api_key, model=args.model)
    except GeneratorError as e:
        print(_red("FAILED"))
        print(_red(f"\n[error] {e}"), file=sys.stderr)
        sys.exit(1)

    print(_green(f"OK — {len(test_cases)} test cases generated"))
    print()

    category_order = ["happy_path", "invalid_input", "not_found", "edge_case"]
    category_colors = {
        "happy_path": _green,
        "invalid_input": _yellow,
        "not_found": _yellow,
        "edge_case": _dim,
    }

    for tc in test_cases:
        color = category_colors.get(tc.category, str)
        cat_label = color(f"[{tc.category}]")
        print(f"  {tc.id}  {cat_label}")
        print(f"    {_dim(tc.description)}")
        print(f"    path_params    = {json.dumps(tc.path_params)}")
        if tc.query_params:
            print(f"    query_params   = {json.dumps(tc.query_params)}")
        if tc.request_body is not None:
            print(f"    request_body   = {json.dumps(tc.request_body)}")
        print(f"    expected_status= {tc.expected_status}  {_dim('(LLM guess — not authoritative)')}")
        print()

    # ── MODULE 3: HTTP Executor ───────────────────────────────────────────────

    print(f"{_bold('MODULE 3 — HTTP Executor')}")
    print(DIVIDER)
    print(f"  Base URL      : {args.base_url}")
    print(f"  Mutations     : {'ENABLED' if args.allow_mutations else _yellow('disabled (pass --allow-mutations to enable)')}")
    print()

    results = []
    for tc in test_cases:
        color = category_colors.get(tc.category, str)
        print(f"  → {_bold(tc.id)}  {color(tc.category)}  —  {tc.description}")

        result = execute(
            test_case=tc,
            endpoint=endpoint,
            base_url=args.base_url,
            allow_mutations=args.allow_mutations,
        )
        results.append(result)

        if result.skipped:
            print(f"    {_yellow('⊘ SKIPPED')}  {result.skip_reason}")
        elif result.error:
            print(f"    {_red('✗ ERROR')}    {result.error}")
        else:
            status_str = str(result.status_code)
            if result.status_code and result.status_code < 300:
                status_str = _green(status_str)
            elif result.status_code and result.status_code < 500:
                status_str = _yellow(status_str)
            else:
                status_str = _red(status_str)

            print(f"    {_green('✓')} HTTP {status_str}  in {result.response_time_ms}ms")

            body_str = json.dumps(result.response_body)
            if len(body_str) > 140:
                body_str = body_str[:140] + "..."
            print(f"    {_dim('body:')} {body_str}")

        print()

    # ── Raw JSON dump ─────────────────────────────────────────────────────────

    print(f"{_bold('RAW EXECUTION RESULTS')}")
    print(DIVIDER)
    output = [dataclasses.asdict(r) for r in results]
    print(json.dumps(output, indent=2))

    # ── MODULE 4: Schema Validator ────────────────────────────────────────────

    from groundtruth.validator import validate  # local import — keeps modules independent

    print(f"\n{_bold('MODULE 4 — Schema Validator')}")
    print(DIVIDER)
    print(f"  {_dim('Outcome is determined by the OpenAPI spec — never by the LLM guess.')}")
    print()

    # Outcome label → (colour fn, icon)
    _outcome_style = {
        "SPEC_MATCH":             (_green,  "✓"),
        "SPEC_VIOLATION":         (_red,    "✗"),
        "UNDOCUMENTED_BEHAVIOR":  (_yellow, "⚠"),
        "SKIPPED":                (_yellow, "⊘"),
        "EXECUTION_ERROR":        (_red,    "✗"),
    }

    # Build a quick lookup: test_case_id → TestCase
    tc_by_id = {tc.id: tc for tc in test_cases}

    validations = []
    for result in results:
        tc = tc_by_id[result.test_case_id]
        vr = validate(endpoint, tc, result)
        validations.append(vr)

        color_fn, icon = _outcome_style.get(vr.outcome, (str, "?"))
        outcome_str = color_fn(f"{icon} {vr.outcome}")

        llm_tag = (
            _dim("(LLM guess matched ✓)")
            if vr.llm_prediction_matched
            else _dim(f"(LLM guessed {vr.llm_expected_status}, got {vr.actual_status_code})")
        )

        print(f"  {_bold(result.test_case_id)}  {outcome_str}  {llm_tag}")
        print(f"    {_dim(vr.reason)}")

        if vr.schema_errors:
            print(f"    {_red('Schema errors:')}")
            for err in vr.schema_errors:
                print(f"      • {err}")

        print()

    # Summary counts
    from collections import Counter
    counts = Counter(vr.outcome for vr in validations)
    print(DIVIDER)
    for label in ["SPEC_MATCH", "SPEC_VIOLATION", "UNDOCUMENTED_BEHAVIOR", "SKIPPED", "EXECUTION_ERROR"]:
        n = counts.get(label, 0)
        if n == 0:
            continue
        color_fn, icon = _outcome_style[label]
        print(f"  {color_fn(f'{icon} {label}'):30s}  {n}")
    print()

    # Final JSON dump (both execution and validation results together)
    print(f"{_bold('RAW VALIDATION RESULTS (JSON)')}")
    print(DIVIDER)
    print(json.dumps([dataclasses.asdict(vr) for vr in validations], indent=2))


if __name__ == "__main__":
    main()

