"""
groundtruth/spec_parser.py  —  Module 1
────────────────────────────────────────
Parses an OpenAPI 3.x spec (JSON or YAML) into structured Python dataclasses.

INPUT:  Path to a .json / .yaml / .yml file
OUTPUT: EndpointDefinition  (or a list of them via list_endpoints)

No LLM is involved here — this is pure deterministic parsing.
The output of this module is the ground-truth schema that Module 4 will
use as the authoritative pass/fail reference. It must be correct.

Public API
──────────
    load_spec(path)                         -> dict
    list_endpoints(spec)                    -> List[Tuple[str, str]]
    parse_endpoint(spec, path, method)      -> EndpointDefinition
    SpecParseError                          (exception class)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from groundtruth.models import (
    EndpointDefinition,
    ParameterDef,
    RequestBodyDef,
    ResponseDef,
)

# ── Exceptions ────────────────────────────────────────────────────────────────


class SpecParseError(Exception):
    """Raised when the spec file cannot be loaded or parsed."""


# ── Internal helpers ──────────────────────────────────────────────────────────

_HTTP_METHODS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
)


def _resolve_ref(spec: dict, ref: str) -> dict:
    """
    Resolve a JSON Reference like '#/components/schemas/Pet'.

    Only internal (same-document) $refs are supported. External file
    or URL refs will raise SpecParseError — they are out of scope for
    this tool.
    """
    if not ref.startswith("#/"):
        raise SpecParseError(
            f"External $ref not supported: '{ref}'. "
            "Only same-document #/ references are handled."
        )
    parts = ref.lstrip("#/").split("/")
    node: Any = spec
    for part in parts:
        # JSON Pointer spec (RFC 6901) requires unescaping ~1 → / and ~0 → ~
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SpecParseError(
                f"$ref path segment '{part}' not found while resolving '{ref}'"
            )
        node = node[part]
    return node  # type: ignore[return-value]


def _deref_schema(
    spec: dict,
    node: Any,
    _visiting: frozenset = frozenset(),
) -> Any:
    """
    Recursively replace every {"$ref": "#/..."} in a schema tree with its
    fully resolved, inline definition.

    After this function returns, the result contains ZERO remaining $ref keys
    anywhere in the tree — every reference has been expanded inline.

    Circular reference guard:
        We track the set of $ref strings currently on the call stack in
        `_visiting` (a frozenset — immutable, so each branch gets its own
        independent copy). If we encounter a ref already being resolved on
        the current path, we return a stub dict instead of recursing.
        This prevents infinite recursion on schemas like:
            Pet -> { category: { $ref: Category } -> { pet: { $ref: Pet } } }

    Args:
        spec:      The full raw spec dict (read-only, used for ref lookups).
        node:      The current schema node being processed.
        _visiting: Internal — the set of $ref paths on the current call stack.
                   Callers outside this module should not pass this argument.

    Returns:
        A fully dereferenced copy of `node` — no $ref keys remain.
    """
    if isinstance(node, list):
        return [_deref_schema(spec, item, _visiting) for item in node]

    if not isinstance(node, dict):
        # Scalar values (str, int, bool, None) — nothing to dereference
        return node

    if "$ref" in node:
        ref = node["$ref"]

        # Circular reference on the current resolution path — return a stub
        # so the caller can continue validating the rest of the schema.
        if ref in _visiting:
            return {"description": f"(circular ref: {ref})"}

        resolved = _resolve_ref(spec, ref)
        # Expand the resolved target, adding this ref to the visited set
        return _deref_schema(spec, resolved, _visiting | {ref})

    # No $ref at this level — recurse into every value in the dict.
    # We recurse into ALL keys unconditionally so we never miss a nested
    # $ref regardless of the keyword (properties, items, allOf, anyOf,
    # oneOf, not, if, then, else, additionalProperties, patternProperties,
    # $defs, definitions, or any future extension keyword).
    return {
        key: _deref_schema(spec, value, _visiting)
        for key, value in node.items()
    }


def _merge_parameters(
    spec: dict,
    path_level: List[Any],
    op_level: List[Any],
) -> List[ParameterDef]:
    """
    Merge path-level and operation-level parameter lists.

    Per OpenAPI 3 rules, operation-level parameters override path-level
    parameters with the same (name, in) pair.
    """
    seen: Dict[Tuple[str, str], ParameterDef] = {}

    for raw in path_level + op_level:
        if "$ref" in raw:
            raw = _resolve_ref(spec, raw["$ref"])

        param_schema = raw.get("schema", {})
        param_schema = _deref_schema(spec, param_schema)

        key = (raw["name"], raw["in"])
        seen[key] = ParameterDef(
            name=raw["name"],
            location=raw["in"],
            required=raw.get("required", raw["in"] == "path"),  # path params are always required
            schema=param_schema,
        )

    return list(seen.values())


def _parse_request_body(spec: dict, raw_body: dict) -> Optional[RequestBodyDef]:
    """Extract and resolve the request body definition."""
    if "$ref" in raw_body:
        raw_body = _resolve_ref(spec, raw_body["$ref"])

    content = raw_body.get("content", {})
    if not content:
        return None

    # Prefer application/json; fall back to the first declared content type
    content_type = (
        "application/json" if "application/json" in content else next(iter(content))
    )
    media_type = content[content_type]
    raw_schema = media_type.get("schema", {})
    resolved_schema = _deref_schema(spec, raw_schema) if raw_schema else {}

    return RequestBodyDef(
        required=raw_body.get("required", False),
        content_type=content_type,
        schema=resolved_schema,
    )


def _parse_responses(spec: dict, raw_responses: dict) -> Dict[str, ResponseDef]:
    """Parse the responses map, resolving $refs and extracting schemas."""
    result: Dict[str, ResponseDef] = {}

    for status_code, raw_resp in raw_responses.items():
        if "$ref" in raw_resp:
            raw_resp = _resolve_ref(spec, raw_resp["$ref"])

        resp_schema: Optional[dict] = None
        content = raw_resp.get("content", {})
        if content:
            ct = (
                "application/json"
                if "application/json" in content
                else next(iter(content))
            )
            raw_schema = content[ct].get("schema", {})
            if raw_schema:
                resp_schema = _deref_schema(spec, raw_schema)

        result[str(status_code)] = ResponseDef(
            description=raw_resp.get("description", ""),
            schema=resp_schema,
        )

    return result


# ── Public API ────────────────────────────────────────────────────────────────


def load_spec(path: str) -> dict:
    """
    Load an OpenAPI 3.x spec from a JSON or YAML file.

    Returns the raw spec dict. Does not validate the full spec; only
    checks that the file parses and contains the 'openapi' top-level key.

    Raises:
        SpecParseError: if the file is missing, unreadable, malformed,
                        or does not appear to be an OpenAPI 3.x document.
    """
    p = Path(path)
    if not p.exists():
        raise SpecParseError(f"Spec file not found: {path!r}")
    if not p.is_file():
        raise SpecParseError(f"Path is not a file: {path!r}")

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecParseError(f"Cannot read spec file: {e}") from e

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            spec = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise SpecParseError(f"YAML parse error in {path!r}: {e}") from e
    elif suffix == ".json":
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SpecParseError(f"JSON parse error in {path!r}: {e}") from e
    else:
        raise SpecParseError(
            f"Unsupported file extension {p.suffix!r}. "
            "Expected .yaml, .yml, or .json"
        )

    if not isinstance(spec, dict):
        raise SpecParseError("Spec must be a JSON/YAML object at the top level.")

    if "openapi" not in spec:
        raise SpecParseError(
            "File does not appear to be an OpenAPI 3.x spec "
            "(missing top-level 'openapi' key)."
        )

    version: str = spec.get("openapi", "")
    if not version.startswith("3"):
        raise SpecParseError(
            f"Only OpenAPI 3.x is supported; found version: {version!r}"
        )

    return spec


def list_endpoints(spec: dict) -> List[Tuple[str, str]]:
    """
    Return every (path, method) pair defined in the spec.

    Useful for discovering what's available or for printing a help menu
    when the user's --endpoint argument doesn't match anything.
    """
    result: List[Tuple[str, str]] = []
    for path, path_item in spec.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method in _HTTP_METHODS:
            if method in path_item:
                result.append((path, method))
    return sorted(result)


def parse_endpoint(spec: dict, path: str, method: str) -> EndpointDefinition:
    """
    Parse a single endpoint from the spec into an EndpointDefinition.

    The returned object is the authoritative schema reference for this
    endpoint. Module 4 will use `endpoint.responses` to validate HTTP
    responses — it is not re-parsed from the spec again at that stage.

    Args:
        spec:   The raw spec dict returned by load_spec().
        path:   The path template exactly as written in the spec,
                e.g. '/pet/{petId}'.
        method: HTTP method string, case-insensitive, e.g. 'GET' or 'get'.

    Raises:
        SpecParseError: if the path or method is not found in the spec.
    """
    paths: dict = spec.get("paths", {})

    if path not in paths:
        raise SpecParseError(
            f"Path '{path}' not found in spec. "
            f"Run list_endpoints() to see all available paths."
        )

    path_item: dict = paths[path]
    method_lower = method.lower()

    if method_lower not in path_item:
        available = [m for m in _HTTP_METHODS if m in path_item]
        raise SpecParseError(
            f"Method '{method.upper()}' not found for path '{path}'. "
            f"Available methods: {[m.upper() for m in available]}"
        )

    operation: dict = path_item[method_lower]
    if "$ref" in operation:
        operation = _resolve_ref(spec, operation["$ref"])

    parameters = _merge_parameters(
        spec,
        path_level=list(path_item.get("parameters", [])),
        op_level=list(operation.get("parameters", [])),
    )

    raw_body = operation.get("requestBody")
    request_body = _parse_request_body(spec, raw_body) if raw_body else None

    responses = _parse_responses(spec, operation.get("responses", {}))

    return EndpointDefinition(
        path=path,
        method=method_lower,
        parameters=parameters,
        request_body=request_body,
        responses=responses,
        operation_id=operation.get("operationId"),
        summary=operation.get("summary"),
    )
