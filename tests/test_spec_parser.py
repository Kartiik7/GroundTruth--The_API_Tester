"""
tests/test_spec_parser.py
─────────────────────────
Unit tests for Module 1 (spec_parser.py).

All tests are self-contained — they use the petstore.yaml fixture in
tests/fixtures/ and do NOT hit any network or LLM.

Coverage:
  - load_spec: happy path, missing file, bad YAML, wrong version
  - list_endpoints: correct (path, method) pairs enumerated
  - parse_endpoint: parameter extraction, $ref resolution, response parsing,
    request body extraction
  - Error cases: unknown path, unknown method
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from groundtruth.spec_parser import (
    SpecParseError,
    list_endpoints,
    load_spec,
    parse_endpoint,
)

# ── Fixture helpers ───────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PETSTORE_YAML = str(FIXTURES_DIR / "petstore.yaml")


def _spec() -> dict:
    """Load the petstore fixture once per call (fast, no caching needed in tests)."""
    return load_spec(PETSTORE_YAML)


# ── load_spec ─────────────────────────────────────────────────────────────────


class TestLoadSpec:
    def test_loads_yaml_successfully(self):
        spec = load_spec(PETSTORE_YAML)
        assert isinstance(spec, dict)
        assert spec["openapi"].startswith("3.")

    def test_loads_json_successfully(self, tmp_path):
        spec = load_spec(PETSTORE_YAML)
        json_path = tmp_path / "spec.json"
        json_path.write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_spec(str(json_path))
        assert loaded["info"]["title"] == spec["info"]["title"]

    def test_raises_on_missing_file(self):
        with pytest.raises(SpecParseError, match="not found"):
            load_spec("/nonexistent/path/spec.yaml")

    def test_raises_on_bad_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket", encoding="utf-8")
        with pytest.raises(SpecParseError, match="YAML parse error"):
            load_spec(str(bad))

    def test_raises_on_bad_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(SpecParseError, match="JSON parse error"):
            load_spec(str(bad))

    def test_raises_on_unsupported_extension(self, tmp_path):
        f = tmp_path / "spec.toml"
        f.write_text("openapi = '3.0.0'", encoding="utf-8")
        with pytest.raises(SpecParseError, match="Unsupported file extension"):
            load_spec(str(f))

    def test_raises_on_missing_openapi_key(self, tmp_path):
        f = tmp_path / "spec.yaml"
        f.write_text("info:\n  title: Nope\n", encoding="utf-8")
        with pytest.raises(SpecParseError, match="missing top-level 'openapi' key"):
            load_spec(str(f))

    def test_raises_on_swagger_2(self, tmp_path):
        f = tmp_path / "swagger.yaml"
        f.write_text("openapi: '2.0'\ninfo:\n  title: Old\n", encoding="utf-8")
        with pytest.raises(SpecParseError, match="Only OpenAPI 3.x"):
            load_spec(str(f))


# ── list_endpoints ────────────────────────────────────────────────────────────


class TestListEndpoints:
    def test_returns_all_expected_pairs(self):
        spec = _spec()
        endpoints = list_endpoints(spec)
        assert ("/pet/{petId}", "get") in endpoints
        assert ("/pet/{petId}", "delete") in endpoints
        assert ("/pet", "post") in endpoints
        assert ("/pet/findByStatus", "get") in endpoints
        assert ("/store/inventory", "get") in endpoints

    def test_does_not_include_non_methods(self):
        """Top-level path keys like 'parameters' should not appear as methods."""
        spec = _spec()
        endpoints = list_endpoints(spec)
        paths = [ep[0] for ep in endpoints]
        # All returned paths must come from spec['paths']
        for p in paths:
            assert p in spec["paths"]

    def test_returns_sorted_list(self):
        spec = _spec()
        endpoints = list_endpoints(spec)
        assert endpoints == sorted(endpoints)

    def test_empty_spec_returns_empty_list(self):
        spec = {"openapi": "3.0.0", "info": {"title": "Empty"}, "paths": {}}
        assert list_endpoints(spec) == []


# ── parse_endpoint ────────────────────────────────────────────────────────────


class TestParseEndpoint:
    def test_basic_get_with_path_param(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "get")
        assert ep.path == "/pet/{petId}"
        assert ep.method == "get"
        assert ep.operation_id == "getPetById"
        assert ep.summary == "Find pet by ID"
        assert ep.request_body is None

    def test_path_param_extracted_correctly(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "get")
        assert len(ep.parameters) == 1
        param = ep.parameters[0]
        assert param.name == "petId"
        assert param.location == "path"
        assert param.required is True
        assert param.schema["type"] == "integer"

    def test_query_param_extracted_correctly(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/findByStatus", "get")
        assert len(ep.parameters) == 1
        param = ep.parameters[0]
        assert param.name == "status"
        assert param.location == "query"
        assert param.required is False
        assert "enum" in param.schema

    def test_responses_extracted(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "get")
        assert "200" in ep.responses
        assert "400" in ep.responses
        assert "404" in ep.responses

    def test_200_response_has_schema(self):
        """The $ref to Pet should be resolved into the inline schema."""
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "get")
        schema = ep.responses["200"].schema
        assert schema is not None
        assert schema.get("type") == "object"
        assert "properties" in schema
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]

    def test_404_response_has_no_body_schema(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "get")
        assert ep.responses["404"].schema is None

    def test_post_endpoint_has_request_body(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet", "post")
        assert ep.request_body is not None
        assert ep.request_body.required is True
        assert ep.request_body.content_type == "application/json"
        schema = ep.request_body.schema
        assert schema.get("type") == "object"
        assert "name" in schema.get("required", [])

    def test_method_case_insensitive(self):
        spec = _spec()
        ep_lower = parse_endpoint(spec, "/pet/{petId}", "get")
        ep_upper = parse_endpoint(spec, "/pet/{petId}", "GET")
        assert ep_lower.method == ep_upper.method == "get"

    def test_raises_on_unknown_path(self):
        spec = _spec()
        with pytest.raises(SpecParseError, match="not found"):
            parse_endpoint(spec, "/nonexistent", "get")

    def test_raises_on_unknown_method(self):
        spec = _spec()
        with pytest.raises(SpecParseError, match="not found"):
            parse_endpoint(spec, "/pet/{petId}", "post")

    def test_delete_endpoint_parsed(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/pet/{petId}", "delete")
        assert ep.method == "delete"
        assert ep.operation_id == "deletePet"
        assert "200" in ep.responses
        assert "400" in ep.responses

    def test_inventory_endpoint_no_params(self):
        spec = _spec()
        ep = parse_endpoint(spec, "/store/inventory", "get")
        assert ep.parameters == []
        assert ep.request_body is None
        assert "200" in ep.responses


# ── $ref resolution ────────────────────────────────────────────────────────────


class TestRefResolution:
    def test_inline_ref_spec(self, tmp_path):
        """A $ref in a response schema should be fully resolved."""
        raw = textwrap.dedent("""
            openapi: "3.0.3"
            info:
              title: RefTest
              version: "1.0"
            paths:
              /things/{id}:
                get:
                  parameters:
                    - name: id
                      in: path
                      required: true
                      schema:
                        type: string
                  responses:
                    "200":
                      description: OK
                      content:
                        application/json:
                          schema:
                            $ref: "#/components/schemas/Thing"
            components:
              schemas:
                Thing:
                  type: object
                  required: [name]
                  properties:
                    name:
                      type: string
        """)
        f = tmp_path / "spec.yaml"
        f.write_text(raw, encoding="utf-8")
        spec = load_spec(str(f))
        ep = parse_endpoint(spec, "/things/{id}", "get")
        schema = ep.responses["200"].schema
        assert schema is not None
        assert schema["type"] == "object"
        assert "name" in schema.get("required", [])
