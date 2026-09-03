"""
tests/test_executor.py
───────────────────────
Unit tests for Module 3 (executor.py).

All HTTP calls are mocked using unittest.mock — no real network traffic.
Tests verify:
  - Mutation guard fires correctly
  - Path parameter substitution
  - Successful response handling (JSON and text bodies)
  - Timeout handling
  - Network error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from groundtruth.executor import execute, _substitute_path_params
from groundtruth.models import EndpointDefinition, ExecutionResult, TestCase


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_endpoint(method: str = "get", path: str = "/pet/{petId}") -> EndpointDefinition:
    return EndpointDefinition(
        path=path,
        method=method,
        parameters=[],
        request_body=None,
        responses={"200": MagicMock(description="OK", schema=None)},
        operation_id=None,
        summary=None,
    )


def _make_test_case(
    path_params: dict | None = None,
    query_params: dict | None = None,
    request_body=None,
) -> TestCase:
    return TestCase(
        id="tc_001",
        category="happy_path",
        description="test case",
        path_params=path_params or {"petId": 10},
        query_params=query_params or {},
        headers={},
        request_body=request_body,
        expected_status=200,
    )


# ── Path param substitution ───────────────────────────────────────────────────


class TestSubstitutePathParams:
    def test_replaces_single_param(self):
        result = _substitute_path_params("/pet/{petId}", {"petId": 42})
        assert result == "/pet/42"

    def test_replaces_multiple_params(self):
        result = _substitute_path_params("/a/{x}/b/{y}", {"x": "foo", "y": "bar"})
        assert result == "/a/foo/b/bar"

    def test_coerces_int_to_str(self):
        result = _substitute_path_params("/items/{id}", {"id": 99})
        assert result == "/items/99"

    def test_leaves_unfilled_placeholder(self):
        result = _substitute_path_params("/pet/{petId}", {})
        assert result == "/pet/{petId}"

    def test_ignores_extra_params(self):
        result = _substitute_path_params("/pet/{petId}", {"petId": 1, "extra": "ignored"})
        assert result == "/pet/1"


# ── Mutation guard ────────────────────────────────────────────────────────────


class TestMutationGuard:
    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    def test_skips_mutation_methods_by_default(self, method):
        ep = _make_endpoint(method=method, path="/pet")
        tc = _make_test_case(path_params={})
        result = execute(tc, ep, "http://localhost", allow_mutations=False)
        assert result.skipped is True
        assert result.status_code is None
        assert "allow-mutations" in result.skip_reason.lower() or "mutation" in result.skip_reason.lower()

    def test_get_not_skipped_by_default(self):
        ep = _make_endpoint(method="get")
        tc = _make_test_case()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"id": 10}

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            result = execute(tc, ep, "http://localhost", allow_mutations=False)

        assert result.skipped is False

    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    def test_mutation_allowed_when_flag_set(self, method):
        ep = _make_endpoint(method=method, path="/pet")
        tc = _make_test_case(path_params={})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            getattr(mock_client, method).return_value = mock_response

            result = execute(tc, ep, "http://localhost", allow_mutations=True)

        assert result.skipped is False


# ── Successful execution ──────────────────────────────────────────────────────


class TestSuccessfulExecution:
    def _run(self, json_body=None, text_body=None, status=200):
        ep = _make_endpoint(method="get")
        tc = _make_test_case()

        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.headers = {"content-type": "application/json"}

        if json_body is not None:
            mock_response.json.return_value = json_body
        else:
            mock_response.json.side_effect = ValueError("not json")
            mock_response.text = text_body

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            result = execute(tc, ep, "http://api.example.com")

        return result

    def test_returns_correct_status_code(self):
        result = self._run(json_body={"id": 10}, status=200)
        assert result.status_code == 200

    def test_parses_json_body(self):
        result = self._run(json_body={"id": 10, "name": "doggie"})
        assert result.response_body == {"id": 10, "name": "doggie"}

    def test_falls_back_to_text_body(self):
        result = self._run(text_body="plain text response")
        assert result.response_body == "plain text response"

    def test_not_skipped(self):
        result = self._run(json_body={})
        assert result.skipped is False
        assert result.error is None

    def test_response_time_is_positive(self):
        result = self._run(json_body={})
        assert result.response_time_ms >= 0

    def test_test_case_id_preserved(self):
        result = self._run(json_body={})
        assert result.test_case_id == "tc_001"

    def test_url_construction_with_base_url_trailing_slash(self):
        """Trailing slash on base_url should not produce double slashes."""
        ep = _make_endpoint(method="get", path="/pet/10")
        tc = _make_test_case(path_params={})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            execute(tc, ep, "http://api.example.com/")

        call_args = mock_client.get.call_args
        url_called = call_args[0][0]
        assert "//" not in url_called.replace("http://", "").replace("https://", "")


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_timeout_returns_error_result(self):
        import httpx
        ep = _make_endpoint()
        tc = _make_test_case()

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("timed out")

            result = execute(tc, ep, "http://api.example.com")

        assert result.status_code is None
        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()
        assert result.skipped is False

    def test_network_error_returns_error_result(self):
        import httpx
        ep = _make_endpoint()
        tc = _make_test_case()

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.ConnectError("connection refused")

            result = execute(tc, ep, "http://localhost:9999")

        assert result.status_code is None
        assert result.error is not None
        assert result.skipped is False

    def test_4xx_response_is_not_an_error(self):
        """HTTP 404 is a valid response — it should NOT be recorded as error=..."""
        ep = _make_endpoint()
        tc = _make_test_case(path_params={"petId": 999999})

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.json.side_effect = ValueError
        mock_response.text = "not found"

        with patch("groundtruth.executor.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            result = execute(tc, ep, "http://api.example.com")

        assert result.status_code == 404
        assert result.error is None
        assert result.skipped is False
