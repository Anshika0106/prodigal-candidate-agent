import uuid

import pytest
import requests

from api_client import (
    ApiAuthError, ApiConflictError, ApiConnectionError, ApiError, ApiNotFoundError,
    ApiRateLimitError, ApiResponseError, ApiServerError, ApiValidationError,
    CandidateApiClient, InvalidPatchError,
)
from config import Config

KEY = "dummy-secret-key-xyz"


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, bad_json=False):
        self.status_code, self._body, self.headers, self._bad = status, body, headers or {}, bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Records every request; returns queued responses. Never touches the network."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(*responses):
    cfg = Config("p", "https://api.test", KEY, "llm", "m")
    session = FakeSession(*responses)
    return CandidateApiClient(cfg, session), session


# --- reads ---
def test_schema_needs_no_auth_header():
    client, s = make_client(FakeResponse(body={"mandatory_fields": ["email"]}))
    assert client.get_schema()["mandatory_fields"] == ["email"]
    call = s.calls[0]
    assert "Authorization" not in call["headers"]
    assert call["url"] == "https://api.test/v1/schema" and call["timeout"] == 15


def test_list_parses_candidates_envelope_and_uses_bearer():
    client, s = make_client(FakeResponse(body={"candidates": [{"id": "c001"}, {"id": "c002"}]}))
    assert [c["id"] for c in client.list_candidates()] == ["c001", "c002"]
    assert s.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert s.calls[0]["method"] == "GET" and s.calls[0]["timeout"] == 15


@pytest.mark.parametrize("body", [[{"id": "c1"}], {"items": []}, {"candidates": "x"}, {"candidates": [1]}])
def test_list_rejects_other_shapes(body):
    client, _ = make_client(FakeResponse(body=body))
    with pytest.raises(ApiResponseError):
        client.list_candidates()


def test_get_candidate_and_id_checks():
    client, s = make_client(FakeResponse(body={"id": "c001", "version": 3}))
    assert client.get_candidate("c001")["version"] == 3
    assert s.calls[0]["url"] == "https://api.test/v1/candidates/c001"
    with pytest.raises(ValueError):
        client.get_candidate("../v1/reset")
    client2, _ = make_client(FakeResponse(body={"id": "c999"}))
    with pytest.raises(ApiResponseError):
        client2.get_candidate("c001")


# --- PATCH headers and body ---
def test_patch_headers_and_body():
    client, s = make_client(FakeResponse(body={"id": "c001", "version": 2}))
    client.patch_candidate("c001", {"email": "a@b.co", "cgpa": 8.4}, 1)
    call = s.calls[0]
    assert call["method"] == "PATCH" and call["url"] == "https://api.test/v1/candidates/c001"
    assert call["headers"]["If-Match"] == '"1"'
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    uuid.UUID(call["headers"]["Idempotency-Key"])
    assert call["json"] == {"email": "a@b.co", "cgpa": 8.4}
    assert call["timeout"] == 15


def test_every_patch_gets_fresh_idempotency_key():
    client, s = make_client(FakeResponse(body={}), FakeResponse(body={}))
    client.patch_candidate("c001", {"notes": "a"}, 1)
    client.patch_candidate("c001", {"notes": "a"}, 1)
    keys = {c["headers"]["Idempotency-Key"] for c in s.calls}
    assert len(keys) == 2


@pytest.mark.parametrize("bad_field", ["id", "phone", "version", "updated_at", "shoe_size"])
def test_patch_refuses_read_only_or_unknown_fields(bad_field):
    client, s = make_client()
    with pytest.raises(InvalidPatchError):
        client.patch_candidate("c001", {"email": "a@b.co", bad_field: "x"}, 1)
    assert s.calls == []  # nothing was sent


def test_patch_refuses_empty_body_and_bad_version():
    client, s = make_client()
    with pytest.raises(InvalidPatchError):
        client.patch_candidate("c001", {}, 1)
    with pytest.raises(InvalidPatchError):
        client.patch_candidate("c001", {"notes": "x"}, "1")
    with pytest.raises(InvalidPatchError):
        client.patch_candidate("c001", {"notes": "x"}, True)
    assert s.calls == []


# --- error mapping ---
@pytest.mark.parametrize("status,exc", [
    (401, ApiAuthError), (403, ApiAuthError), (404, ApiNotFoundError), (409, ApiConflictError),
    (422, ApiValidationError), (429, ApiRateLimitError), (500, ApiServerError), (503, ApiServerError),
    (418, ApiError),
])
def test_status_mapping(status, exc):
    client, _ = make_client(FakeResponse(status=status, body={}))
    with pytest.raises(exc) as err:
        client.get_candidate("c001")
    assert type(err.value) is exc and err.value.status_code == status


def test_422_reports_field_names_not_values():
    body = {"detail": [{"loc": ["body", "phone"], "msg": "read-only", "input": "+919876543210"}]}
    client, _ = make_client(FakeResponse(status=422, body=body))
    with pytest.raises(ApiValidationError) as err:
        client.get_candidate("c001")
    assert err.value.fields == ["phone"] and "9876" not in str(err.value)


def test_429_retry_after():
    client, _ = make_client(FakeResponse(status=429, body={}, headers={"Retry-After": "7"}))
    with pytest.raises(ApiRateLimitError) as err:
        client.list_candidates()
    assert err.value.retry_after == 7


def test_timeout_and_connection_errors():
    client, _ = make_client(requests.Timeout("slow"), requests.ConnectionError("down"))
    with pytest.raises(ApiConnectionError) as err:
        client.get_candidate("c001")
    assert "timed out" in str(err.value)
    with pytest.raises(ApiConnectionError):
        client.get_candidate("c001")


def test_malformed_json():
    client, _ = make_client(FakeResponse(bad_json=True))
    with pytest.raises(ApiResponseError):
        client.get_candidate("c001")


def test_no_automatic_retry_on_409_or_5xx():
    client, s = make_client(FakeResponse(status=409, body={}))
    with pytest.raises(ApiConflictError):
        client.patch_candidate("c001", {"notes": "x"}, 1)
    assert len(s.calls) == 1
    client, s = make_client(FakeResponse(status=503, body={}))
    with pytest.raises(ApiServerError):
        client.patch_candidate("c001", {"notes": "x"}, 1)
    assert len(s.calls) == 1


def test_errors_and_repr_never_contain_key():
    client, _ = make_client(FakeResponse(status=401, body={}))
    with pytest.raises(ApiAuthError) as err:
        client.get_candidate("c001")
    assert KEY not in str(err.value) and KEY not in repr(client)
