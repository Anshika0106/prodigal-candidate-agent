"""HTTP client for the candidate REST API.

Design rules:
  * One attempt per call. No automatic retries - in particular a 409 is raised to the
    caller, who must re-fetch the record and re-decide (see services.apply_updates).
  * Errors never contain the API key, headers, request bodies or response bodies
    (bodies can echo personal data).
"""
import re
import uuid

import requests

from config import Config
from validators import READ_ONLY_FIELDS, WRITABLE_FIELDS

TIMEOUT_SECONDS = 15
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FIELD_NAME_RE = re.compile(r"^[a-z_]{1,64}$")


# --- Exceptions ------------------------------------------------------------------
class ApiError(Exception):
    """Base class. `status_code` is None for non-HTTP failures."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ApiConnectionError(ApiError):
    """Timeout or network failure. For a PATCH the outcome is UNKNOWN - re-fetch to check."""


class ApiAuthError(ApiError):
    """401 / 403: missing or invalid API key."""


class ApiNotFoundError(ApiError):
    """404."""


class ApiConflictError(ApiError):
    """409 (or 412): the record changed since it was read. Re-fetch and reconsider."""


class ApiValidationError(ApiError):
    """422: unknown / read-only field or empty body. `fields` lists offending field names."""

    def __init__(self, message: str, status_code: int | None = 422, fields: list[str] | None = None):
        super().__init__(message, status_code)
        self.fields = fields or []


class ApiRateLimitError(ApiError):
    """429 (limit is 120 requests/minute)."""

    def __init__(self, message: str, status_code: int | None = 429, retry_after: int | None = None):
        super().__init__(message, status_code)
        self.retry_after = retry_after


class ApiServerError(ApiError):
    """5xx. A PATCH may or may not have been applied."""


class ApiResponseError(ApiError):
    """2xx response that was not the expected JSON shape."""


class InvalidPatchError(ValueError):
    """Programming error: the PATCH request itself is not allowed (client-side check)."""


# --- Response parsing ---------------------------------------------------------------
def extract_record(body) -> dict:
    """The single-record response: an object that has an 'id' (tolerates one wrapper key)."""
    if isinstance(body, dict) and "id" in body:
        return body
    if isinstance(body, dict):
        inner = [v for v in body.values() if isinstance(v, dict) and "id" in v]
        if len(inner) == 1:
            return inner[0]
    raise ApiResponseError("Unexpected candidate record shape in response")


def _error_field_names(resp) -> list[str]:
    """Best-effort: names of rejected fields from a 422 body, never values."""
    try:
        detail = resp.json().get("detail")
        names = []
        if isinstance(detail, list):
            for item in detail:
                loc = item.get("loc") if isinstance(item, dict) else None
                if loc and isinstance(loc[-1], str) and _FIELD_NAME_RE.match(loc[-1]):
                    names.append(loc[-1])
        return sorted(set(names))
    except Exception:
        return []


def _raise_for_status(resp, method: str, path: str) -> None:
    code = resp.status_code
    if 200 <= code < 300:
        return
    where = f"{method} {path}"
    if code in (401, 403):
        raise ApiAuthError(f"{where}: HTTP {code} - check CANDIDATE_DB_API_KEY", code)
    if code == 404:
        raise ApiNotFoundError(f"{where}: HTTP 404 not found", code)
    if code in (409, 412):
        raise ApiConflictError(f"{where}: HTTP {code} - record changed since it was read", code)
    if code == 422:
        fields = _error_field_names(resp)
        extra = f" (fields: {', '.join(fields)})" if fields else ""
        raise ApiValidationError(f"{where}: HTTP 422 request rejected{extra}", code, fields)
    if code == 429:
        retry = resp.headers.get("Retry-After", "")
        raise ApiRateLimitError(
            f"{where}: HTTP 429 rate limited", code, int(retry) if retry.isdigit() else None
        )
    if 500 <= code < 600:
        raise ApiServerError(f"{where}: HTTP {code} server error", code)
    raise ApiError(f"{where}: HTTP {code} unexpected response", code)


# --- Client ----------------------------------------------------------------------------
class CandidateApiClient:
    def __init__(self, config: Config, session: requests.Session | None = None):
        self._base_url = config.api_base_url
        self._api_key = config.api_key
        self._session = session or requests.Session()

    def _request(self, method: str, path: str, *, auth: bool = True,
                 headers: dict | None = None, json_body: dict | None = None):
        all_headers = {"Accept": "application/json", **(headers or {})}
        if auth:
            all_headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = self._session.request(
                method, f"{self._base_url}{path}",
                headers=all_headers, json=json_body, timeout=TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            raise ApiConnectionError(f"{method} {path}: timed out after {TIMEOUT_SECONDS}s") from None
        except requests.RequestException as exc:
            raise ApiConnectionError(f"{method} {path}: request failed ({type(exc).__name__})") from None
        _raise_for_status(resp, method, path)
        try:
            return resp.json()
        except ValueError:
            raise ApiResponseError(f"{method} {path}: response was not valid JSON") from None

    @staticmethod
    def _check_id(candidate_id: str) -> str:
        if not isinstance(candidate_id, str) or not _ID_RE.match(candidate_id):
            raise ValueError("candidate_id must be a simple id such as 'c001'")
        return candidate_id

    def get_schema(self) -> dict:
        """GET /v1/schema (public, no Authorization header sent)."""
        body = self._request("GET", "/v1/schema", auth=False)
        if not isinstance(body, dict) or not isinstance(body.get("mandatory_fields"), list):
            raise ApiResponseError("GET /v1/schema: unexpected schema shape")
        return body

    def list_candidates(self) -> list[dict]:
        """GET /v1/candidates -> the list inside {"candidates": [...]}."""
        body = self._request("GET", "/v1/candidates")
        items = body.get("candidates") if isinstance(body, dict) else None
        if not isinstance(items, list) or not all(isinstance(i, dict) and "id" in i for i in items):
            raise ApiResponseError("GET /v1/candidates: expected {'candidates': [objects with 'id']}")
        return items

    def get_candidate(self, candidate_id: str) -> dict:
        """GET /v1/candidates/{id} -> the candidate record (includes 'version')."""
        self._check_id(candidate_id)
        record = extract_record(self._request("GET", f"/v1/candidates/{candidate_id}"))
        if record.get("id") != candidate_id:
            raise ApiResponseError("GET candidate: returned record id does not match request")
        return record

    def patch_candidate(self, candidate_id: str, fields: dict, version: int) -> dict:
        """PATCH /v1/candidates/{id} - exactly one attempt, optimistic-concurrency guarded.

        Sends If-Match: "<version>" and a fresh Idempotency-Key. Only explicitly writable
        fields are allowed; id/phone/version/updated_at raise InvalidPatchError.
        """
        self._check_id(candidate_id)
        if not isinstance(version, int) or isinstance(version, bool):
            raise InvalidPatchError("version must be the integer read from the record")
        if not fields:
            raise InvalidPatchError("no fields to update")
        bad = sorted(set(fields) - WRITABLE_FIELDS)
        if bad:
            kind = "read-only" if set(bad) & READ_ONLY_FIELDS else "unknown/non-writable"
            raise InvalidPatchError(f"{kind} field(s) in PATCH: {', '.join(bad)}")
        body = self._request(
            "PATCH", f"/v1/candidates/{candidate_id}",
            headers={"If-Match": f'"{version}"', "Idempotency-Key": str(uuid.uuid4())},
            json_body=dict(fields),
        )
        if not isinstance(body, dict):
            raise ApiResponseError("PATCH response was not a JSON object")
        return body
