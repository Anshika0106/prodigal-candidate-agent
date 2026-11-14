"""Conversation transcripts (chat + tool calls + API calls) with personal data redacted."""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from api_client import ApiError

SENSITIVE_KEYS = {"full_name", "phone", "email", "linkedin_url", "github_url"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_URL_RE = re.compile(r"(?:https?://)?(?:[\w-]+\.)*(?:linkedin|github)\.com\S*", re.I)


class Redactor:
    """Masks the candidate's name, phone, emails and profile URLs in any JSON-like data."""

    def __init__(self, record: dict | None = None):
        record = record or {}
        name = record.get("full_name") if isinstance(record.get("full_name"), str) else ""
        phone = record.get("phone") if isinstance(record.get("phone"), str) else ""
        self._exact = [s for s in (name.strip(), phone.strip()) if s]
        self._tokens = [t for t in re.split(r"\s+", name.strip()) if len(t) >= 3]

    def text(self, s: str) -> str:
        for value in sorted(self._exact, key=len, reverse=True):
            s = re.sub(re.escape(value), "[REDACTED]", s, flags=re.I)
        for token in self._tokens:
            s = re.sub(rf"\b{re.escape(token)}\b", "[NAME]", s, flags=re.I)
        s = _EMAIL_RE.sub("[EMAIL]", s)
        s = _URL_RE.sub("[URL]", s)
        return _PHONE_RE.sub("[PHONE]", s)

    def obj(self, value: Any, key: str | None = None) -> Any:
        if key in SENSITIVE_KEYS and isinstance(value, (str, list)):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {k: self.obj(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.obj(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


class Transcript:
    def __init__(self, now_fn: Callable[[], datetime] | None = None):
        self._now = now_fn or (lambda: datetime.now().astimezone())
        self.started_at = self._now().isoformat()
        self.events: list[dict] = []

    def add(self, kind: str, data: dict) -> None:
        self.events.append({"t": self._now().isoformat(timespec="seconds"), "type": kind, **data})

    def save(self, directory: Path, candidate_id: str, meta: dict, record: dict | None) -> Path:
        redactor = Redactor(record)
        document = redactor.obj({
            "candidate_id": candidate_id,
            "redacted": True,
            "redaction_note": "Name, phone, email and profile URLs are masked. Candidate replies in "
                              "sample runs were typed by the developer (test data), not a real person.",
            "started_at": self.started_at,
            "ended_at": self._now().isoformat(),
            **meta,
            "events": self.events,
        })
        document["candidate_id"] = candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"transcript_{candidate_id}_{self._now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class RecordingClient:
    """Wraps a CandidateApiClient and logs each call and response (redacted when saved)."""

    def __init__(self, inner, log: Callable[[str, dict], None]):
        self._inner, self._log = inner, log

    def get_candidate(self, candidate_id: str) -> dict:
        entry = {"method": "GET", "path": f"/v1/candidates/{candidate_id}"}
        try:
            record = self._inner.get_candidate(candidate_id)
        except ApiError as exc:
            self._log("api_call", {**entry, "error": f"{type(exc).__name__}: {exc}"})
            raise
        self._log("api_call", {**entry, "status": "ok", "response": record})
        return record

    def patch_candidate(self, candidate_id: str, fields: dict, version: int) -> dict:
        entry = {"method": "PATCH", "path": f"/v1/candidates/{candidate_id}",
                 "if_match": f'"{version}"', "body": dict(fields)}
        try:
            record = self._inner.patch_candidate(candidate_id, fields, version)
        except ApiError as exc:
            self._log("api_call", {**entry, "error": f"{type(exc).__name__}: {exc}"})
            raise
        self._log("api_call", {**entry, "status": "ok", "response": record})
        return record
