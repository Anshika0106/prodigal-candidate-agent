"""Test doubles: an in-memory candidate API and a scripted LLM. No network anywhere."""
import copy

from api_client import ApiConflictError, ApiServerError
from llm import LLMTurn, ToolCall
from validators import READ_ONLY_FIELDS


def make_record(**over) -> dict:
    """Test candidate c001-like: only email and cgpa are missing."""
    rec = {
        "id": "c001", "version": 1, "updated_at": "2026-11-01T10:00:00+00:00",
        "full_name": "Aarav Mehta", "phone": "+919876543210", "email": None,
        "college": "Test Institute", "degree": "B.Tech", "branch": "Computer Engineering",
        "graduation_year": 2027, "cgpa": None, "preferred_location": "Remote",
        "available_from": "2026-06-01", "internship_duration_months": 3,
        "linkedin_url": None, "github_url": None, "skills": [],
        "collection_status": "pending", "attempt_count": 2, "last_contacted_at": None,
        "notes": None,
    }
    rec.update(over)
    return rec


class FakeApi:
    """Mimics GET/PATCH with optimistic concurrency. `script` entries are consumed per PATCH:
    'conflict' (another writer changed the record first), an Exception to raise, or None."""

    def __init__(self, record=None, script=(), concurrent_changes=None):
        self.record = copy.deepcopy(record if record is not None else make_record())
        self.script = list(script)
        self.concurrent_changes = concurrent_changes or {}
        self.patches: list[dict] = []     # bodies of successful PATCHes
        self.attempts: list[dict] = []    # bodies of every PATCH attempt
        self.versions: list[int] = []     # If-Match versions sent
        self.gets = 0

    def get_candidate(self, candidate_id):
        self.gets += 1
        return copy.deepcopy(self.record)

    def patch_candidate(self, candidate_id, fields, version):
        assert not set(fields) & READ_ONLY_FIELDS, "tried to write a read-only field"
        self.attempts.append(copy.deepcopy(fields))
        self.versions.append(version)
        step = self.script.pop(0) if self.script else None
        if step == "conflict":
            self.record.update(self.concurrent_changes)
            self.record["version"] += 1
            raise ApiConflictError("409", 409)
        if isinstance(step, Exception):
            raise step
        if version != self.record["version"]:
            raise ApiConflictError("409", 409)
        self.record.update(fields)
        self.record["version"] += 1
        self.patches.append(copy.deepcopy(fields))
        return copy.deepcopy(self.record)


def server_error():
    return ApiServerError("HTTP 503 server error", 503)


def call(name, **args) -> ToolCall:
    return ToolCall(name, args, id=None)


def turn(text="", *calls) -> LLMTurn:
    return LLMTurn(text, list(calls))


class ScriptedLLM:
    """Returns pre-written turns in order and records everything it was sent."""

    def __init__(self, *turns):
        self.turns, self.received = list(turns), []

    def _next(self):
        return self.turns.pop(0)

    def send_text(self, text):
        self.received.append(("text", text))
        return self._next()

    def send_tool_results(self, results):
        self.received.append(("results", [(c.name, r) for c, r in results]))
        return self._next()


class Inputs:
    """Candidate inputs for the agent's `read`. Exceptions in the list are raised; None = EOF."""

    def __init__(self, *items):
        self.items = list(items)

    def __call__(self):
        item = self.items.pop(0)
        if isinstance(item, BaseException) or (isinstance(item, type) and issubclass(item, BaseException)):
            raise item
        return item
