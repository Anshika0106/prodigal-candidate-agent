from datetime import datetime, timedelta, timezone

import pytest

from api_client import ApiConflictError, ApiServerError
from services import (
    TooManyConflictsError, append_note, apply_updates, build_tracking_update,
    next_attempt_count, plan_updates,
)

NOW = datetime(2026, 11, 20, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def record(**over):
    rec = {
        "id": "c900", "version": 1, "full_name": "Test Person", "phone": "+919876543210",
        "email": "t@example.com", "college": "Test College", "degree": "B.Tech",
        "branch": "Computer Engineering", "graduation_year": 2027, "cgpa": None,
        "preferred_location": None, "available_from": None, "internship_duration_months": 3,
        "collection_status": "pending", "attempt_count": 2, "notes": None,
    }
    rec.update(over)
    return rec


class FakeClient:
    """Stands in for CandidateApiClient; scripted PATCH outcomes, no network."""

    def __init__(self, patch_outcomes, fetch_records=()):
        self.patch_outcomes, self.fetch_records = list(patch_outcomes), list(fetch_records)
        self.patches, self.fetches = [], 0

    def patch_candidate(self, cid, fields, version):
        self.patches.append((cid, dict(fields), version))
        outcome = self.patch_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get_candidate(self, cid):
        self.fetches += 1
        return self.fetch_records.pop(0)


# --- planning ---
def test_fills_empty_fields_and_normalises():
    plan = plan_updates(record(), {"cgpa": "8.4", "preferred_location": "Bangalore"})
    assert plan.to_send == {"cgpa": 8.4, "preferred_location": "Bengaluru"}


def test_valid_existing_value_not_overwritten_without_explicit_correction():
    plan = plan_updates(record(), {"email": "new@example.com"})
    assert plan.to_send == {} and "not explicitly corrected" in plan.skipped["email"]


def test_explicit_correction_overwrites_valid_value():
    plan = plan_updates(record(), {"email": "new@example.com"}, explicit_corrections={"email"})
    assert plan.to_send == {"email": "new@example.com"}


def test_invalid_existing_value_can_be_replaced():
    plan = plan_updates(record(graduation_year=2035, email="broken"), {"graduation_year": "2027", "email": "ok@x.com"})
    assert plan.to_send == {"graduation_year": 2027, "email": "ok@x.com"}


def test_unchanged_value_not_resent():
    plan = plan_updates(record(), {"degree": "b tech"}, explicit_corrections={"degree"})
    assert plan.to_send == {} and plan.skipped["degree"] == "unchanged"


def test_invalid_values_never_sent():
    plan = plan_updates(record(), {"cgpa": 85, "phone": "+911111111111", "id": "c1", "version": 9})
    assert plan.to_send == {}
    assert set(plan.skipped) == {"cgpa", "phone", "id", "version"}
    assert all(r.startswith("invalid") for r in plan.skipped.values())


# --- tracking helpers ---
def test_attempt_count_increments_from_stored_value():
    assert next_attempt_count(record(attempt_count=2)) == 3
    assert next_attempt_count(record(attempt_count=None)) == 1
    assert next_attempt_count(record(attempt_count=50)) == 50  # capped


def test_append_note_preserves_existing_and_respects_limit():
    assert append_note(None, "first") == "first"
    assert append_note("old", "new") == "old\nnew"
    out = append_note("o" * 3900, "n" * 500)
    assert len(out) <= 4000 and out.startswith("o" * 3900) and out.endswith("…")
    assert append_note("o" * 3990, "new note here that is long") == "o" * 3990  # no room: keep as is
    assert len(append_note(None, "x" * 5000)) == 4000


def test_tracking_update_only_includes_requested_changes():
    assert build_tracking_update(record()) == {}
    upd = build_tracking_update(record(notes="old"), status="in_progress", note="left: cgpa",
                                increment_attempt=True, contacted=True, now=NOW)
    assert upd == {
        "collection_status": "in_progress", "attempt_count": 3,
        "last_contacted_at": "2026-11-20T15:30:00+05:30", "notes": "old\nleft: cgpa",
    }


# --- apply_updates ---
def test_apply_sends_one_guarded_patch_with_version():
    client = FakeClient([{"id": "c900", "version": 2}])
    res = apply_updates(client, "c900", record(), {"cgpa": 8.4}, status="completed")
    assert client.patches == [("c900", {"cgpa": 8.4, "collection_status": "completed"}, 1)]
    assert res.sent == {"cgpa": 8.4, "collection_status": "completed"}
    assert res.record["version"] == 2 and client.fetches == 0


def test_apply_with_nothing_to_send_makes_no_call():
    client = FakeClient([])
    res = apply_updates(client, "c900", record(collection_status="completed"), {"cgpa": 99}, status="completed")
    assert client.patches == [] and res.sent == {}


def test_apply_refetches_when_patch_response_is_not_a_record():
    client = FakeClient([{}], fetch_records=[record(version=2)])
    res = apply_updates(client, "c900", record(), {"cgpa": 8.4})
    assert client.fetches == 1 and res.record["version"] == 2


def test_conflict_refetches_and_reevaluates():
    # Someone else already filled cgpa and bumped attempt_count to 5 -> we must not clobber them.
    fresh = record(version=2, cgpa=9.0, attempt_count=5)
    client = FakeClient([ApiConflictError("409", 409), {"id": "c900", "version": 3}], [fresh])
    res = apply_updates(client, "c900", record(), {"cgpa": 8.4, "email": None, "preferred_location": "Remote"},
                        increment_attempt=True)
    assert res.conflicts == 1 and client.fetches == 1
    first, second = client.patches
    assert first[1]["cgpa"] == 8.4 and first[1]["attempt_count"] == 3 and first[2] == 1
    assert "cgpa" not in second[1]                      # cgpa now valid in DB -> not overwritten
    assert second[1]["attempt_count"] == 6 and second[2] == 2   # incremented from FRESH stored value
    assert second[1]["preferred_location"] == "Remote"


def test_conflict_where_nothing_remains_makes_no_second_patch():
    fresh = record(version=2, cgpa=8.4)
    client = FakeClient([ApiConflictError("409", 409)], [fresh])
    res = apply_updates(client, "c900", record(), {"cgpa": 8.4})
    assert len(client.patches) == 1 and res.sent == {} and res.conflicts == 1


def test_too_many_conflicts_stops():
    client = FakeClient([ApiConflictError("409", 409)] * 3, [record(version=2), record(version=3)])
    with pytest.raises(TooManyConflictsError):
        apply_updates(client, "c900", record(), {"cgpa": 8.4}, max_conflicts=1)
    assert len(client.patches) == 2


def test_other_errors_propagate_without_retry():
    client = FakeClient([ApiServerError("boom", 500)])
    with pytest.raises(ApiServerError):
        apply_updates(client, "c900", record(), {"cgpa": 8.4})
    assert len(client.patches) == 1 and client.fetches == 0
