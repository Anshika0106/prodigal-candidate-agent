"""Safe, minimal database updates for one candidate.

Rules enforced here (the API itself validates nothing):
  * only validated values are sent;
  * a valid existing value is never overwritten unless the candidate explicitly
    corrected that field; an invalid or empty existing value may be replaced;
  * values identical to what is stored are not re-sent;
  * on 409 the record is re-fetched and the plan is rebuilt from the fresh data;
  * attempt_count is incremented from the stored value; notes are appended;
  * a PATCH that fails for any other reason is NOT retried (the exception propagates).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from api_client import ApiConflictError, ApiResponseError, CandidateApiClient
from validators import (
    MAX_NOTES_LEN, TRACKING_FIELDS, analyse_record, check_stored, is_empty, validate_field,
)

MAX_ATTEMPTS = 50
MIN_NOTE_SPACE = 20  # don't append a note fragment shorter than this


@dataclass
class UpdatePlan:
    to_send: dict[str, Any] = field(default_factory=dict)   # validated, normalised
    skipped: dict[str, str] = field(default_factory=dict)   # field -> reason (no values)


@dataclass
class UpdateResult:
    record: dict                                  # latest known record
    sent: dict[str, Any]                          # what was written ({} if nothing)
    skipped: dict[str, str]
    conflicts: int = 0                            # how many 409s were re-evaluated


class IncompleteRecordError(Exception):
    """Refused to write collection_status='completed': mandatory fields are not all valid."""

    def __init__(self, fields: list[str]):
        super().__init__(f"record is not complete; outstanding: {', '.join(fields)}")
        self.fields = fields


class TooManyConflictsError(Exception):
    """The record kept changing underneath us; stop and let a human/the agent decide."""


# --- Tracking helpers ---------------------------------------------------------------
def next_attempt_count(record: dict) -> int:
    """Stored attempt_count + 1, capped at the schema maximum."""
    current = record.get("attempt_count")
    if not isinstance(current, int) or isinstance(current, bool) or current < 0:
        current = 0
    return min(current + 1, MAX_ATTEMPTS)


def append_note(existing: Any, new_note: str, limit: int = MAX_NOTES_LEN) -> str:
    """Append to existing notes without exceeding `limit`. Existing text is never cut;
    the new text is truncated (with an ellipsis) if needed, or dropped if no room."""
    old = existing.strip() if isinstance(existing, str) else ""
    new = new_note.strip()
    if not new:
        return old
    if not old:
        return new if len(new) <= limit else new[: limit - 1] + "…"
    room = limit - len(old) - 1  # 1 for the newline separator
    if room < MIN_NOTE_SPACE:
        return old
    if len(new) > room:
        new = new[: room - 1] + "…"
    return f"{old}\n{new}"


def build_tracking_update(record: dict, *, status: str | None = None, note: str | None = None,
                          increment_attempt: bool = False, contacted: bool = False,
                          now: datetime | None = None) -> dict:
    """Only the tracking fields we have a reason to change; everything else is preserved."""
    update: dict[str, Any] = {}
    if status is not None:
        update["collection_status"] = status
    if increment_attempt:
        update["attempt_count"] = next_attempt_count(record)
    if contacted:
        update["last_contacted_at"] = (now or datetime.now().astimezone()).isoformat()
    if note:
        update["notes"] = append_note(record.get("notes"), note)
    return update


# --- Planning -------------------------------------------------------------------------
def plan_updates(record: dict, proposed: dict[str, Any],
                 explicit_corrections: Iterable[str] = ()) -> UpdatePlan:
    """Decide which proposed values may be written, given the current record."""
    corrections = set(explicit_corrections)
    plan = UpdatePlan()
    for name, value in proposed.items():
        result = validate_field(name, value)
        if not result.valid:
            plan.skipped[name] = f"invalid: {result.reason}"
            continue
        new, existing = result.normalized_value, record.get(name)
        if name in TRACKING_FIELDS:
            # Tracking fields are ours to maintain; only skip a no-op write.
            if new == existing:
                plan.skipped[name] = "unchanged"
            else:
                plan.to_send[name] = new
        elif is_empty(existing) or not check_stored(name, existing).valid:
            plan.to_send[name] = new                       # fill a gap / fix bad data
        elif new == existing:
            plan.skipped[name] = "unchanged"
        elif name in corrections:
            plan.to_send[name] = new                       # candidate explicitly corrected it
        else:
            plan.skipped[name] = "existing value is valid and was not explicitly corrected"
    return plan


def apply_updates(client: CandidateApiClient, candidate_id: str, record: dict,
                  proposed: dict[str, Any] | None = None,
                  explicit_corrections: Iterable[str] = (), *,
                  status: str | None = None, note: str | None = None,
                  increment_attempt: bool = False, contacted: bool = False,
                  max_conflicts: int = 3, now: datetime | None = None,
                  require_complete: bool = False) -> UpdateResult:
    """Write validated changes (plus optional tracking changes) in one guarded PATCH.

    `record` is the record as last read. On a 409 the record is re-fetched and
    everything (including attempt_count and notes) is recomputed from the fresh data.
    Any other API error propagates unchanged and is never retried here.
    With require_complete=True nothing is written unless the record, as it would look
    after the update, has every mandatory field valid (re-checked after each 409).
    """
    conflicts = 0
    while True:
        tracking = build_tracking_update(
            record, status=status, note=note, increment_attempt=increment_attempt,
            contacted=contacted, now=now,
        )
        plan = plan_updates(record, {**(proposed or {}), **tracking}, explicit_corrections)
        if require_complete:
            outstanding = analyse_record({**record, **plan.to_send}).fields_to_collect
            if outstanding:
                raise IncompleteRecordError(outstanding)
        if not plan.to_send:
            return UpdateResult(record, {}, plan.skipped, conflicts)
        try:
            response = client.patch_candidate(candidate_id, plan.to_send, record.get("version"))
        except ApiConflictError:
            conflicts += 1
            if conflicts > max_conflicts:
                raise TooManyConflictsError(
                    f"{candidate_id}: record changed {conflicts} times while updating"
                ) from None
            record = client.get_candidate(candidate_id)   # re-read, then re-evaluate
            continue
        if isinstance(response, dict) and response.get("id") == candidate_id and "version" in response:
            latest = response
        else:
            latest = client.get_candidate(candidate_id)
        if not isinstance(latest, dict):
            raise ApiResponseError("could not determine the updated record")
        return UpdateResult(latest, plan.to_send, plan.skipped, conflicts)
