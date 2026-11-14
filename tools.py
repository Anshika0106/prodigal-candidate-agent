"""The tools the LLM can call - the ONLY way it can read or change candidate data.

Every rule is enforced here in Python, not in the prompt:
  * nothing is collected before identity is confirmed;
  * every value goes through validators.validate_field before it is staged;
  * writes go through services.apply_updates (If-Match, idempotency, 409 re-evaluation,
    no overwrite of valid data without an explicit correction);
  * `completed` is refused unless every mandatory field would be valid in the database;
  * phone and other read-only fields can never be written.
The LLM sees the candidate's name and which fields are needed - not stored values.
"""
from datetime import datetime
from typing import Any, Callable

from api_client import ApiConflictError, ApiConnectionError, ApiError, ApiServerError
from services import IncompleteRecordError, TooManyConflictsError, apply_updates
from validators import (
    DEGREES, DURATIONS, LOCATIONS, MANDATORY_FIELDS, OPTIONAL_FIELDS, READ_ONLY_FIELDS,
    TRACKING_FIELDS, analyse_record, check_stored, is_empty, validate_field,
)

# Fields the candidate can supply (full_name is for identity only; phone is read-only).
COLLECTABLE_FIELDS = [f for f in MANDATORY_FIELDS if f not in READ_ONLY_FIELDS and f != "full_name"] + list(
    OPTIONAL_FIELDS
)

FIELD_GUIDE = {
    "email": "a valid email address",
    "college": "name of their college/university (2-150 chars)",
    "degree": f"one of: {', '.join(DEGREES)}",
    "branch": "specialisation/stream, e.g. Computer Engineering",
    "graduation_year": "4-digit graduation year, 2026-2028",
    "cgpa": "CGPA on a 10-point scale (0-10)",
    "preferred_location": f"one of: {', '.join(LOCATIONS)}",
    "available_from": "exact date they can start (day, month and year), between 2026-01-01 and 2027-12-31",
    "internship_duration_months": f"internship length in months: {', '.join(map(str, DURATIONS))}",
}

LLM_OUTCOMES = ("completed", "in_progress", "refused", "no_answer")
ESCALATION_CATEGORIES = ("wrong_person", "contradiction", "upset", "other")
MAX_FREE_TEXT = 500
MAX_PREVIOUS_NOTES = 1000


def _clip(text: Any, limit: int = MAX_FREE_TEXT) -> str:
    return " ".join(str(text or "").split())[:limit]


class CollectionSession:
    """State for one candidate conversation plus the tool implementations."""

    def __init__(self, client, record: dict, now_fn: Callable[[], datetime] | None = None):
        self.client = client
        self.record = record
        self.candidate_id = record["id"]
        self._now = now_fn or (lambda: datetime.now().astimezone())
        self.identity_confirmed = False
        self.contacted = False                # set by the agent once a message reached the candidate
        self.staged: dict[str, Any] = {}      # validated, normalised, not yet saved
        self.corrections: set[str] = set()    # fields the candidate explicitly corrected
        self.saved: list[str] = []            # data fields written during this conversation
        self.ended = False
        self.outcome: str | None = None
        self._tools = {
            "get_candidate_status": self.get_candidate_status,
            "confirm_identity": self.confirm_identity,
            "record_answers": self.record_answers,
            "save_progress": self.save_progress,
            "finish_conversation": self.finish_conversation,
            "escalate_to_human": self.escalate_to_human,
        }

    # ---- helpers ---------------------------------------------------------------
    @property
    def analysis(self):
        return analyse_record(self.record)

    def remaining_fields(self) -> list[str]:
        return [f for f in self.analysis.fields_to_collect if f not in self.staged]

    def _guard(self, need_identity: bool = True) -> dict | None:
        if self.ended:
            return {"error": "The conversation has already ended. Send a short goodbye only."}
        if need_identity and not self.identity_confirmed:
            return {"error": "Identity not confirmed yet. Confirm you are talking to the right "
                             "person, then call confirm_identity."}
        return None

    def _note(self, status: str, text: str) -> str:
        return f"[{self._now():%Y-%m-%d %H:%M}] {status}: {text}"

    def _progress_text(self) -> str:
        collected = sorted({*self.saved, *self.staged})
        left = self.remaining_fields()
        return (f"Collected this chat: {', '.join(collected) or 'nothing'}. "
                f"Still needed: {', '.join(left) or 'nothing'}.")

    def _commit(self, status: str, text: str, count_attempt: bool = True) -> dict:
        """Single guarded write: staged answers + final collection tracking.

        count_attempt=False skips attempt_count/last_contacted_at (nobody was contacted)."""
        result = apply_updates(
            self.client, self.candidate_id, self.record, self.staged, self.corrections,
            status=status, note=self._note(status, text), increment_attempt=count_attempt,
            contacted=count_attempt, now=self._now(), require_complete=(status == "completed"),
        )
        self.record = result.record
        self.saved += [f for f in result.sent if f not in TRACKING_FIELDS and f not in self.saved]
        self.staged.clear()
        self.corrections.clear()
        self.ended, self.outcome = True, status
        return {
            "recorded": True,
            "collection_status": self.record.get("collection_status"),
            "message": "Outcome recorded. Send ONE short closing message to the candidate. "
                       "Make no more tool calls.",
        }

    # ---- tools ---------------------------------------------------------------------
    def get_candidate_status(self) -> dict:
        a = self.analysis
        remaining = self.remaining_fields()
        status = {
            "candidate_name": self.record.get("full_name"),
            "identity_confirmed": self.identity_confirmed,
            "fields_needed": [
                {
                    "field": f,
                    "problem": "missing" if f in a.missing else "on file but invalid - collect again",
                    "expected": FIELD_GUIDE.get(f, ""),
                }
                for f in remaining
            ],
            "answers_pending_save": sorted(self.staged),
            "saved_this_conversation": list(self.saved),
            "all_mandatory_fields_ready": not remaining,
            "optional_fields_on_file": a.optional_populated,
            "previous_attempts": a.attempt_count if isinstance(a.attempt_count, int) else 0,
        }
        if a.blocked_read_only:
            status["cannot_fix_read_only_fields"] = a.blocked_read_only
        notes = self.record.get("notes")
        if isinstance(notes, str) and notes.strip():
            status["previous_notes"] = notes.strip()[-MAX_PREVIOUS_NOTES:]
        return status

    def confirm_identity(self, is_correct_person: bool, details: str = "") -> dict:
        if (guard := self._guard(need_identity=False)):
            return guard
        if is_correct_person is True:
            self.identity_confirmed = True
            return {"confirmed": True, "next": "Start collecting the needed fields."}
        return self.escalate_to_human(
            reason=_clip(details) or "Person on chat is not the named candidate", category="wrong_person"
        )

    def record_answers(self, answers: dict, confirmed_corrections: list | None = None) -> dict:
        if (guard := self._guard()):
            return guard
        if not isinstance(answers, dict) or not answers:
            return {"error": "answers must be an object of field -> value"}
        confirmed = {c for c in (confirmed_corrections or []) if isinstance(c, str)}
        results: dict[str, dict] = {}
        for name, raw in answers.items():
            if name in READ_ONLY_FIELDS or name == "full_name" or name not in COLLECTABLE_FIELDS:
                results[name] = {"status": "rejected", "reason": "not a field you can collect or change"}
                continue
            if is_empty(raw):
                results[name] = {"status": "invalid", "reason": "no value given"}
                continue
            check = validate_field(name, raw)
            if not check.valid:
                results[name] = {"status": "invalid", "reason": check.reason}
                continue
            value, existing = check.normalized_value, self.record.get(name)
            on_file_valid = not is_empty(existing) and check_stored(name, existing).valid
            correcting = name in confirmed
            if name in self.staged and self.staged[name] != value and not correcting:
                results[name] = {"status": "conflicts_with_earlier_answer",
                                 "reason": "The candidate gave a different value earlier in this chat. "
                                           "Ask which is correct; once clear, call again with this "
                                           "field in confirmed_corrections."}
            elif on_file_valid and value == existing:
                results[name] = {"status": "already_on_file", "reason": "same as the stored value"}
            elif on_file_valid and not correcting:
                results[name] = {"status": "needs_confirmation",
                                 "reason": "A valid value is already on file. Ask the candidate to "
                                           "confirm they want to change it, then call again with this "
                                           "field in confirmed_corrections."}
            else:
                self.staged[name] = value
                if correcting:
                    self.corrections.add(name)
                results[name] = {"status": "accepted", "value": value}
        remaining = self.remaining_fields()
        return {
            "results": results,
            "still_needed": remaining,
            "all_mandatory_fields_ready": not remaining,
            "hint": "Everything needed is collected - call save_progress or finish_conversation."
                    if not remaining else "Ask for the remaining fields.",
        }

    def save_progress(self) -> dict:
        if (guard := self._guard()):
            return guard
        if not self.staged:
            return {"saved_fields": [], "message": "Nothing pending to save.",
                    "still_needed": self.remaining_fields()}
        result = apply_updates(self.client, self.candidate_id, self.record, self.staged, self.corrections)
        self.record = result.record
        self.saved += [f for f in result.sent if f not in self.saved]
        not_saved = {k: v for k, v in result.skipped.items() if v != "unchanged"}
        self.staged.clear()
        self.corrections.clear()
        return {"saved_fields": sorted(result.sent), "not_saved": not_saved,
                "still_needed": self.remaining_fields()}

    def finish_conversation(self, outcome: str, note: str = "", follow_up_time: str = "",
                            refused_fields: list | None = None) -> dict:
        if (guard := self._guard()):
            return guard
        if outcome not in LLM_OUTCOMES:
            return {"error": f"outcome must be one of {', '.join(LLM_OUTCOMES)} "
                             "(use escalate_to_human for needs_human)"}
        extra = _clip(note)
        if outcome == "completed":
            outstanding = analyse_record({**self.record, **self._planned()}).fields_to_collect
            if outstanding:
                return {"error": "Cannot mark completed: mandatory fields still missing or invalid.",
                        "still_needed": outstanding}
            text = f"All mandatory fields collected. {extra}".strip()
        elif outcome == "refused":
            declined = [f for f in (refused_fields or []) if isinstance(f, str)]
            valid = [f for f in declined if f in self.remaining_fields()]
            if not valid:
                return {"error": "refused_fields must name outstanding fields the candidate declined.",
                        "still_needed": self.remaining_fields()}
            text = (f"Candidate declined to provide: {', '.join(valid)}. {self._progress_text()} "
                    f"{extra}").strip()
        elif outcome == "no_answer":
            if self.saved or self.staged:
                return {"error": "Information was collected; use in_progress instead of no_answer."}
            text = f"No information received. {extra}".strip()
        else:  # in_progress
            when = f" Candidate suggested: {_clip(follow_up_time, 120)}." if follow_up_time else ""
            text = f"{self._progress_text()}{when} {extra}".strip()
        return self._commit(outcome, text)

    def escalate_to_human(self, reason: str, category: str = "other") -> dict:
        if (guard := self._guard(need_identity=False)):
            return guard
        if category not in ESCALATION_CATEGORIES:
            category = "other"
        if category == "wrong_person":
            self.staged.clear()          # never save anything said by the wrong person
            self.corrections.clear()
        text = f"{category}: {_clip(reason) or 'needs human review'}. {self._progress_text()}"
        return self._commit("needs_human", text)

    def _planned(self) -> dict:
        """What the staged answers would write (used to pre-check completion)."""
        from services import plan_updates
        return plan_updates(self.record, self.staged, self.corrections).to_send

    # ---- exits triggered by the application, not the LLM ----------------------------
    def finish_from_exit(self, reason: str, kind: str = "left") -> dict | None:
        """Record an outcome when the chat ends without the LLM concluding it.

        kind: 'left' (quit/EOF/Ctrl+C/idle), 'later' (/later), 'technical' (our failure).
        """
        if self.ended:
            return None
        collected = bool(self.saved or self.staged)
        if kind == "later" or kind == "technical" or collected:
            status = "in_progress"
            text = f"Chat ended early ({_clip(reason, 120)}). {self._progress_text()}"
        else:
            status = "no_answer"
            text = f"Candidate left before providing any information ({_clip(reason, 120)})."
        # A failure of ours before the candidate heard anything is not a contact attempt.
        count = not (kind == "technical" and not self.contacted)
        try:
            return self._commit(status, text, count_attempt=count)
        except Exception as exc:   # report, never crash during shutdown
            return {"error": f"could not record outcome: {type(exc).__name__}"}

    # ---- dispatch ----------------------------------------------------------------------
    def dispatch(self, name: str, args: dict | None) -> dict:
        """Run a tool requested by the LLM. Always returns a JSON-serialisable dict."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return tool(**(args or {}))
        except TypeError:
            return {"error": "invalid arguments for this tool"}
        except IncompleteRecordError as exc:
            return {"error": "Cannot mark completed: mandatory fields still missing or invalid.",
                    "still_needed": exc.fields}
        except (TooManyConflictsError, ApiConflictError):
            return {"error": "The record kept changing; nothing was lost. Try again once, "
                             "otherwise escalate_to_human."}
        except (ApiConnectionError, ApiServerError) as exc:
            self._refresh_quietly()
            return {"error": f"Database unavailable ({type(exc).__name__}). Do not tell the "
                             "candidate their answers are saved. You may retry once."}
        except ApiError as exc:
            return {"error": f"Database rejected the request: {exc}"}
        except Exception:
            return {"error": "internal error"}

    def _refresh_quietly(self) -> None:
        """After an unknown-outcome write, re-read so the next attempt starts from the truth."""
        try:
            self.record = self.client.get_candidate(self.candidate_id)
        except Exception:
            pass


# ---- Tool declarations (provider-neutral JSON schema; llm.py converts them) -----------
_ANSWER_PROPS = {f: {"type": "string", "description": FIELD_GUIDE.get(f, "optional, as the candidate said it")}
                 for f in COLLECTABLE_FIELDS if f != "skills"}
_ANSWER_PROPS["skills"] = {"type": "array", "items": {"type": "string"},
                           "description": "optional list of skills"}

TOOL_SPECS = [
    {
        "name": "get_candidate_status",
        "description": "Get the candidate's name, which mandatory fields are still needed (and why), "
                       "what is pending save or saved, and notes from earlier attempts. Stored values "
                       "are never returned.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "confirm_identity",
        "description": "Call after the candidate answers your identity question. is_correct_person=true "
                       "unlocks data collection; false (wrong person / wrong number) ends the chat and "
                       "escalates to a human.",
        "parameters": {"type": "object", "properties": {
            "is_correct_person": {"type": "boolean"},
            "details": {"type": "string", "description": "what the person said, if not the candidate"},
        }, "required": ["is_correct_person"]},
    },
    {
        "name": "record_answers",
        "description": "Submit answers the candidate gave. Pass each value exactly as the candidate said "
                       "it; the system validates and normalises it and tells you which were accepted, "
                       "invalid or ambiguous. Never guess or fill in values yourself. Accepted answers "
                       "are held until save_progress or finish_conversation.",
        "parameters": {"type": "object", "properties": {
            "answers": {"type": "object", "properties": _ANSWER_PROPS},
            "confirmed_corrections": {
                "type": "array", "items": {"type": "string"},
                "description": "field names the candidate explicitly confirmed they want CHANGED "
                               "(replacing a valid stored value, or an earlier answer this chat)"},
        }, "required": ["answers"]},
    },
    {
        "name": "save_progress",
        "description": "Write all accepted answers to the database now. Call this regularly during the chat.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "finish_conversation",
        "description": "Record the outcome and end the chat (saves any pending answers). "
                       "completed: every mandatory field collected (refused by the system otherwise). "
                       "in_progress: candidate wants to continue later or stopped after giving some info. "
                       "refused: candidate declined one or more fields (list them). "
                       "no_answer: nothing was collected.",
        "parameters": {"type": "object", "properties": {
            "outcome": {"type": "string", "enum": list(LLM_OUTCOMES)},
            "note": {"type": "string", "description": "short context for a human; no personal data values"},
            "follow_up_time": {"type": "string", "description": "when the candidate asked to continue, if said"},
            "refused_fields": {"type": "array", "items": {"type": "string"}},
        }, "required": ["outcome"]},
    },
    {
        "name": "escalate_to_human",
        "description": "End the chat and flag a human: wrong person, contradictory or disputed record, "
                       "upset or abusive candidate, or anything you cannot resolve.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
            "category": {"type": "string", "enum": list(ESCALATION_CATEGORIES)},
        }, "required": ["reason", "category"]},
    },
]
