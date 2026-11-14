import json
from datetime import datetime, timedelta, timezone

from helpers import FakeApi, make_record, server_error
from tools import CollectionSession

NOW = datetime(2026, 11, 20, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def make_session(record=None, **api_kwargs):
    api = FakeApi(record, **api_kwargs)
    return CollectionSession(api, api.get_candidate("c001"), now_fn=lambda: NOW), api


def confirmed_session(record=None, **api_kwargs):
    session, api = make_session(record, **api_kwargs)
    assert session.confirm_identity(True)["confirmed"]
    return session, api


# ---- status & privacy ---------------------------------------------------------------
def test_status_lists_needed_fields_without_leaking_stored_values():
    session, _ = make_session()
    status = session.get_candidate_status()
    assert [f["field"] for f in status["fields_needed"]] == ["email", "cgpa"]
    blob = json.dumps(status)
    for secret in ("+919876543210", "Test Institute", "Computer Engineering", "2026-06-01"):
        assert secret not in blob
    assert status["candidate_name"] == "Aarav Mehta"       # needed for the identity check
    assert "phone" not in status


def test_status_marks_invalid_on_file_and_includes_previous_notes():
    session, _ = make_session(make_record(email="broken", cgpa=8.4, notes="asked to call tomorrow"))
    status = session.get_candidate_status()
    needed = status["fields_needed"]
    assert needed[0]["field"] == "email" and "invalid" in needed[0]["problem"]
    assert "broken" not in json.dumps(status)
    assert status["previous_notes"] == "asked to call tomorrow"


# ---- identity gate --------------------------------------------------------------------
def test_nothing_recorded_before_identity_confirmed():
    session, api = make_session()
    for result in (session.dispatch("record_answers", {"answers": {"cgpa": "8"}}),
                   session.dispatch("save_progress", {}),
                   session.dispatch("finish_conversation", {"outcome": "completed"})):
        assert "Identity not confirmed" in result["error"]
    assert api.patches == [] and session.staged == {}


def test_wrong_person_escalates_and_saves_no_data():
    session, api = make_session()
    result = session.dispatch("confirm_identity", {"is_correct_person": False, "details": "wrong number"})
    assert result["recorded"] and session.ended and session.outcome == "needs_human"
    body = api.patches[0]
    assert body["collection_status"] == "needs_human" and body["attempt_count"] == 3
    assert "wrong_person" in body["notes"]
    assert set(body) == {"collection_status", "attempt_count", "last_contacted_at", "notes"}


# ---- record_answers ----------------------------------------------------------------------
def test_answers_validated_and_normalised():
    session, _ = confirmed_session()
    out = session.record_answers({"email": " a@b.co ", "cgpa": "8.456"})
    assert out["results"]["email"] == {"status": "accepted", "value": "a@b.co"}
    assert out["results"]["cgpa"]["value"] == 8.46
    assert out["all_mandatory_fields_ready"] and out["still_needed"] == []


def test_invalid_and_forbidden_answers_rejected():
    session, _ = confirmed_session()
    out = session.record_answers({"cgpa": "85", "email": "nope", "phone": "+911111111111",
                                  "full_name": "Other Name", "id": "c9", "shoe_size": "9"})
    results = out["results"]
    assert results["cgpa"]["status"] == "invalid" and results["email"]["status"] == "invalid"
    for forbidden in ("phone", "full_name", "id", "shoe_size"):
        assert results[forbidden]["status"] == "rejected"
    assert session.staged == {} and not out["all_mandatory_fields_ready"]


def test_ambiguous_date_is_not_guessed():
    session, _ = confirmed_session(make_record(available_from=None))
    out = session.record_answers({"available_from": "01/02/2027"})
    assert out["results"]["available_from"]["status"] == "invalid"
    assert "ambiguous" in out["results"]["available_from"]["reason"]
    assert "available_from" not in session.staged


def test_conflicting_answers_in_same_chat_need_confirmation():
    session, _ = confirmed_session()
    session.record_answers({"cgpa": "8.4"})
    out = session.record_answers({"cgpa": "9.1"})
    assert out["results"]["cgpa"]["status"] == "conflicts_with_earlier_answer"
    assert session.staged["cgpa"] == 8.4
    out = session.record_answers({"cgpa": "9.1"}, confirmed_corrections=["cgpa"])
    assert out["results"]["cgpa"]["status"] == "accepted" and session.staged["cgpa"] == 9.1


def test_valid_stored_value_needs_explicit_confirmation_to_change():
    session, api = confirmed_session()
    out = session.record_answers({"college": "Other College"})
    assert out["results"]["college"]["status"] == "needs_confirmation" and "college" not in session.staged
    assert session.record_answers({"college": "Test Institute"})["results"]["college"]["status"] == "already_on_file"
    out = session.record_answers({"college": "Other College"}, confirmed_corrections=["college"])
    assert out["results"]["college"]["status"] == "accepted"
    session.save_progress()
    assert api.patches[0] == {"college": "Other College"}


# ---- save_progress -----------------------------------------------------------------------
def test_save_progress_writes_only_validated_fields_without_tracking_changes():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co", "cgpa": "999"})
    out = session.save_progress()
    assert out["saved_fields"] == ["email"] and out["still_needed"] == ["cgpa"]
    assert api.patches == [{"email": "a@b.co"}] and api.versions == [1]
    assert session.record["version"] == 2 and not session.ended
    assert session.save_progress()["saved_fields"] == []          # nothing pending: no API call
    assert len(api.attempts) == 1


def test_save_failure_keeps_answers_staged_and_reports_error():
    session, api = confirmed_session(script=[server_error()])
    session.record_answers({"email": "a@b.co"})
    out = session.dispatch("save_progress", {})
    assert "Database unavailable" in out["error"] and "email" not in session.saved
    assert session.staged == {"email": "a@b.co"}
    assert session.dispatch("save_progress", {})["saved_fields"] == ["email"]   # explicit retry works


# ---- finish_conversation -------------------------------------------------------------------
def test_completed_refused_while_fields_outstanding():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co"})
    out = session.finish_conversation("completed")
    assert out["still_needed"] == ["cgpa"] and not session.ended and api.attempts == []


def test_completed_is_one_guarded_write_with_tracking_and_appended_notes():
    session, api = confirmed_session(make_record(notes="earlier note"))
    session.record_answers({"email": "a@b.co", "cgpa": "8.4"})
    out = session.finish_conversation("completed", note="all good")
    assert out["recorded"] and session.ended and session.outcome == "completed"
    assert len(api.attempts) == 1 and api.versions == [1]
    body = api.patches[0]
    assert body["email"] == "a@b.co" and body["cgpa"] == 8.4
    assert body["collection_status"] == "completed"
    assert body["attempt_count"] == 3                              # 2 stored + 1
    assert body["last_contacted_at"] == "2026-11-20T15:30:00+05:30"
    assert body["notes"].startswith("earlier note\n[2026-11-20 15:30] completed:")
    assert set(body) <= {"email", "cgpa", "collection_status", "attempt_count", "last_contacted_at", "notes"}
    assert "phone" not in body and "id" not in body


def test_completed_when_invalid_stored_value_is_corrected():
    session, api = confirmed_session(make_record(email="broken", cgpa=8.4))
    assert session.finish_conversation("completed")["still_needed"] == ["email"]
    session.record_answers({"email": "fixed@x.com"})
    assert session.finish_conversation("completed")["recorded"]
    assert api.record["email"] == "fixed@x.com" and api.record["collection_status"] == "completed"


def test_completed_never_written_if_record_changes_to_incomplete_during_conflict():
    # While we write, another actor blanks the college. Re-evaluation must refuse 'completed'.
    session, api = confirmed_session(script=["conflict"], concurrent_changes={"college": None})
    session.record_answers({"email": "a@b.co", "cgpa": "8.4"})
    out = session.dispatch("finish_conversation", {"outcome": "completed"})
    assert out["still_needed"] == ["college"]
    assert api.patches == [] and api.record["collection_status"] == "pending"


def test_refused_requires_outstanding_fields_and_records_them():
    session, api = confirmed_session()
    assert "error" in session.finish_conversation("refused")
    assert "error" in session.finish_conversation("refused", refused_fields=["phone"])
    session.record_answers({"email": "a@b.co"})
    out = session.finish_conversation("refused", refused_fields=["cgpa"])
    assert out["recorded"]
    body = api.patches[0]
    assert body["collection_status"] == "refused" and body["email"] == "a@b.co"
    assert "declined to provide: cgpa" in body["notes"]


def test_no_answer_rejected_once_information_collected():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co"})
    assert "in_progress" in session.finish_conversation("no_answer")["error"]
    assert api.attempts == []


def test_in_progress_records_what_is_left_and_suggested_time():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co"})
    session.finish_conversation("in_progress", follow_up_time="tomorrow 5pm")
    body = api.patches[0]
    assert body["collection_status"] == "in_progress" and body["email"] == "a@b.co"
    assert "Still needed: cgpa" in body["notes"] and "tomorrow 5pm" in body["notes"]


def test_invalid_outcome_rejected():
    session, _ = confirmed_session()
    assert "error" in session.finish_conversation("needs_human")
    assert "error" in session.finish_conversation("done")


def test_conflict_during_finish_reevaluates_and_increments_from_fresh_value():
    session, api = confirmed_session(script=["conflict"],
                                     concurrent_changes={"cgpa": 9.0, "attempt_count": 5})
    session.record_answers({"email": "a@b.co", "cgpa": "8.4"})
    assert session.finish_conversation("completed")["recorded"]
    assert api.versions == [1, 2] and len(api.attempts) == 2
    body = api.patches[0]
    assert "cgpa" not in body                       # someone else's valid value is not clobbered
    assert body["attempt_count"] == 6 and api.record["cgpa"] == 9.0


# ---- escalation & exits -----------------------------------------------------------------------
def test_escalation_saves_valid_answers_but_not_for_wrong_person():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co"})
    out = session.escalate_to_human("record disputed", "contradiction")
    assert out["recorded"] and api.patches[0]["email"] == "a@b.co"
    assert api.patches[0]["collection_status"] == "needs_human"

    session2, api2 = confirmed_session()
    session2.record_answers({"email": "a@b.co"})
    session2.escalate_to_human("not them", "wrong_person")
    assert "email" not in api2.patches[0]


def test_nothing_works_after_conversation_ended():
    session, api = confirmed_session()
    session.escalate_to_human("upset", "upset")
    for name, args in (("record_answers", {"answers": {"cgpa": "8"}}), ("save_progress", {}),
                       ("finish_conversation", {"outcome": "in_progress"}),
                       ("escalate_to_human", {"reason": "x", "category": "other"})):
        assert "already ended" in session.dispatch(name, args)["error"]
    assert len(api.attempts) == 1


def test_exit_without_information_is_no_answer():
    session, api = make_session()
    session.finish_from_exit("candidate typed /quit", "left")
    body = api.patches[0]
    assert body["collection_status"] == "no_answer" and body["attempt_count"] == 3
    assert body["last_contacted_at"] and "any information" in body["notes"]


def test_exit_after_partial_information_is_in_progress():
    session, api = confirmed_session()
    session.record_answers({"email": "a@b.co"})
    session.finish_from_exit("input closed", "left")
    body = api.patches[0]
    assert body["collection_status"] == "in_progress" and body["email"] == "a@b.co"


def test_later_and_technical_exits_are_in_progress_even_with_nothing_collected():
    for kind in ("later", "technical"):
        session, api = make_session()
        session.finish_from_exit("reason", kind)
        assert api.patches[0]["collection_status"] == "in_progress"


def test_exit_is_recorded_once_and_survives_api_failure():
    session, api = make_session(script=[server_error()])
    out = session.finish_from_exit("quit", "left")
    assert "could not record" in out["error"] and not session.ended
    session2, api2 = make_session()
    session2.finish_from_exit("quit", "left")
    assert session2.finish_from_exit("again", "left") is None and len(api2.patches) == 1


def test_dispatch_handles_unknown_tools_and_bad_arguments():
    session, _ = confirmed_session()
    assert "unknown tool" in session.dispatch("drop_database", {})["error"]
    assert "invalid arguments" in session.dispatch("record_answers", {"nonsense": 1})["error"]
    assert "error" in session.dispatch("record_answers", {"answers": "not a dict"})
