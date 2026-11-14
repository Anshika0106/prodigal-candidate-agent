import json

import pytest

import app
from api_client import ApiResponseError
from config import Config
from helpers import FakeApi, make_record
from transcript import RecordingClient, Redactor, Transcript

COMPLETE = make_record(email="aarav@example.com", cgpa=8.4, id="c003")


def patch_runtime(monkeypatch, record):
    api = FakeApi(record)
    monkeypatch.setattr(app, "load_config",
                        lambda: Config("p", "https://api.test", "dummy-db-key", "dummy-llm-key", "model-x"))
    monkeypatch.setattr(app, "CandidateApiClient", lambda config: api)

    def forbidden(*a, **k):
        raise AssertionError("LLM client must not be created when nothing is needed")
    monkeypatch.setattr(app, "GeminiChat", forbidden)
    return api


def test_no_conversation_for_complete_candidate(monkeypatch, capsys):
    api = patch_runtime(monkeypatch, COMPLETE)
    assert app.main(["--candidate-id", "c003"]) == 0
    out = capsys.readouterr().out
    assert "no conversation needed" in out
    assert api.attempts == [] and api.record["collection_status"] == "pending"   # nothing written


def test_mark_complete_flag_writes_only_status(monkeypatch, capsys):
    api = patch_runtime(monkeypatch, COMPLETE)
    assert app.main(["--candidate-id", "c003", "--mark-complete-if-done"]) == 0
    assert api.patches == [{"collection_status": "completed"}]
    assert api.record["attempt_count"] == 2          # no contact happened: tracking untouched


def test_mark_complete_flag_leaves_other_statuses_alone(monkeypatch):
    api = patch_runtime(monkeypatch, {**COMPLETE, "collection_status": "needs_human"})
    app.main(["--candidate-id", "c003", "--mark-complete-if-done"])
    assert api.attempts == []


def test_missing_config_and_unknown_candidate_fail_cleanly(monkeypatch, capsys):
    patch_runtime(monkeypatch, COMPLETE)
    monkeypatch.setattr(app, "load_config",
                        lambda: Config("p", "https://api.test", "k", "", ""))
    assert app.main(["--candidate-id", "c003"]) == 1
    assert "LLM_API_KEY" in capsys.readouterr().out

    class Missing:
        def get_candidate(self, cid):
            raise ApiResponseError("boom")
    monkeypatch.setattr(app, "load_config",
                        lambda: Config("p", "https://api.test", "k", "k", "m"))
    monkeypatch.setattr(app, "CandidateApiClient", lambda config: Missing())
    assert app.main(["--candidate-id", "c999"]) == 1


def test_redactor_masks_name_phone_email_and_profile_urls():
    r = Redactor(make_record())
    text = ("Hi Aarav Mehta, call +919876543210 or write to me@example.com, "
            "see https://www.linkedin.com/in/aarav and github.com/aarav. aarav!")
    out = r.text(text)
    for leaked in ("Aarav", "Mehta", "919876543210", "me@example.com", "linkedin", "github.com/aarav"):
        assert leaked.lower() not in out.lower()
    assert "[NAME]" in out and "[EMAIL]" in out


def test_redactor_masks_sensitive_keys_in_nested_data():
    r = Redactor(make_record())
    out = r.obj({"response": {"full_name": "Aarav Mehta", "phone": "+919876543210",
                              "email": "x@y.co", "cgpa": 8.4, "college": "Test Institute"},
                 "body": {"email": "x@y.co"}, "results": {"email": {"status": "accepted", "value": "x@y.co"}}})
    blob = json.dumps(out)
    assert "x@y.co" not in blob and "Mehta" not in blob and "919876543210" not in blob
    assert out["response"]["cgpa"] == 8.4 and out["results"]["email"]["status"] == "accepted"


def test_transcript_saved_redacted_with_api_calls(tmp_path):
    transcript = Transcript()
    api = FakeApi(make_record())
    client = RecordingClient(api, transcript.add)
    record = client.get_candidate("c001")
    transcript.add("agent_message", {"text": "Hi Aarav, is that you?"})
    transcript.add("candidate_message", {"text": "yes, mail me at me@example.com"})
    client.patch_candidate("c001", {"email": "me@example.com"}, record["version"])
    path = transcript.save(tmp_path, "c001", {"outcome": "completed"}, record)
    raw = path.read_text(encoding="utf-8")
    for leaked in ("Aarav", "Mehta", "me@example.com", "+919876543210"):
        assert leaked not in raw
    doc = json.loads(raw)
    assert doc["redacted"] and doc["candidate_id"] == "c001" and doc["outcome"] == "completed"
    types = [e["type"] for e in doc["events"]]
    assert types == ["api_call", "agent_message", "candidate_message", "api_call"]
    patch = doc["events"][-1]
    assert patch["method"] == "PATCH" and patch["if_match"] == '"1"' and patch["status"] == "ok"


def test_console_input_idle_timeout(monkeypatch):
    class Silent:
        def readline(self):
            import time
            time.sleep(5)
            return ""
        def isatty(self):
            return True
    monkeypatch.setattr(app.sys, "stdin", Silent())
    reader = app.ConsoleInput(idle_timeout=0.6)
    with pytest.raises(app.IdleTimeout):
        reader.read()


def test_redactor_does_not_mangle_timestamps_and_dates():
    out = Redactor(make_record()).text("[2026-11-14 15:16] in_progress at 2026-11-14T15:16:48+05:30, 9876543210")
    assert "2026-11-14 15:16" in out and "2026-11-14T15:16:48+05:30" in out
    assert "9876543210" not in out
