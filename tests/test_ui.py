import ui
from agent import CandidateAgent
from helpers import FakeApi, Inputs, ScriptedLLM, call, turn
from tools import CollectionSession


def run_conversation(capsys, llm, inputs):
    api = FakeApi()
    session = CollectionSession(api, api.get_candidate("c001"))
    view = ui.ChatView(session, len(session.analysis.fields_to_collect))
    agent = CandidateAgent(session, llm, view.agent_message, Inputs(*inputs), view.on_event)
    agent.run()
    return session, api, capsys.readouterr().out


def test_progress_bar_bounds():
    assert ui.progress_bar(0, 4, width=8) == "[--------] 0/4"
    assert ui.progress_bar(2, 4, width=8) == "[####----] 2/4"
    assert ui.progress_bar(9, 4, width=8) == "[########] 4/4"      # clamped
    assert ui.progress_bar(0, 0, width=4).endswith("0/1")         # no division by zero


def test_no_colour_codes_when_not_a_terminal(capsys):
    assert ui.style("x", "bold", "red") == "x"
    ui.banner("model-x")
    ui.candidate_panel("c001", "pending", 0, ["email", "cgpa"])
    ui.commands_hint()
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "PRODIGAL CANDIDATE INFORMATION AGENT" in out and "model-x" in out
    assert "c001" in out and "2 field(s): email, cgpa" in out and "/quit" in out and "/later" in out


def test_colour_used_on_a_terminal(monkeypatch):
    monkeypatch.setattr(ui, "color_enabled", lambda: True)
    assert ui.style("hi", "bold", "green") == "\x1b[1;32mhi\x1b[0m"


def test_no_colour_env_var_disables_colour(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True, raising=False)
    assert ui.color_enabled() is False


def test_agent_message_is_labelled_and_wrapped(capsys):
    ui.agent_message("word " * 40)
    out = capsys.readouterr().out
    assert out.strip().startswith("Agent:")
    assert max(len(line) for line in out.splitlines()) <= ui.WIDTH + 10


def test_conversation_shows_validation_saves_progress_and_hides_personal_values(capsys):
    llm = ScriptedLLM(
        turn("Hi Aarav, am I speaking with Aarav Mehta?"),
        turn("", call("confirm_identity", is_correct_person=True)),
        turn("Thanks! Your email and CGPA?"),
        turn("", call("record_answers", answers={"email": "secret.person@example.com", "cgpa": "85%"})),
        turn("That CGPA looks like a percentage - what is it out of 10?"),
        turn("", call("record_answers", answers={"cgpa": "8.4 out of 10"}),
             call("save_progress")),
        turn("", call("finish_conversation", outcome="completed")),
        turn("All done, thanks!"),
    )
    session, api, out = run_conversation(capsys, llm, ["yes", "secret.person@example.com and 85%", "8.4 out of 10"])
    assert "Agent: Hi Aarav" in out
    assert "[ok]" in out and "identity confirmed" in out
    assert "email validated" in out and "cgpa validated -> 8.4" in out
    assert "[x]" in out and "cgpa must be on the 10-point scale" in out        # invalid answer explained
    assert "[saved]" in out and "cgpa, email" in out
    assert "Progress [" in out and "2/2" in out and "saved to database: 2" in out
    assert "outcome recorded: completed" in out
    assert "secret.person" not in out                                           # email value never shown
    assert session.outcome == "completed"                                       # behaviour unchanged


def test_confirmation_and_rejection_states_are_rendered(capsys):
    llm = ScriptedLLM(
        turn("Hi?"), turn("", call("confirm_identity", is_correct_person=True)), turn("Details?"),
        turn("", call("record_answers", answers={"college": "Other College", "phone": "+911", "email": "a@b.co"})),
        turn("Anything else?"),
    )
    session, api, out = run_conversation(capsys, llm, ["yes", "x", "/quit"])
    assert "college: needs the candidate to confirm" in out
    assert "phone: cannot be changed here" in out
    assert "email validated" in out


def test_final_summary_for_each_outcome(capsys):
    for status, text in (("completed", "COMPLETED"), ("in_progress", "IN_PROGRESS"),
                         ("refused", "REFUSED"), ("needs_human", "NEEDS_HUMAN"), ("no_answer", "NO_ANSWER")):
        ui.final_summary("c001", {"collection_status": status, "attempt_count": 3},
                         ["email"], ["cgpa"], "runs/t.json")
        out = capsys.readouterr().out
        assert "COLLECTION RESULT" in out and text in out
        assert "Attempts" in out and "email" in out and "cgpa" in out and "runs/t.json" in out
    ui.final_summary("c001", {"collection_status": "completed", "attempt_count": 1}, [], [])
    assert "none" in capsys.readouterr().out
