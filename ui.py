"""Terminal presentation only: banner, panels, message styling, progress and final summary.

No logic lives here. It renders what the app and the agent's event stream already report.
Colours are used only on a real terminal (and never when NO_COLOR is set).
"""
import os
import sys
import textwrap

WIDTH = 64
INDENT = "       "          # continuation indent for wrapped agent text
BAR_WIDTH = 20

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
          "magenta": "35", "cyan": "36"}

# Values of these fields are never echoed in the UI (the candidate typed them; names are enough).
HIDE_VALUE = {"email", "full_name", "phone", "linkedin_url", "github_url", "skills"}

STATUS_STYLE = {
    "completed": ("green", "All mandatory fields collected"),
    "in_progress": ("yellow", "Follow-up pending - some information still needed"),
    "no_answer": ("yellow", "Candidate did not provide any information"),
    "refused": ("magenta", "Candidate declined one or more fields"),
    "needs_human": ("red", "Needs a human to review"),
    "pending": ("dim", "No outcome recorded"),
}


def color_enabled() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def enable_ansi() -> None:
    """Turn on ANSI escape handling in Windows consoles (no-op elsewhere)."""
    if os.name == "nt" and color_enabled():
        os.system("")


def style(text: str, *names: str) -> str:
    if not color_enabled() or not names:
        return text
    codes = ";".join(_CODES[n] for n in names)
    return f"\x1b[{codes}m{text}\x1b[0m"


def rule(char: str = "=") -> str:
    return char * WIDTH


def banner(model: str = "") -> None:
    subtitle = "ACME Corp internship applications"
    if model:
        subtitle += f"  |  model: {model}"
    print(style(rule(), "cyan"))
    print(style("  PRODIGAL CANDIDATE INFORMATION AGENT", "bold", "cyan"))
    print(style(f"  {subtitle}", "dim"))
    print(style(rule(), "cyan"))


def candidate_panel(candidate_id: str, status, attempts, needed: list[str]) -> None:
    attempts = attempts if isinstance(attempts, int) else 0
    print(f"  {style('Candidate', 'bold'):<18}{candidate_id}")
    print(f"  {style('Status', 'bold'):<18}{status} (attempts so far: {attempts})")
    if needed:
        print(f"  {style('To collect', 'bold'):<18}{len(needed)} field(s): {', '.join(needed)}")
    else:
        print(f"  {style('To collect', 'bold'):<18}nothing")


def commands_hint() -> None:
    print(style("  Commands: /quit leave | /later continue later | Ctrl+C end chat", "dim"))
    print(style(rule("-"), "dim"))


def progress_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    total = max(total, 1)
    done = max(0, min(done, total))
    filled = round(width * done / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total}"


def prompt() -> str:
    return style("You: ", "bold", "green")


def agent_message(text: str) -> None:
    """Agent text, wrapped, with a coloured label so it stands apart from candidate input."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=WIDTH + 8 - len(INDENT)) or [""]
        lines.extend(wrapped)
    body = ("\n" + INDENT).join(lines)
    print(f"\n{style('Agent:', 'bold', 'cyan')} {body}\n", flush=True)


def note(tag: str, message: str, colour: str) -> None:
    print(f"  {style(tag.ljust(7), colour)} {message}", flush=True)


def info(message: str) -> None:
    print(message)


class ChatView:
    """Turns the agent's tool-call events into short status lines and a progress bar."""

    def __init__(self, session, total_fields: int):
        self.session, self.total = session, total_fields

    def agent_message(self, text: str) -> None:
        agent_message(text)

    def progress(self) -> None:
        collected = self.total - len(self.session.remaining_fields())
        print(style(f"  Progress {progress_bar(collected, self.total)}  "
                    f"(saved to database: {len(self.session.saved)})", "dim"), flush=True)

    def on_event(self, kind: str, data: dict) -> None:
        if kind != "tool_call":
            return
        name, result = data.get("name"), data.get("result") or {}
        if name == "confirm_identity" and result.get("confirmed"):
            note("[ok]", "identity confirmed", "green")
        elif name == "record_answers":
            self._show_answers(result)
        elif name == "save_progress":
            saved = result.get("saved_fields") or []
            if saved:
                note("[saved]", f"{', '.join(saved)}", "green")
            for field, reason in (result.get("not_saved") or {}).items():
                note("[!]", f"{field} not saved: {reason}", "yellow")
            if result.get("error"):
                note("[!]", "could not save right now (will retry)", "red")
            self.progress()
        elif name in ("finish_conversation", "escalate_to_human") and result.get("recorded"):
            note("[done]", f"outcome recorded: {result.get('collection_status')}", "green")

    def _show_answers(self, result: dict) -> None:
        if result.get("error"):
            return
        for field, res in (result.get("results") or {}).items():
            status = res.get("status")
            if status == "accepted":
                value = "" if field in HIDE_VALUE else f" -> {res.get('value')}"
                note("[ok]", f"{field} validated{value}", "green")
            elif status == "invalid":
                note("[x]", f"{field}: {res.get('reason')}", "red")
            elif status in ("needs_confirmation", "conflicts_with_earlier_answer"):
                note("[?]", f"{field}: needs the candidate to confirm", "yellow")
            elif status == "already_on_file":
                note("[=]", f"{field}: already on file", "dim")
            elif status == "rejected":
                note("[x]", f"{field}: cannot be changed here", "red")
        self.progress()


def final_summary(candidate_id: str, record: dict, saved: list[str], still_needed: list[str],
                  transcript_path=None) -> None:
    status = record.get("collection_status")
    colour, meaning = STATUS_STYLE.get(status, ("dim", ""))
    print()
    print(style(rule(), "cyan"))
    print(style("  COLLECTION RESULT", "bold", "cyan"))
    print(style(rule("-"), "dim"))
    print(f"  {style('Candidate', 'bold'):<18}{candidate_id}")
    print(f"  {style('Final status', 'bold'):<18}{style(str(status).upper(), 'bold', colour)}  - {meaning}")
    print(f"  {style('Attempts', 'bold'):<18}{record.get('attempt_count')}")
    print(f"  {style('Saved this chat', 'bold'):<18}{', '.join(saved) or 'none'}")
    print(f"  {style('Still needed', 'bold'):<18}{', '.join(still_needed) or 'none'}")
    if transcript_path:
        print(f"  {style('Transcript', 'bold'):<18}{transcript_path}")
    print(style(rule(), "cyan"))
