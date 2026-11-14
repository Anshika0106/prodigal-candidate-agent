"""Terminal entry point:  python app.py --candidate-id c001

Chat commands:  /quit  leave now      /later  continue later      Ctrl+C / EOF  also end the chat
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

from agent import SYSTEM_PROMPT, CandidateAgent, IdleTimeout
from api_client import ApiError, CandidateApiClient
from config import ConfigError, load_config
from llm import GeminiChat
from services import IncompleteRecordError, apply_updates
from tools import TOOL_SPECS, CollectionSession
from transcript import RecordingClient, Transcript
from validators import analyse_record

DEFAULT_IDLE_TIMEOUT = 600  # seconds without a candidate reply before the chat is closed


class ConsoleInput:
    """Reads candidate lines on a background thread so an idle timeout and Ctrl+C both work."""

    def __init__(self, idle_timeout: float):
        self._idle, self._queue, self._started = idle_timeout, queue.Queue(), False

    def _pump(self) -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                self._queue.put(None)
                return
            self._queue.put(line.rstrip("\r\n"))

    def read(self) -> str | None:
        if not self._started:
            threading.Thread(target=self._pump, daemon=True).start()
            self._started = True
        print("You: ", end="", flush=True)
        deadline = time.monotonic() + self._idle if self._idle > 0 else None
        while True:
            try:
                line = self._queue.get(timeout=0.5)   # short polls keep Ctrl+C responsive
            except queue.Empty:
                if deadline and time.monotonic() >= deadline:
                    print()
                    raise IdleTimeout
                continue
            if not sys.stdin.isatty() and line is not None:
                print(line)                           # echo piped input so the terminal log is readable
            return line


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat agent that completes a candidate's application.")
    p.add_argument("--candidate-id", required=True, help="e.g. c001")
    p.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT,
                   help="seconds of silence before the chat is closed (0 = never)")
    p.add_argument("--transcript-dir", default="runs", help="where to save the redacted transcript")
    p.add_argument("--mark-complete-if-done", action="store_true",
                   help="if nothing is missing, set collection_status=completed (no chat, no other change)")
    return p.parse_args(argv)


def say(text: str) -> None:
    print(f"\nAgent: {text}\n", flush=True)


def handle_nothing_to_collect(client, record: dict, mark_complete: bool) -> None:
    print("Nothing is missing or invalid - no conversation needed.")
    if not mark_complete:
        return
    if record.get("collection_status") not in ("pending", "in_progress"):
        print(f"collection_status is '{record.get('collection_status')}'; left unchanged.")
        return
    try:
        result = apply_updates(client, record["id"], record, status="completed", require_complete=True)
    except (ApiError, IncompleteRecordError) as exc:
        print(f"Could not mark completed: {exc}")
        return
    print("collection_status set to 'completed'." if result.sent else "Already completed; no write.")


def main(argv=None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    if not config.llm_api_key or not config.llm_model:
        print("Configuration error: set LLM_API_KEY and LLM_MODEL in .env")
        return 1

    transcript = Transcript()
    client = RecordingClient(CandidateApiClient(config), transcript.add)
    try:
        record = client.get_candidate(args.candidate_id)
    except (ApiError, ValueError) as exc:
        print(f"Could not load candidate: {exc}")
        return 1

    analysis = analyse_record(record)
    print(f"Candidate {record['id']}: collection_status={analysis.collection_status}, "
          f"attempt_count={analysis.attempt_count}")
    if analysis.blocked_read_only:
        print(f"Read-only field problem ({', '.join(analysis.blocked_read_only)}): needs a human; "
              "the agent cannot fix it.")
    if not analysis.needs_collection:
        handle_nothing_to_collect(client, record, args.mark_complete_if_done)
        return 0
    print(f"Fields to collect: {', '.join(analysis.fields_to_collect)}")

    session = CollectionSession(client, record)
    try:
        llm = GeminiChat(config.llm_api_key, config.llm_model, SYSTEM_PROMPT, TOOL_SPECS)
    except Exception as exc:
        print(f"Could not start the LLM client ({type(exc).__name__}).")
        return 1

    print("Chat started. Type /quit to leave or /later to continue later.\n")
    agent = CandidateAgent(session, llm, say, ConsoleInput(args.idle_timeout).read, transcript.add)
    try:
        agent.run()
    finally:
        meta = {"model": config.llm_model, "outcome": session.outcome,
                "final_collection_status": session.record.get("collection_status"),
                "final_attempt_count": session.record.get("attempt_count"),
                "fields_saved": session.saved}
        path = transcript.save(Path(args.transcript_dir), record["id"], meta, record)
    final = analyse_record(session.record)
    print(f"\n[Chat ended] collection_status={session.record.get('collection_status')}, "
          f"attempt_count={session.record.get('attempt_count')}, "
          f"fields saved: {', '.join(session.saved) or 'none'}, "
          f"still needed: {', '.join(final.fields_to_collect) or 'none'}")
    print(f"Transcript saved to {path}")
    if agent.llm_failed:
        return 2
    return 0 if session.outcome else 1


if __name__ == "__main__":
    sys.exit(main())
