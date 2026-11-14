# Candidate Information Agent

A terminal chat agent that takes **one candidate ID**, works out what is missing or wrong in their
application record, chats with the candidate (Google Gemini, genuine function calling), validates
what they say, writes it to the candidate database with safe PATCHes, and records the outcome.

## Setup and run (Python 3.12)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then fill in CANDIDATE_DB_API_KEY, LLM_API_KEY, LLM_MODEL
python app.py --candidate-id c001
```

`.env` is git-ignored; never commit keys. `LLM_MODEL` was verified with `gemini-3.5-flash`
(listed by the SDK and tested with a real function call).

Options: `--idle-timeout SECONDS` (default 600; 0 = never), `--transcript-dir runs`,
`--mark-complete-if-done` (see decisions below).

**Ending a chat.** The candidate can type `/quit` (leave now), `/later` (continue later), press
Ctrl+C, close input (EOF), or simply stop replying (idle timeout). Or just say so in words:
the LLM decides. Every path records an outcome in the database.

Other commands: `pytest` (all tests, no network), `python inspect_api.py` (read-only API report that
prints no personal values).

## Architecture

```
app.py            CLI: load candidate -> nothing missing? stop : start chat
  agent.py        system prompt + tool-calling loop; /quit /later Ctrl+C EOF idle-timeout handling
    llm.py        Gemini chat (google-genai), automatic function calling DISABLED, retry/backoff
    tools.py      CollectionSession: the 6 tools + all safety rules (the ONLY path to the data)
      services.py   plan/apply updates: minimal PATCH, If-Match, 409 re-evaluation, notes/attempts
        api_client.py  HTTP client (Bearer, 15s timeout, Idempotency-Key, distinct errors, no retries)
        validators.py  validation + conservative normalisation for every schema field
  transcript.py   redacted run transcripts (chat + tool calls + API calls)
```

**Agent loop.** Per candidate message: send it to Gemini -> while Gemini requests tools, run them in
Python and send the results back -> print the reply -> read the next message. Multiple tool calls in
one turn and tool errors are handled; a runaway tool loop is cut off after 8 rounds.

**Tools the LLM can call** (all enforced in Python, not in the prompt):

| Tool | What it does |
|---|---|
| `get_candidate_status` | name, which mandatory fields are needed and why, what is saved/pending, previous notes. **Never returns stored values** |
| `confirm_identity` | unlocks collection; `false` (wrong person) escalates and discards everything said |
| `record_answers` | validates and normalises answers, stages the accepted ones; reports invalid / ambiguous / conflicting / needs-confirmation |
| `save_progress` | writes staged answers (guarded PATCH) |
| `finish_conversation` | `completed` / `in_progress` / `refused` / `no_answer`, with guards (below) |
| `escalate_to_human` | `needs_human` with a reason (wrong person, contradiction, upset, other) |

**State.** The database record is the source of truth. In memory (`CollectionSession`): the last-read
record (with `version`), staged-but-unsaved answers, explicit corrections, fields saved this chat,
identity-confirmed flag, outcome.

**Data sent to the LLM:** the candidate's name, the list of fields still needed (names, not values),
previous notes, and the conversation. Stored email/phone/college etc. are never sent.

## Decisions (edge cases)

- **Nothing missing (e.g. c003, c006):** no conversation, no write. With `--mark-complete-if-done`
  the app sets only `collection_status=completed` (and only if the status is `pending`/`in_progress`);
  attempt count and last-contacted are untouched because no contact happened.
- **Existing data is checked, not trusted:** a stored value that is the wrong type or not canonical
  (CGPA 85, graduation year 2035, `"btech"`, malformed email) counts as invalid and is collected again.
  Invalid *optional* fields (LinkedIn/GitHub/skills) never block completion.
- **Validation:** every value passes `validators.validate_field`; the LLM passes the candidate's raw
  words and Python normalises (`b tech` -> `B.Tech`, `Bangalore` -> `Bengaluru`, `8.4 out of 10` -> `8.40`).
  Nothing is guessed: `27` is not turned into 2027, `next month` is not turned into a date,
  `01/02/2027` (ambiguous), `March 2027` (no day), `85%` (not a 10-point CGPA) are rejected with a reason
  so the agent asks again.
- **No overwriting good data:** a valid stored value changes only if the candidate explicitly confirms
  the correction (`confirmed_corrections`); a conflicting answer within the same chat needs
  confirmation too. Values equal to what is stored are not re-sent. `phone`, `id`, `version`,
  `updated_at` can never be written (blocked in validators, services and the HTTP client).
- **Outcomes** (all through one guarded write that also increments `attempt_count` from the stored
  value, sets `last_contacted_at`, and **appends** to `notes`, never replacing them):
  - `completed`: refused by Python unless every mandatory field would be valid in the database; this is
    re-checked after any 409.
  - `in_progress`: candidate wants to continue later / leaves after giving some information; the note lists
    what is left and any time they suggested. `/later` always records this.
  - `no_answer`: nothing was collected and the candidate left (`/quit`, EOF, Ctrl+C, idle timeout).
  - `refused`: candidate declined named fields; the note lists them; the rest is still saved.
  - `needs_human`: wrong person (nothing saved), contradictions, upset candidate, unresolved issues.
  - Our own failure (Gemini down/rate-limited): recorded as `in_progress` with a note; answers already
    saved are kept.
- **409 conflicts:** the client never retries; `services.apply_updates` re-reads the record and rebuilds
  the whole update (including `attempt_count` and notes) from the fresh data.
- **Other write errors** are never blindly retried; the LLM is told the data is not saved.
- **Prompt injection:** candidate text is declared untrusted in the prompt, but the real protection is
  that the tools enforce every rule (tested with a manipulated LLM trying to write `phone`, set
  `completed` early, and touch another record).

## Sample runs (`runs/`)

Redacted transcripts (name, phone, email and profile URLs masked) with the chat, every tool call
and result, and every API call with its response. These are **real runs** against the live API and live
Gemini. The candidate's replies were written by the assistant that built this project as test
input (not by a real candidate), and the database was not reset between runs.

- `transcript_c001_*`: confirms identity, asks what the data is for, gives an invalid email and a
  percentage instead of a CGPA, corrects both -> `completed`. Starts from an earlier `in_progress` record
  (an earlier run that failed on a bug, since fixed), so the existing note is preserved and appended to.
- `transcript_c007_*` (first): nine fields missing; multi-answer messages, `b tech` and `Bangalore`
  normalised, `27` and `next month` refused as ambiguous, progress saved in three PATCHes. Gemini then
  returned HTTP 429 (rate limit) and the app recorded `in_progress` with the right "still needed" note.
  The candidate's last message (continue tomorrow 6pm) was lost with the failure, so that part is *not*
  demonstrated here.
- `transcript_c007_*` (second, 4 events): a follow-up chat to finish the last field (`available_from`)
  that failed at the greeting because the Gemini API key's quota was exhausted
  (`RESOURCE_EXHAUSTED`). It records `in_progress` with a note and nothing else. This run also exposed a
  bug (a failure before any message was delivered still counted as a contact attempt, so c007 has
  `attempt_count=2`); it is fixed and unit-tested, but the record was not repaired.
- Not captured: a live run that ends in `refused`, `needs_human`, `no_answer`, or a natural-language
  "continue later" (blocked by the Gemini quota). Those paths are unit-tested with a scripted LLM only.

## Status

**Works (live-tested):** identity confirmation by name, multi-field answers, validation and
normalisation, clarification of invalid/ambiguous answers, progressive saving, `completed` outcome,
existing-note preservation, graceful handling of Gemini rate limiting. 244 unit tests pass (all use
fakes; no network).

**Not live-tested / known limits:**
- `refused`, `needs_human` (wrong person, contradiction), natural-language "continue later" and the
  idle timeout are covered by unit tests with a scripted LLM, but were not run against live Gemini.
- Free-tier Gemini limits (HTTP 429, including daily quota exhaustion) can interrupt or prevent a chat; we
  retry with backoff (about 95 s total) and then record `in_progress`. Use a key with enough quota for a
  live evaluation. The model's reply quality varies (it sometimes uses numbered lists).
- If a PATCH times out, its outcome is unknown; the agent re-reads, but a retry could double-count
  `attempt_count` if the first write had actually applied.
- Notes are appended with a timestamp; very long histories get truncated (existing text is never cut).
- The PATCH response shape was assumed to be a candidate record; if not, the app re-fetches with one GET.
- Enum aliases (cities, degrees) are a small hand-written list; unknown phrasings are asked again.
- Windows console only tested; Unicode output is forced to UTF-8.

**Next with more time:** live runs for every outcome; LLM-based review of free-text answers for
ambiguous branch/college names; resumable follow-ups driven by `notes`; an idempotent retry for unknown
PATCH outcomes; a small evaluation set of scripted candidate personas run against Gemini.
