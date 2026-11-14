"""Read-only inspection of the candidate database API.

Uses only GET calls (via CandidateApiClient): schema, candidate list, candidate records.

Privacy: prints only candidate IDs, collection_status, attempt_count and field NAMES
with generic reasons. No candidate values are printed; only the public schema is saved.
"""
import json
import sys
from pathlib import Path

from api_client import ApiError, CandidateApiClient
from config import ConfigError, load_config
from validators import MANDATORY_FIELDS, analyse_record

SCHEMA_PATH = Path("runs") / "api_schema.json"


def save_schema(client: CandidateApiClient) -> dict:
    schema = client.get_schema()
    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"[schema] OK - {len(schema['mandatory_fields'])} mandatory fields; saved to {SCHEMA_PATH}")
    if set(schema["mandatory_fields"]) != set(MANDATORY_FIELDS):
        print("[schema] WARNING: mandatory fields differ from validators.MANDATORY_FIELDS")
    return schema


def report_candidate(client: CandidateApiClient, candidate_id: str, mandatory: list[str]) -> None:
    analysis = analyse_record(client.get_candidate(candidate_id), mandatory)
    print(f"\n{candidate_id}")
    print(f"  collection_status : {analysis.collection_status}")
    print(f"  attempt_count     : {analysis.attempt_count}")
    print(f"  missing mandatory : {', '.join(analysis.missing) or '-'}")
    for name, reason in analysis.invalid.items():
        print(f"  INVALID mandatory : {name} ({reason})")
    for name, reason in analysis.optional_invalid.items():
        print(f"  invalid optional  : {name} ({reason})  [does not block completion]")
    print(f"  optional populated: {', '.join(analysis.optional_populated) or '-'}")
    if analysis.blocked_read_only:
        print(f"  NOTE: read-only problem field(s): {', '.join(analysis.blocked_read_only)}")
    if not analysis.needs_collection:
        print("  => nothing to collect")


def main() -> int:
    try:
        client = CandidateApiClient(load_config())
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 1
    try:
        schema = save_schema(client)
        ids = [c["id"] for c in client.list_candidates()]
        print(f"[candidates] found {len(ids)}: {', '.join(ids)}")
        failures = 0
        for candidate_id in ids:
            try:
                report_candidate(client, candidate_id, schema["mandatory_fields"])
            except ApiError as exc:
                failures += 1
                print(f"\n{candidate_id}\n  ERROR: {exc}")
        return 1 if failures else 0
    except ApiError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
