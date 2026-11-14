"""Tool functions the agent can call (validation, API read/update)."""
from api_client import CandidateApiClient


def find_missing_fields(candidate: dict) -> list[str]:
    """Return names of fields that are missing or invalid."""
    raise NotImplementedError


def validate_field(name: str, value: str) -> tuple[bool, str]:
    """Return (is_valid, message) for a candidate-supplied value."""
    raise NotImplementedError


def save_fields(client: CandidateApiClient, candidate_id: str, fields: dict) -> dict:
    """Persist validated fields via the API."""
    raise NotImplementedError
