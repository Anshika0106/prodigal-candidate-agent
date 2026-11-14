"""Thin wrapper around the candidate REST API."""
from config import Config


class CandidateApiClient:
    def __init__(self, config: Config):
        self.config = config

    def get_candidate(self, candidate_id: str) -> dict:
        """Fetch a candidate record."""
        raise NotImplementedError

    def update_candidate(self, candidate_id: str, fields: dict) -> dict:
        """Update fields on a candidate record."""
        raise NotImplementedError
