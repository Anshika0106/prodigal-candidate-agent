"""Conversation agent: decides what to ask next and when to call tools."""
from api_client import CandidateApiClient
from config import Config


class CandidateAgent:
    def __init__(self, config: Config, client: CandidateApiClient):
        self.config = config
        self.client = client

    def respond(self, user_message: str) -> str:
        """Produce the agent's reply to one user message."""
        raise NotImplementedError
