"""Configuration loaded from environment variables (never hardcode secrets)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    participant_name: str
    api_base_url: str
    api_key: str
    llm_api_key: str
    llm_model: str


def load_config() -> Config:
    """Build a Config from the environment. Raises if required values are missing."""
    required = ["CANDIDATE_DB_BASE_URL", "CANDIDATE_DB_API_KEY"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return Config(
        participant_name=os.environ.get("PARTICIPANT_NAME", ""),
        api_base_url=os.environ["CANDIDATE_DB_BASE_URL"].rstrip("/"),
        api_key=os.environ["CANDIDATE_DB_API_KEY"],
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", ""),
    )
