"""Configuration loaded from environment variables (never hardcode secrets)."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ("CANDIDATE_DB_BASE_URL", "CANDIDATE_DB_API_KEY")
PLACEHOLDER_PREFIX = "your_"


class ConfigError(RuntimeError):
    """Configuration is missing or invalid. The message never contains secret values."""


@dataclass(frozen=True)
class Config:
    participant_name: str
    api_base_url: str
    # repr=False: printing/logging a Config (or a traceback showing it) must never leak keys.
    api_key: str = field(repr=False)
    llm_api_key: str = field(repr=False)
    llm_model: str


def load_config() -> Config:
    """Build a Config from the environment, raising ConfigError with a clear message."""
    missing = [n for n in REQUIRED_VARS if not os.environ.get(n, "").strip()]
    if missing:
        raise ConfigError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your values."
        )
    placeholders = [n for n in REQUIRED_VARS if os.environ[n].strip().startswith(PLACEHOLDER_PREFIX)]
    if placeholders:
        raise ConfigError(f"Still set to placeholder values: {', '.join(placeholders)}")
    return Config(
        participant_name=os.environ.get("PARTICIPANT_NAME", ""),
        api_base_url=os.environ["CANDIDATE_DB_BASE_URL"].strip().rstrip("/"),
        api_key=os.environ["CANDIDATE_DB_API_KEY"].strip(),
        llm_api_key=os.environ.get("LLM_API_KEY", "").strip(),
        llm_model=os.environ.get("LLM_MODEL", "").strip(),
    )
