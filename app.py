"""Terminal entry point."""
from agent import CandidateAgent
from api_client import CandidateApiClient
from config import load_config


def main() -> None:
    config = load_config()
    agent = CandidateAgent(config, CandidateApiClient(config))
    print("Candidate agent scaffold. Not implemented yet.")
    _ = agent


if __name__ == "__main__":
    main()
