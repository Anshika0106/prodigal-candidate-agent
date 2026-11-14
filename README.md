# Candidate Information Agent

Terminal agent that collects missing/invalid candidate application data and updates a REST API.

## Setup (Python 3.12)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then fill in your own values
```

## Run

```
python app.py
```

## Test

```
pytest
```

## Layout

- `app.py` – terminal loop
- `agent.py` – conversation logic
- `tools.py` – validation and API-backed tools
- `api_client.py` – REST client
- `config.py` – env-based configuration
- `runs/` – sample conversation logs

Status: scaffold only. LLM provider TBD.
