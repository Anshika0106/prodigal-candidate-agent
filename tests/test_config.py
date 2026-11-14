import pytest

from config import ConfigError, load_config

KEY, LLM_KEY = "dummy-db-key-123", "dummy-llm-key-456"


def set_env(monkeypatch, **overrides):
    values = {
        "CANDIDATE_DB_BASE_URL": "https://example.test/",
        "CANDIDATE_DB_API_KEY": KEY,
        "LLM_API_KEY": LLM_KEY,
        "LLM_MODEL": "m",
        "PARTICIPANT_NAME": "tester",
    }
    values.update(overrides)
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_repr_and_str_hide_both_keys(monkeypatch):
    set_env(monkeypatch)
    cfg = load_config()
    assert cfg.api_key == KEY and cfg.llm_api_key == LLM_KEY  # values still usable
    for text in (repr(cfg), str(cfg), f"{cfg!r}"):
        assert KEY not in text and LLM_KEY not in text
    assert cfg.api_base_url == "https://example.test"  # trailing slash stripped


def test_missing_variables_named_in_error(monkeypatch):
    set_env(monkeypatch, CANDIDATE_DB_BASE_URL=None, CANDIDATE_DB_API_KEY="  ")
    with pytest.raises(ConfigError) as err:
        load_config()
    assert "CANDIDATE_DB_BASE_URL" in str(err.value)
    assert "CANDIDATE_DB_API_KEY" in str(err.value)


def test_placeholder_key_rejected_without_echoing_it(monkeypatch):
    set_env(monkeypatch, CANDIDATE_DB_API_KEY="your_candidate_db_key_here")
    with pytest.raises(ConfigError) as err:
        load_config()
    assert "CANDIDATE_DB_API_KEY" in str(err.value)
    assert "your_candidate_db_key_here" not in str(err.value)


def test_llm_settings_optional(monkeypatch):
    set_env(monkeypatch, LLM_API_KEY=None, LLM_MODEL=None)
    assert load_config().llm_api_key == ""
