from backend.app import config as config_mod
from backend.app.config import get_settings


def test_get_settings_reads_fatigue_thresholds_from_env(monkeypatch):
    monkeypatch.setenv("DMS_FATIGUE_YAWN_THRESHOLD", "0.7")
    monkeypatch.setenv("DMS_FATIGUE_LOOK_AWAY_THRESHOLD", "0.8")
    monkeypatch.setenv("DMS_FATIGUE_FATIGUE_THRESHOLD", "0.9")

    settings = get_settings()

    assert settings.fatigue_yawn_threshold == 0.7
    assert settings.fatigue_look_away_threshold == 0.8
    assert settings.fatigue_fatigue_threshold == 0.9


def test_get_settings_defaults_llm_timeout_to_sixty_seconds(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(config_mod, "load_dotenv_file", lambda path: None)

    settings = get_settings()

    assert settings.llm.timeout_seconds == 60
