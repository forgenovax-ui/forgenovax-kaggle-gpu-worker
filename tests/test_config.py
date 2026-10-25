import pytest

from src.config import Settings


API_KEY = "test-api-key-with-at-least-24-characters"


def test_measured_context_is_the_environment_default(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_LENGTH", raising=False)
    assert Settings.from_env().context_length == 8192


@pytest.mark.parametrize(
    "settings,match",
    [
        (Settings(api_key=API_KEY, model="unknown:1b"), "MODEL must be one of"),
        (Settings(api_key=API_KEY, context_length=256), "CONTEXT_LENGTH"),
        (Settings(api_key=API_KEY, ollama_num_parallel=5), "OLLAMA_NUM_PARALLEL"),
    ],
)
def test_invalid_model_runtime_configuration_is_rejected(
    settings: Settings, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        settings.validate_for_startup()
