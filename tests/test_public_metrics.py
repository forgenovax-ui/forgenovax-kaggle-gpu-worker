import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app

API_KEY = "test-api-key-with-at-least-24-characters"


def client(metrics: Path | None = None, kill_switch: Path | None = None) -> TestClient:
    settings = Settings(
        api_key=API_KEY,
        public_metrics_path=str(metrics) if metrics else "",
        public_metrics_kill_switch_path=str(kill_switch) if kill_switch else "",
    )
    return TestClient(
        create_app(
            settings,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )
    )


def test_public_metrics_is_read_only_allowlisted_and_unauthenticated(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "experiment_id": "FNX-R002",
                "model": "FNX-1 Reflex",
                "status": "COVERAGE_SWEEP",
                "system_accuracy": 0.94,
                "raw_prompt": "private",
                "environment": {"FORGENOVAX_API_KEY": "secret"},
                "private_path": "/kaggle/working/private.jsonl",
                "traceback": "Traceback secret",
                "dataset": [{"private": True}],
                "credentials": "ghp_abcdefghijklmnopqrstuvwxyz123456",
            }
        )
    )

    with client(metrics) as test_client:
        response = test_client.get("/fnx/live/metrics")
        post = test_client.post("/fnx/live/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "experiment_id": "FNX-R002",
        "model": "FNX-1 Reflex",
        "status": "COVERAGE_SWEEP",
        "system_accuracy": 0.94,
    }
    serialized = response.text.lower()
    for forbidden in (
        "raw_prompt",
        "environment",
        "api_key",
        "/kaggle",
        "traceback",
        "dataset",
        "credential",
        "ghp_",
    ):
        assert forbidden not in serialized
    assert response.headers["cache-control"] == "no-store"
    assert post.status_code == 405


def test_public_metrics_disabled_by_default_and_kill_switch(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"experiment_id": "FNX-R002", "status": "PREPARING"}))
    kill_switch = tmp_path / "PUBLIC_DISABLED"

    with client() as test_client:
        unavailable = test_client.get("/fnx/live/metrics")
    kill_switch.touch()
    with client(metrics, kill_switch) as test_client:
        disabled = test_client.get("/fnx/live/metrics")

    assert unavailable.status_code == 503
    assert unavailable.json() == {"status": "TELEMETRY_UNAVAILABLE"}
    assert disabled.status_code == 503
    assert disabled.json() == {"status": "PUBLIC_TELEMETRY_DISABLED"}


def test_public_metrics_rejects_invalid_or_non_r002_payload(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    for content in ("not-json", json.dumps({"experiment_id": "OTHER", "status": "COMPLETE"})):
        metrics.write_text(content)
        with client(metrics) as test_client:
            response = test_client.get("/fnx/live/metrics")
        assert response.status_code == 503
        assert response.json() == {"status": "TELEMETRY_UNAVAILABLE"}
