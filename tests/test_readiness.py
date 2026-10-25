from fastapi.testclient import TestClient
import httpx
import json

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"


def test_readiness_requires_auth_and_real_sentinel_inference() -> None:
    calls = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 128
        calls += 1
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "FORGENOVAX_READY"}},
        )

    app = create_app(
        Settings(api_key=API_KEY, readiness_cache_seconds=60),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        missing = client.get("/readyz")
        first = client.get("/readyz", headers={"Authorization": f"Bearer {API_KEY}"})
        cached = client.get("/readyz", headers={"Authorization": f"Bearer {API_KEY}"})

    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "invalid_api_key"
    assert first.status_code == 200
    assert first.json() == {
        "ready": True,
        "service": "forgenovax-kaggle-ai",
        "model": "qwen3-coder:30b",
        "inference": "verified",
    }
    assert cached.status_code == 200
    assert calls == 1


def test_readiness_rejects_nonmatching_model_output() -> None:
    app = create_app(
        Settings(api_key=API_KEY),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"message": {"content": "almost ready"}}
            )
        ),
    )
    with TestClient(app) as client:
        response = client.get(
            "/readyz", headers={"Authorization": f"Bearer {API_KEY}"}
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "readiness_mismatch"
