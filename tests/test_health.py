from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"


def test_health_endpoint_is_public_and_minimal() -> None:
    app = create_app(
        Settings(api_key=API_KEY),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "service": "forgenovax-kaggle-ai"}
    assert "api" not in response.text.lower().replace("kaggle-ai", "")
