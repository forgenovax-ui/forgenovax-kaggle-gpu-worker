from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"


def _client() -> TestClient:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"object": "list", "data": []})
    )
    return TestClient(create_app(Settings(api_key=API_KEY), transport=transport))


def test_missing_authentication_is_rejected() -> None:
    with _client() as client:
        response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_authentication_is_rejected() -> None:
    with _client() as client:
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer incorrect-key"}
        )

    assert response.status_code == 401


def test_correct_authentication_is_accepted() -> None:
    with _client() as client:
        response = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
        )

    assert response.status_code == 200
