from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def test_oversize_request_is_rejected_before_upstream() -> None:
    called = False

    def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    app = create_app(
        Settings(api_key=API_KEY, max_request_bytes=32),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", headers=HEADERS, content=b"x" * 33
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert called is False


def test_oversize_response_is_not_forwarded() -> None:
    app = create_app(
        Settings(api_key=API_KEY, max_response_bytes=8),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * 9)
        ),
    )
    with TestClient(app) as client:
        response = client.get("/v1/models", headers=HEADERS)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "response_too_large"
