from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"


def test_models_endpoint_proxies_response_without_forwarding_credentials() -> None:
    observed: dict[str, object] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "qwen3-coder:30b", "object": "model"}],
            },
        )

    app = create_app(
        Settings(api_key=API_KEY), transport=httpx.MockTransport(upstream)
    )
    with TestClient(app) as client:
        response = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen3-coder:30b"
    assert observed == {
        "method": "GET",
        "url": "http://127.0.0.1:11434/v1/models",
        "authorization": None,
    }


def test_upstream_status_code_and_body_are_preserved() -> None:
    app = create_app(
        Settings(api_key=API_KEY),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={"error": {"message": "unknown model"}})
        ),
    )
    with TestClient(app) as client:
        response = client.get(
            "/v1/models/missing", headers={"Authorization": f"Bearer {API_KEY}"}
        )

    assert response.status_code == 404
    assert response.json() == {"error": {"message": "unknown model"}}
