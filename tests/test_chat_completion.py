import json

from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app


API_KEY = "test-api-key-with-at-least-24-characters"


def test_non_streaming_chat_completion_preserves_json_request() -> None:
    payload = {
        "model": "qwen3-coder:30b",
        "messages": [{"role": "user", "content": "Reply with exactly: OPENAI_API_READY"}],
        "stream": False,
    }
    observed: dict[str, object] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["path"] = request.url.path
        observed["payload"] = json.loads(request.content)
        observed["content_type"] = request.headers.get("content-type")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OPENAI_API_READY"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    app = create_app(
        Settings(api_key=API_KEY), transport=httpx.MockTransport(upstream)
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "OPENAI_API_READY"
    assert observed == {
        "method": "POST",
        "path": "/v1/chat/completions",
        "payload": payload,
        "content_type": "application/json",
    }


def test_ollama_connection_failure_is_sanitized() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private upstream detail", request=request)

    app = create_app(
        Settings(api_key=API_KEY), transport=httpx.MockTransport(unavailable)
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "qwen3-coder:30b", "messages": [], "stream": False},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference service is unavailable"}
    assert "private" not in response.text
