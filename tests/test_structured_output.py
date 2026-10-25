import json

from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.proxy.app import create_app
from src.research_contracts import POST_CLASSIFICATION_SCHEMA


API_KEY = "test-api-key-with-at-least-24-characters"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def payload(*, stream: bool = False) -> dict[str, object]:
    return {
        "model": "qwen3:14b",
        "messages": [{"role": "user", "content": "synthetic discussion"}],
        "stream": stream,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "post_classification",
                "strict": True,
                "schema": POST_CLASSIFICATION_SCHEMA,
            },
        },
    }


def completion(content: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": json.dumps(content)}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        },
    )


def test_valid_schema_constrained_output_is_forwarded() -> None:
    value = {
        "relevance_score": 91,
        "category": "ai_engineering",
        "reasoning_summary": "The discussion provides evaluation evidence.",
        "additional_research_justified": True,
    }
    app = create_app(
        Settings(api_key=API_KEY),
        transport=httpx.MockTransport(lambda request: completion(value)),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=payload())

    assert response.status_code == 200
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == value


def test_schema_invalid_model_output_is_rejected() -> None:
    app = create_app(
        Settings(api_key=API_KEY),
        transport=httpx.MockTransport(
            lambda request: completion({"category": "ai_engineering"})
        ),
    )
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=HEADERS, json=payload())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "schema_validation_failed"


def test_streaming_structured_output_is_rejected_before_upstream() -> None:
    called = False

    def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return completion({})

    app = create_app(
        Settings(api_key=API_KEY), transport=httpx.MockTransport(upstream)
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", headers=HEADERS, json=payload(stream=True)
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "structured_stream_unsupported"
    assert called is False
