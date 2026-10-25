"""Authenticated loopback gateway for Ollama's OpenAI-compatible API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.config import Settings
from src.research_contracts import (
    StructuredOutputError,
    checked_schema,
    validate_instance,
)


SERVICE_NAME = "forgenovax-kaggle-ai"
READINESS_SENTINEL = "FORGENOVAX_READY"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
REQUEST_HEADERS_TO_REMOVE = HOP_BY_HOP_HEADERS | {
    "authorization",
    "host",
    "content-length",
}
# httpx transparently decodes compressed responses, so the original encoding
# header must not be forwarded with the decoded body.
RESPONSE_HEADERS_TO_REMOVE = HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}


class GatewayError(Exception):
    """An intentionally sanitized OpenAI-compatible gateway error."""

    def __init__(self, status_code: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


def _filtered_headers(headers: httpx.Headers, blocked: set[str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in blocked}


def _error_body(error: GatewayError) -> dict[str, object]:
    return {
        "error": {
            "message": error.message,
            "type": "gateway_error",
            "param": None,
            "code": error.code,
        }
    }


def _structured_schema(api_path: str, body: bytes) -> dict[str, Any] | None:
    if api_path != "chat/completions" or not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GatewayError(400, "Request body must be valid JSON", "invalid_json") from exc
    if not isinstance(payload, dict):
        raise GatewayError(400, "Request body must be a JSON object", "invalid_request")
    try:
        schema = checked_schema(payload.get("response_format"))
    except StructuredOutputError as exc:
        raise GatewayError(400, str(exc), "invalid_response_format") from exc
    if schema is not None and payload.get("stream") is True:
        raise GatewayError(
            400,
            "Structured responses must use stream=false so the gateway can validate them",
            "structured_stream_unsupported",
        )
    return schema


def _validate_structured_completion(content: bytes, schema: dict[str, Any]) -> None:
    try:
        completion = json.loads(content)
        choices = completion["choices"]
        model_content = choices[0]["message"]["content"]
        value = json.loads(model_content)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise GatewayError(
            502,
            "Inference service returned malformed structured output",
            "invalid_model_output",
        ) from exc
    try:
        validate_instance(value, schema)
    except StructuredOutputError as exc:
        raise GatewayError(502, str(exc), "schema_validation_failed") from exc


async def _bounded_content(response: httpx.Response, limit: int) -> bytes:
    content = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > limit:
                raise GatewayError(
                    502,
                    "Inference response exceeded the configured size limit",
                    "response_too_large",
                )
    finally:
        await response.aclose()
    return bytes(content)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_settings.validate_for_startup()
        application.state.ollama_client = httpx.AsyncClient(
            base_url=runtime_settings.ollama_base_url,
            timeout=httpx.Timeout(runtime_settings.upstream_timeout_seconds),
            transport=transport,
        )
        application.state.readiness = None
        try:
            yield
        finally:
            await application.state.ollama_client.aclose()

    application = FastAPI(
        title="ForgeNovaX Kaggle GPU Worker",
        version="1.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc),
            headers=headers,
        )

    async def require_bearer(request: Request) -> None:
        expected = runtime_settings.api_key
        if not expected:
            raise GatewayError(503, "API authentication is not configured", "auth_unconfigured")

        authorization = request.headers.get("authorization", "")
        scheme, separator, provided = authorization.partition(" ")
        valid = (
            bool(separator)
            and scheme.lower() == "bearer"
            and bool(provided)
            and hmac.compare_digest(provided, expected)
        )
        if not valid:
            raise GatewayError(401, "Invalid or missing bearer token", "invalid_api_key")

    @application.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"ok": True, "service": SERVICE_NAME}

    @application.get("/readyz", dependencies=[Depends(require_bearer)])
    async def readyz(request: Request) -> dict[str, object]:
        now = time.monotonic()
        cached = request.app.state.readiness
        if cached and now - cached[0] <= runtime_settings.readiness_cache_seconds:
            return cached[1]
        client: httpx.AsyncClient = request.app.state.ollama_client
        try:
            response = await client.post(
                "/api/chat",
                json={
                    "model": runtime_settings.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Reply with exactly: {READINESS_SENTINEL}",
                        }
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": runtime_settings.ollama_keep_alive,
                    "options": {
                        "temperature": 0,
                        "num_ctx": min(runtime_settings.context_length, 2048),
                        # gpt-oss may emit a short thinking trace even when
                        # think=false. Keep enough bounded budget for the
                        # sentinel to reach the content field.
                        "num_predict": 128,
                    },
                },
                timeout=runtime_settings.readiness_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["message"]["content"].strip()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise GatewayError(503, "Inference service is unavailable", "upstream_unavailable") from None
        except httpx.TimeoutException:
            raise GatewayError(504, "Readiness inference timed out", "readiness_timeout") from None
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
            raise GatewayError(503, "Model inference is not ready", "model_not_ready") from None
        if content != READINESS_SENTINEL:
            raise GatewayError(503, "Model inference is not ready", "readiness_mismatch")
        result = {
            "ready": True,
            "service": SERVICE_NAME,
            "model": runtime_settings.model,
            "inference": "verified",
        }
        request.app.state.readiness = (now, result)
        return result

    @application.api_route(
        "/v1/{api_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        dependencies=[Depends(require_bearer)],
    )
    async def proxy_openai_request(api_path: str, request: Request) -> Response:
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                content_length = int(raw_length)
            except ValueError:
                raise GatewayError(400, "Invalid Content-Length", "invalid_content_length") from None
            if content_length > runtime_settings.max_request_bytes:
                raise GatewayError(413, "Request body is too large", "request_too_large")
        body = await request.body()
        if len(body) > runtime_settings.max_request_bytes:
            raise GatewayError(413, "Request body is too large", "request_too_large")
        schema = _structured_schema(api_path, body)

        target = f"/v1/{api_path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        client: httpx.AsyncClient = request.app.state.ollama_client
        upstream_request = client.build_request(
            request.method,
            target,
            content=body,
            headers=_filtered_headers(request.headers, REQUEST_HEADERS_TO_REMOVE),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
            content = await _bounded_content(upstream, runtime_settings.max_response_bytes)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise GatewayError(503, "Inference service is unavailable", "upstream_unavailable") from None
        except httpx.TimeoutException:
            raise GatewayError(504, "Inference request timed out", "upstream_timeout") from None
        except httpx.HTTPError:
            raise GatewayError(502, "Inference service returned an invalid response", "bad_gateway") from None

        if schema is not None and 200 <= upstream.status_code < 300:
            _validate_structured_completion(content, schema)

        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_filtered_headers(upstream.headers, RESPONSE_HEADERS_TO_REMOVE),
            media_type=None,
        )

    return application


app = create_app()
