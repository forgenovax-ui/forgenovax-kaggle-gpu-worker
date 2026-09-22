"""Authenticated loopback gateway for Ollama's OpenAI-compatible API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from src.config import Settings
from src.proxy.public_metrics import sanitize_public_metrics


SERVICE_NAME = "forgenovax-kaggle-ai"
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


def _filtered_headers(headers: httpx.Headers, blocked: set[str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() not in blocked}


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
        try:
            yield
        finally:
            await application.state.ollama_client.aclose()

    application = FastAPI(
        title="ForgeNovaX Kaggle GPU Worker",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def require_bearer(request: Request) -> None:
        expected = runtime_settings.api_key
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured",
            )

        authorization = request.headers.get("authorization", "")
        scheme, separator, provided = authorization.partition(" ")
        valid = (
            bool(separator)
            and scheme.lower() == "bearer"
            and bool(provided)
            and hmac.compare_digest(provided, expected)
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @application.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"ok": True, "service": SERVICE_NAME}

    @application.get("/fnx/live/metrics")
    async def public_live_metrics() -> JSONResponse:
        metrics_path = Path(runtime_settings.public_metrics_path)
        kill_switch_path = Path(runtime_settings.public_metrics_kill_switch_path)
        headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        }
        if not runtime_settings.public_metrics_path:
            return JSONResponse(
                {"status": "TELEMETRY_UNAVAILABLE"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers=headers,
            )
        if runtime_settings.public_metrics_kill_switch_path and kill_switch_path.exists():
            return JSONResponse(
                {"status": "PUBLIC_TELEMETRY_DISABLED"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers=headers,
            )
        try:
            sanitized = sanitize_public_metrics(
                json.loads(metrics_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            sanitized = {}
        if not sanitized or sanitized.get("experiment_id") != "FNX-R002":
            return JSONResponse(
                {"status": "TELEMETRY_UNAVAILABLE"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers=headers,
            )
        return JSONResponse(sanitized, headers=headers)

    @application.api_route(
        "/v1/{api_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        dependencies=[Depends(require_bearer)],
    )
    async def proxy_openai_request(api_path: str, request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.ollama_client
        body = await request.body()
        target = f"/v1/{api_path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"

        try:
            upstream = await client.request(
                request.method,
                target,
                content=body,
                headers=_filtered_headers(request.headers, REQUEST_HEADERS_TO_REMOVE),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference service is unavailable",
            ) from None
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference request timed out",
            ) from None
        except httpx.HTTPError:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Inference service returned an invalid response",
            ) from None

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_filtered_headers(upstream.headers, RESPONSE_HEADERS_TO_REMOVE),
            media_type=None,
        )

    return application


app = create_app()
