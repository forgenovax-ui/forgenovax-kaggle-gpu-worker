"""Central runtime configuration for the ForgeNovaX worker."""

from __future__ import annotations

from dataclasses import dataclass
import os


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_MODEL = "qwen3-coder:30b"
SUPPORTED_MODELS = ("qwen3:14b", "gpt-oss:20b", "qwen3-coder:30b")


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration sourced from environment variables.

    Network hosts are intentionally fixed to loopback. Public traffic may only
    enter through Cloudflare and the authenticated proxy.
    """

    model: str = DEFAULT_MODEL
    context_length: int = 32768
    ollama_num_parallel: int = 1
    ollama_keep_alive: str = "20m"
    ollama_host: str = "127.0.0.1:11434"
    proxy_host: str = LOOPBACK_HOST
    proxy_port: int = 8000
    api_key: str = ""
    upstream_timeout_seconds: float = 900.0
    readiness_timeout_seconds: float = 180.0
    readiness_cache_seconds: float = 30.0
    max_request_bytes: int = 2 * 1024 * 1024
    max_response_bytes: int = 16 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434").strip()
        proxy_host = os.getenv("PROXY_HOST", LOOPBACK_HOST).strip()
        if ollama_host != "127.0.0.1:11434":
            raise ValueError("OLLAMA_HOST must be 127.0.0.1:11434")
        if proxy_host != LOOPBACK_HOST:
            raise ValueError("PROXY_HOST must be 127.0.0.1")

        return cls(
            model=os.getenv("MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            context_length=_positive_int("CONTEXT_LENGTH", 32768),
            ollama_num_parallel=_positive_int("OLLAMA_NUM_PARALLEL", 1),
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "20m").strip() or "20m",
            ollama_host=ollama_host,
            proxy_host=proxy_host,
            proxy_port=_positive_int("PROXY_PORT", 8000),
            api_key=os.getenv("FORGENOVAX_API_KEY", ""),
            upstream_timeout_seconds=_positive_float("UPSTREAM_TIMEOUT_SECONDS", 900.0),
            readiness_timeout_seconds=_positive_float(
                "READINESS_TIMEOUT_SECONDS", 180.0
            ),
            readiness_cache_seconds=_positive_float("READINESS_CACHE_SECONDS", 30.0),
            max_request_bytes=_positive_int("MAX_REQUEST_BYTES", 2 * 1024 * 1024),
            max_response_bytes=_positive_int("MAX_RESPONSE_BYTES", 16 * 1024 * 1024),
        )

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}"

    @property
    def proxy_base_url(self) -> str:
        return f"http://{self.proxy_host}:{self.proxy_port}"

    def validate_for_startup(self) -> None:
        if len(self.api_key) < 24:
            raise ValueError(
                "FORGENOVAX_API_KEY is missing or too short; use at least 24 characters"
            )
        if self.model not in SUPPORTED_MODELS:
            choices = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"MODEL must be one of: {choices}")
        if not 512 <= self.context_length <= 131072:
            raise ValueError("CONTEXT_LENGTH must be between 512 and 131072")
        if self.ollama_num_parallel > 4:
            raise ValueError("OLLAMA_NUM_PARALLEL must be between 1 and 4")
