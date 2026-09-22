"""Central runtime configuration for the ForgeNovaX worker."""

from __future__ import annotations

from dataclasses import dataclass
import os


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_MODEL = "qwen3-coder:30b"


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
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
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "900")),
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
