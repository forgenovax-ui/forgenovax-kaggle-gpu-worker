"""Runtime health checks for the complete Kaggle worker path."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import httpx

from src.config import Settings


WORK_DIR = Path(os.getenv("FORGENOVAX_WORK_DIR", "/kaggle/working"))


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 8.0,
) -> httpx.Response | None:
    try:
        return httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None


def _pid_is_running(pid_file: Path, expected_command: str | None = None) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    command_line = Path(f"/proc/{pid}/cmdline")
    if expected_command and command_line.exists():
        try:
            command = command_line.read_bytes().replace(b"\x00", b" ").decode(
                errors="replace"
            )
        except OSError:
            return False
        return expected_command in command
    return True


def _gpu_checks() -> list[Check]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            Check("GPU runtime", False, "nvidia-smi unavailable"),
            Check("GPU 0", False, "not detected"),
            Check("GPU 1", False, "not detected"),
        ]

    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    by_index = {row.split(",", 1)[0].strip(): row for row in rows}
    gpu_runtime_ok = len(rows) == 2 and all("T4" in row for row in rows)
    checks = [
        Check(
            "GPU runtime",
            gpu_runtime_ok,
            f"{len(rows)} NVIDIA GPU(s) visible; T4 x2 {'confirmed' if gpu_runtime_ok else 'not confirmed'}",
        )
    ]
    for index in ("0", "1"):
        row = by_index.get(index)
        checks.append(
            Check(f"GPU {index}", row is not None and "T4" in row, row or "not detected")
        )
    return checks


def _json(response: httpx.Response | None) -> dict[str, Any]:
    if response is None:
        return {}
    try:
        value = response.json()
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def run_checks(settings: Settings | None = None) -> list[Check]:
    config = settings or Settings.from_env()
    checks = _gpu_checks()

    ollama_process = _pid_is_running(WORK_DIR / "ollama.pid", "ollama")
    checks.append(
        Check(
            "Ollama process",
            ollama_process,
            "process running" if ollama_process else "process not running",
        )
    )

    tags_response = _http_get(f"{config.ollama_base_url}/api/tags")
    ollama_ok = tags_response is not None and tags_response.status_code == 200
    checks.append(
        Check("Ollama API", ollama_ok, "API reachable" if ollama_ok else "API unavailable")
    )

    model_names = {
        str(model.get("name", ""))
        for model in _json(tags_response).get("models", [])
        if isinstance(model, dict)
    }
    installed = config.model in model_names or any(
        name.removesuffix(":latest") == config.model.removesuffix(":latest")
        for name in model_names
    )
    checks.append(
        Check(
            "Model installed",
            installed,
            config.model if installed else f"{config.model} not present",
        )
    )

    ps_response = _http_get(f"{config.ollama_base_url}/api/ps")
    loaded_names = {
        str(model.get("name", ""))
        for model in _json(ps_response).get("models", [])
        if isinstance(model, dict)
    }
    loaded = config.model in loaded_names or any(
        name.removesuffix(":latest") == config.model.removesuffix(":latest")
        for name in loaded_names
    )
    checks.append(Check("Model loaded", loaded, config.model if loaded else "not loaded"))

    health_response = _http_get(f"{config.proxy_base_url}/healthz")
    health_body = _json(health_response)
    proxy_ok = (
        health_response is not None
        and health_response.status_code == 200
        and health_body.get("ok") is True
        and health_body.get("service") == "forgenovax-kaggle-ai"
    )
    checks.append(Check("Auth proxy", proxy_ok, "healthy" if proxy_ok else "unavailable"))

    unauthorized = _http_get(f"{config.proxy_base_url}/v1/models")
    unauthorized_ok = unauthorized is not None and unauthorized.status_code == 401
    checks.append(
        Check(
            "Unauthorized access blocked",
            unauthorized_ok,
            f"HTTP {unauthorized.status_code}" if unauthorized is not None else "no response",
        )
    )

    auth_headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    authorized = _http_get(f"{config.proxy_base_url}/v1/models", headers=auth_headers)
    authorized_ok = (
        proxy_ok
        and bool(config.api_key)
        and authorized is not None
        and authorized.status_code == 200
    )
    checks.append(
        Check(
            "Authorized models endpoint",
            authorized_ok,
            f"HTTP {authorized.status_code}" if authorized is not None else "no response",
        )
    )

    cloudflare_running = _pid_is_running(WORK_DIR / "cloudflared.pid", "cloudflared")
    checks.append(
        Check(
            "Cloudflare",
            cloudflare_running,
            "process running" if cloudflare_running else "process not running",
        )
    )

    tunnel_url = os.getenv("FORGENOVAX_TUNNEL_URL", "").strip()
    tunnel_file = WORK_DIR / "tunnel_url"
    if not tunnel_url and tunnel_file.exists():
        tunnel_url = tunnel_file.read_text(encoding="utf-8").strip()

    public_health = _http_get(f"{tunnel_url}/healthz", timeout=20.0) if tunnel_url else None
    public_health_body = _json(public_health)
    public_health_ok = (
        cloudflare_running
        and public_health is not None
        and public_health.status_code == 200
        and public_health_body.get("ok") is True
        and public_health_body.get("service") == "forgenovax-kaggle-ai"
    )
    checks.append(
        Check(
            "Public health",
            public_health_ok,
            f"HTTP {public_health.status_code}" if public_health is not None else "no tunnel URL/response",
        )
    )

    public_models = (
        _http_get(f"{tunnel_url}/v1/models", headers=auth_headers, timeout=30.0)
        if tunnel_url and config.api_key
        else None
    )
    public_api_ok = (
        public_health_ok and public_models is not None and public_models.status_code == 200
    )
    checks.append(
        Check(
            "Public API",
            public_api_ok,
            f"HTTP {public_models.status_code}" if public_models is not None else "no authenticated response",
        )
    )
    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["ForgeNovaX Kaggle GPU Worker", ""]
    for check in checks:
        state = "PASS" if check.passed else "FAIL"
        suffix = f" ({check.detail})" if check.detail else ""
        lines.append(f"{check.name}: {state}{suffix}")
    lines.extend(
        ["", f"OVERALL STATUS: {'READY' if all(item.passed for item in checks) else 'NOT READY'}"]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        checks = run_checks()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(format_report(checks))
    return 0 if all(item.passed for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
