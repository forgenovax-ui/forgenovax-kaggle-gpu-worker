from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_live_preflight_starts_allowlisted_telemetry_without_heldout(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    frozen = executable(
        tmp_path / "frozen.sh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
test "$FNX_R002_PREFLIGHT_ONLY" = true
test -n "$FNX_PUBLIC_METRICS_PATH"
mkdir -p "$(dirname "$FNX_PUBLIC_METRICS_PATH")"
printf '{"experiment_id":"FNX-R002","status":"PREPARING","dry_run":false,"production_status":"NOT_PRODUCTION_CERTIFIED"}\\n' > "$FNX_PUBLIC_METRICS_PATH"
printf 'frozen\\n' >> "$FNX_TEST_MARKER"
""",
    )
    shutdown = executable(
        tmp_path / "shutdown.sh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'shutdown\\n' >> "$FNX_TEST_MARKER"
""",
    )
    mode = executable(
        tmp_path / "mode.sh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
test "$MODE" = inference
printf 'https://r002-preflight.trycloudflare.com\\n' > "$FORGENOVAX_WORK_DIR/tunnel_url"
printf 'mode\\n' >> "$FNX_TEST_MARKER"
""",
    )
    fake_curl = executable(
        tmp_path / "curl.sh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
output=''
while (($#)); do
  if [[ "$1" = --output ]]; then output="$2"; shift 2; else shift; fi
done
cp "$FNX_PUBLIC_METRICS_PATH" "$output"
""",
    )
    environment = {
        **os.environ,
        "FORGENOVAX_WORK_DIR": str(tmp_path),
        "FNX_R002_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        "FNX_R002_FROZEN_RUNNER": str(frozen),
        "FNX_R002_SHUTDOWN_RUNNER": str(shutdown),
        "FNX_R002_MODE_RUNNER": str(mode),
        "FNX_R002_CURL_BIN": str(fake_curl),
        "FNX_TEST_MARKER": str(marker),
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_r002_live_preflight.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert marker.read_text(encoding="utf-8").splitlines() == [
        "frozen",
        "shutdown",
        "mode",
    ]
    assert "FNX-R002 PUBLIC TELEMETRY: READY" in completed.stdout
    assert "HELD-OUT CLOSED" in completed.stdout
    assert not (tmp_path / "artifacts" / "live-boundary.json").exists()
    assert not (tmp_path / "artifacts" / "strong-heldout.json").exists()


def test_live_preflight_respects_emergency_kill_switch(tmp_path: Path) -> None:
    kill_switch = tmp_path / "artifacts" / "live" / "PUBLIC_DISABLED"
    kill_switch.parent.mkdir(parents=True)
    kill_switch.touch()
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_r002_live_preflight.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FORGENOVAX_WORK_DIR": str(tmp_path),
            "FNX_R002_ARTIFACT_DIR": str(tmp_path / "artifacts"),
        },
    )

    assert completed.returncode != 0
    assert "emergency kill switch is active" in completed.stderr
