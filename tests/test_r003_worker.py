from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r003_worker_enforces_exact_sha_and_private_persistence() -> None:
    script = (ROOT / "scripts" / "run_r003_reflex.sh").read_text(encoding="utf-8")
    assert "FNX_R003_SOURCE_GIT_SHA" in script
    assert "^[0-9a-f]{40}$" in script
    assert "require_idle_t4_x2" in script
    assert "shutdown.sh" in script
    assert "FNX-R003-private-artifacts.tar.gz" in script
    assert "Kaggle notebook output" in script


def test_r003_worker_does_not_start_public_stream() -> None:
    script = (ROOT / "scripts" / "run_r003_reflex.sh").read_text(encoding="utf-8")
    assert "start_tunnel.sh" not in script
    assert "youtube" not in script.lower()
