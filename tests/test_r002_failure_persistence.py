from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_post_boundary_failure_is_truthfully_persisted(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    source = tmp_path / "source"
    artifact.mkdir()
    (source / "reports").mkdir(parents=True)
    (source / "reports" / "R002_RESULTS.json").write_text(
        json.dumps({"technical_status": "NOT_RUN", "r002_status": "NOT_RUN"}),
        encoding="utf-8",
    )
    (artifact / "live-boundary.json").write_text(
        json.dumps({"held_out_accessed": True}), encoding="utf-8"
    )
    archive_marker = tmp_path / "archive-attempted"
    shell = ROOT / "scripts" / "run_r002_frozen.sh"
    command = f"""
source {shell!s}
persist_artifacts() {{ touch {archive_marker!s}; return 0; }}
record_post_boundary_failure 17 REFLEX_FINAL
"""
    environment = {
        **os.environ,
        "FORGENOVAX_WORK_DIR": str(tmp_path),
        "FNX_R002_SOURCE_SHA256": "archive",
        "FNX_R002_SOURCE_TREE_SHA256": "tree",
        "FNX_R002_SOURCE_GIT_SHA": "git",
        "FNX_R002_ARTIFACT_DIR": str(artifact),
        "FNX_R002_SOURCE_DIR": str(source),
        "FNX_PUBLIC_METRICS_PATH": str(artifact / "live" / "metrics.json"),
    }
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    failure = json.loads((artifact / "failure-evidence.json").read_text())
    assert failure == {
        "experiment_id": "FNX-R002",
        "exit_status": 17,
        "failed_stage": "REFLEX_FINAL",
        "held_out_accessed": True,
        "production_status": "NOT_PRODUCTION_CERTIFIED",
        "r002_status": "NO_WINNER",
        "technical_status": "FAIL",
        "recorded_at": failure["recorded_at"],
    }
    metrics = json.loads((artifact / "live" / "metrics.json").read_text())
    assert metrics["status"] == "FAILED"
    assert metrics["technical_status"] == "FAIL"
    assert metrics["final_result"] == "NO_WINNER"
    assert metrics["production_status"] == "NOT_PRODUCTION_CERTIFIED"
    assert archive_marker.exists()
    final_report = json.loads((source / "reports" / "R002_RESULTS.json").read_text())
    assert final_report["technical_status"] == "FAIL"
    assert final_report["r002_status"] == "NO_WINNER"
    assert final_report["failed_stage"] == "REFLEX_FINAL"
    assert "Technical status: **FAIL**" in (
        source / "docs" / "R002_RESULTS.md"
    ).read_text()


def test_post_boundary_error_trap_preserves_original_exit_status(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    source = tmp_path / "source"
    artifact.mkdir()
    source.mkdir()
    (artifact / "live-boundary.json").write_text(
        json.dumps({"held_out_accessed": True}), encoding="utf-8"
    )
    shell = ROOT / "scripts" / "run_r002_frozen.sh"
    command = f"""
source {shell!s}
persist_artifacts() {{ return 0; }}
post_boundary_stage=STRONG_FINAL
trap post_boundary_error ERR
fail() {{ return 23; }}
fail
"""
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FORGENOVAX_WORK_DIR": str(tmp_path),
            "FNX_R002_SOURCE_SHA256": "archive",
            "FNX_R002_SOURCE_TREE_SHA256": "tree",
            "FNX_R002_SOURCE_GIT_SHA": "git",
            "FNX_R002_ARTIFACT_DIR": str(artifact),
            "FNX_R002_SOURCE_DIR": str(source),
            "FNX_PUBLIC_METRICS_PATH": str(artifact / "live" / "metrics.json"),
        },
    )

    assert completed.returncode == 23
    failure = json.loads((artifact / "failure-evidence.json").read_text())
    assert failure["exit_status"] == 23
    assert failure["failed_stage"] == "STRONG_FINAL"
