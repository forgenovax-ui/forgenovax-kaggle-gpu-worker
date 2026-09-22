from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_private_archive_has_persistent_verification_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts"
    source = tmp_path / "source"
    artifact.mkdir()
    (source / "reports").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / "datasets" / "manifests").mkdir(parents=True)
    (artifact / "source-provenance.json").write_text(
        json.dumps(
            {
                "source_git_sha": "source-git",
                "source_tree_sha256": "source-tree",
                "source_archive_sha256": "source-archive",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "live-boundary.json").write_text(
        json.dumps({"held_out_accessed": True}), encoding="utf-8"
    )
    (source / "reports" / "R002_RESULTS.json").write_text(
        json.dumps(
            {
                "technical_status": "PASS",
                "r002_status": "NO_TRAINING_JUSTIFIED",
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "R002_PLAN.md",
        "R002_PREEXPERIMENT_READINESS.md",
        "R002_COVERAGE_CURVE.md",
        "R002_RESULTS.md",
    ):
        (source / "docs" / name).write_text("evidence\n", encoding="utf-8")
    for name in ("fnx-core-002.json", "fnx-core-002-contamination.json"):
        (source / "datasets" / "manifests" / name).write_text("{}\n")
    shell = ROOT / "scripts" / "run_r002_frozen.sh"
    command = f"""
source {shell!s}
persist_artifacts
"""
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FORGENOVAX_WORK_DIR": str(tmp_path),
            "FNX_R002_SOURCE_SHA256": "archive",
            "FNX_R002_SOURCE_TREE_SHA256": "tree",
            "FNX_R002_SOURCE_GIT_SHA": "git",
            "FNX_R002_SOURCE_DIR": str(source),
            "FNX_R002_ARTIFACT_DIR": str(artifact),
            "FNX_PUBLIC_METRICS_PATH": str(artifact / "live" / "metrics.json"),
        },
    )

    archive = tmp_path / "FNX-R002-private-artifacts.tar.gz"
    manifest_path = tmp_path / "FNX-R002-private-artifacts.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["archive_bytes"] == archive.stat().st_size
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert manifest["worker_git_sha"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert manifest["source_git_sha"] == "source-git"
    assert manifest["source_tree_sha256"] == "source-tree"
    assert manifest["held_out_accessed"] is True
    assert manifest["technical_status"] == "PASS"
    assert manifest["r002_status"] == "NO_TRAINING_JUSTIFIED"
    assert manifest["production_status"] == "NOT_PRODUCTION_CERTIFIED"
    assert oct(manifest_path.stat().st_mode & 0o777) == "0o600"
    (artifact / "failure-evidence.json").write_text(
        json.dumps(
            {
                "technical_status": "FAIL",
                "r002_status": "NO_WINNER",
                "held_out_accessed": True,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FORGENOVAX_WORK_DIR": str(tmp_path),
            "FNX_R002_SOURCE_SHA256": "archive",
            "FNX_R002_SOURCE_TREE_SHA256": "tree",
            "FNX_R002_SOURCE_GIT_SHA": "git",
            "FNX_R002_SOURCE_DIR": str(source),
            "FNX_R002_ARTIFACT_DIR": str(artifact),
            "FNX_PUBLIC_METRICS_PATH": str(artifact / "live" / "metrics.json"),
        },
    )
    failed_manifest = json.loads(manifest_path.read_text())
    assert failed_manifest["technical_status"] == "FAIL"
    assert failed_manifest["r002_status"] == "NO_WINNER"
