import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.r002_adaptation import final_status, training_is_required, validate_config

ROOT = Path(__file__).resolve().parents[1]


def config() -> dict[str, object]:
    return json.loads((ROOT / "configs/r002_adaptation.json").read_text())


def point(*, accuracy: float, compute: float, false_allow: float = 0.0) -> dict[str, float]:
    return {
        "system_accuracy": accuracy,
        "normalized_compute_reduction_vs_0pct": compute,
        "false_allow_rate": false_allow,
        "false_autonomy_rate": 0.0,
    }


def test_adaptation_config_is_complete_and_pinned() -> None:
    validate_config(config())


def test_training_requires_every_frozen_gate() -> None:
    result = {
        "r002_status": "TRAINING_JUSTIFIED",
        "training_justification": {
            "decision": True,
            "useful_operating_region": True,
            "uncertainty_separates_errors": True,
            "calibration_usable": True,
        },
    }
    assert training_is_required(result)
    for key in result["training_justification"]:
        changed = json.loads(json.dumps(result))
        changed["training_justification"][key] = False
        assert not training_is_required(changed)


def test_adapted_candidate_must_make_material_system_gain() -> None:
    frozen = point(accuracy=0.90, compute=0.30)
    assert final_status(frozen, point(accuracy=0.91, compute=0.30), config())[0] == "ADAPTED_FNX_WINNER"
    assert final_status(frozen, point(accuracy=0.896, compute=0.351), config())[0] == "ADAPTED_FNX_WINNER"
    assert final_status(frozen, point(accuracy=0.905, compute=0.34), config())[0] == "FROZEN_DECIDER_WINNER"


def test_security_regression_always_blocks_adapted_promotion() -> None:
    frozen = point(accuracy=0.90, compute=0.30)
    adapted = point(accuracy=0.95, compute=0.50, false_allow=0.001)
    assert final_status(frozen, adapted, config())[0] == "FROZEN_DECIDER_WINNER"


def test_no_qualifying_candidate_has_truthful_outcome() -> None:
    assert final_status(point(accuracy=0.9, compute=0.3), None, config())[0] == "FROZEN_DECIDER_WINNER"
    assert final_status(None, None, config())[0] == "NO_WINNER"
    assert final_status(None, point(accuracy=0.9, compute=0.3), config())[0] == "ADAPTED_FNX_WINNER"


def test_invalid_config_fails_closed() -> None:
    invalid = config()
    invalid["base_revision"] = "floating"
    with pytest.raises(ValueError, match="not pinned"):
        validate_config(invalid)


def make_dry_run_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    artifact = tmp_path / "artifacts"
    decider = tmp_path / "decider"
    (source / "scripts").mkdir(parents=True)
    (source / "configs" / "eval").mkdir(parents=True)
    (source / "reports").mkdir(parents=True)
    artifact.mkdir()
    (decider / "decider").mkdir(parents=True)
    (source / "scripts" / "run_r002_experiment.py").write_text("# frozen\n")
    (source / "configs" / "eval" / "r002.json").write_text("{}\n")
    (source / "reports" / "R002_RESULTS.json").write_text(
        json.dumps(
            {
                "experiment_id": "FNX-R002",
                "r002_status": "NO_TRAINING_JUSTIFIED",
                "training_justification": {"decision": False},
            }
        )
    )
    for name in ("calibration.json", "strong-heldout.json", "reflex-heldout.json"):
        (artifact / name).write_text("{}\n")
    (decider / "decider" / "model.py").write_text("# pinned source\n")
    return source, artifact, decider


def test_dry_run_validates_without_emitting_public_metrics(tmp_path: Path) -> None:
    source, artifact, decider = make_dry_run_tree(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_r002_adaptation.py"),
            "--source-dir",
            str(source),
            "--artifact-dir",
            str(artifact),
            "--decider-dir",
            str(decider),
            "--config",
            str(ROOT / "configs" / "r002_adaptation.json"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DRY RUN: READY" in completed.stdout
    assert not (artifact / "live" / "metrics.json").exists()


def test_dry_run_fails_closed_on_missing_prerequisite(tmp_path: Path) -> None:
    source, artifact, decider = make_dry_run_tree(tmp_path)
    (artifact / "reflex-heldout.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_r002_adaptation.py"),
            "--source-dir",
            str(source),
            "--artifact-dir",
            str(artifact),
            "--decider-dir",
            str(decider),
            "--config",
            str(ROOT / "configs" / "r002_adaptation.json"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "missing R002 adaptation prerequisites" in completed.stderr
