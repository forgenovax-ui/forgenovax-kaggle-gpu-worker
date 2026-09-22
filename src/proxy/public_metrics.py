"""Independent strict allowlist for the worker's public telemetry endpoint."""

from __future__ import annotations

import json
import math
import re
from typing import Any

EXPERIMENT_ID = re.compile(r"^FNX-R\d{3}$")
SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+:-]{0,95}$")
RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
SENSITIVE = re.compile(
    r"(?i)(?:gh[opurs]_[a-z0-9]{20,}|hf_[a-z0-9]{20,}|bearer\s+|api[_-]?key|"
    r"password|secret|token|traceback|/users/|/kaggle/|[a-z]:\\)"
)
STATES = {
    "PREPARING",
    "BASELINE",
    "COVERAGE_SWEEP",
    "TRAINING",
    "CALIBRATION",
    "FINAL_EVAL",
    "COMPLETE",
    "FAILED",
    "NOT_STARTED",
}
FINAL_RESULTS = {
    "CASCADE_WINNER",
    "FROZEN_DECIDER_WINNER",
    "ADAPTED_FNX_WINNER",
    "NO_TRAINING_JUSTIFIED",
    "NO_WINNER",
}
NUMERIC_FIELDS: dict[str, tuple[float | None, float | None]] = {
    "step": (0, None),
    "epoch": (0, None),
    "train_loss": (0, None),
    "validation_loss": (0, None),
    "learning_rate": (0, None),
    "accuracy": (0, 1),
    "nll": (0, None),
    "ece": (0, 1),
    "brier": (0, 2),
    "gpu_utilization": (0, 100),
    "vram_used_mib": (0, None),
    "throughput": (0, None),
    "elapsed_seconds": (0, None),
    "reflex_coverage": (0, 1),
    "system_accuracy": (0, 1),
    "baseline_accuracy": (0, 1),
    "false_allow_rate": (0, 1),
    "false_autonomy_rate": (0, 1),
    "strong_model_escalation_rate": (0, 1),
    "human_escalation_rate": (0, 1),
    "average_latency_ms": (0, None),
    "p50_latency_ms": (0, None),
    "p95_latency_ms": (0, None),
    "decisions_per_second": (0, None),
    "vram_mib": (0, None),
    "gpu_0_utilization": (0, 100),
    "gpu_1_utilization": (0, 100),
    "gpu_0_vram_mib": (0, None),
    "gpu_1_vram_mib": (0, None),
    "calibration_nll_before": (0, None),
    "calibration_nll_after": (0, None),
    "calibration_brier": (0, 2),
    "calibration_ece": (0, 1),
    "calibration_temperature": (0, None),
    "best_cascade_accuracy": (0, 1),
    "best_reflex_coverage": (0, 1),
    "strong_model_reduction": (0, 1),
}
CURVE_FIELDS = {
    "reflex_coverage": (0, 1),
    "system_accuracy": (0, 1),
    "average_latency_ms": (0, None),
    "strong_model_escalation_rate": (0, 1),
    "strong_model_calls": (0, None),
    "false_autonomy_rate": (0, 1),
}


def _number(value: Any, bounds: tuple[float | None, float | None]) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    low, high = bounds
    if (low is not None and value < low) or (high is not None and value > high):
        return None
    return value


def _curve(value: Any) -> list[dict[str, float | int]] | None:
    if not isinstance(value, list) or len(value) > 64:
        return None
    result = []
    for point in value:
        if not isinstance(point, dict):
            return None
        clean = {}
        for key, bounds in CURVE_FIELDS.items():
            if key in point:
                number = _number(point[key], bounds)
                if number is None:
                    return None
                clean[key] = number
        if "reflex_coverage" not in clean:
            return None
        result.append(clean)
    return result


def sanitize_public_metrics(payload: Any) -> dict[str, Any]:
    """Reconstruct the public object from known typed fields only."""
    if not isinstance(payload, dict):
        return {}
    clean: dict[str, Any] = {}
    experiment_id = payload.get("experiment_id")
    if isinstance(experiment_id, str) and EXPERIMENT_ID.fullmatch(experiment_id):
        clean["experiment_id"] = experiment_id
    model = payload.get("model")
    if isinstance(model, str) and SAFE_TEXT.fullmatch(model) and not SENSITIVE.search(model):
        clean["model"] = model
    state = payload.get("status")
    if isinstance(state, str) and state in STATES:
        clean["status"] = state
    for key, bounds in NUMERIC_FIELDS.items():
        if key in payload:
            number = _number(payload[key], bounds)
            if number is not None:
                clean[key] = number
    for key in ("updated_at", "projected_completion"):
        value = payload.get(key)
        if isinstance(value, str) and RFC3339.fullmatch(value):
            clean[key] = value
    for key in ("dry_run", "replay"):
        if isinstance(payload.get(key), bool):
            clean[key] = payload[key]
    final_result = payload.get("final_result")
    if isinstance(final_result, str) and final_result in FINAL_RESULTS:
        clean["final_result"] = final_result
    if payload.get("technical_status") in {"PASS", "FAIL"}:
        clean["technical_status"] = payload["technical_status"]
    if payload.get("production_status") == "NOT_PRODUCTION_CERTIFIED":
        clean["production_status"] = "NOT_PRODUCTION_CERTIFIED"
    if "coverage_curve" in payload:
        curve = _curve(payload["coverage_curve"])
        if curve is not None:
            clean["coverage_curve"] = curve
    serialized = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    return dict(sorted(clean.items())) if not SENSITIVE.search(serialized) else {}
