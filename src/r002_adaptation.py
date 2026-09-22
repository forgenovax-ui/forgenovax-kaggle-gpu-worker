"""Frozen decision rules for optional FNX-R002 Decider adaptation."""

from __future__ import annotations

from typing import Any

REQUIRED_CONFIG = {
    "architecture",
    "base_model",
    "base_revision",
    "epochs",
    "gradient_accumulation_steps",
    "learning_rate",
    "lora_alpha",
    "lora_dropout",
    "lora_rank",
    "max_context_tokens",
    "max_tokens_per_microbatch",
    "minimum_accuracy_gain",
    "minimum_compute_reduction_gain",
    "maximum_accuracy_regression_for_efficiency_win",
    "seed",
    "weight_decay",
}


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CONFIG - config.keys())
    if missing:
        raise ValueError("adaptation config is incomplete: " + ", ".join(missing))
    if config["base_model"] != "Mapika/decider-2b":
        raise ValueError("adaptation base model is not pinned Decider 2B")
    if config["base_revision"] != "b37f7e1ba3fbc9238004cf531fabbee2619973fd":
        raise ValueError("adaptation base revision is not pinned")
    for key in ("epochs", "gradient_accumulation_steps", "lora_alpha", "lora_rank"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "learning_rate",
        "minimum_accuracy_gain",
        "minimum_compute_reduction_gain",
        "maximum_accuracy_regression_for_efficiency_win",
    ):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be within [0, 1]")


def training_is_required(frozen_result: dict[str, Any]) -> bool:
    justification = frozen_result.get("training_justification")
    return (
        frozen_result.get("r002_status") == "TRAINING_JUSTIFIED"
        and isinstance(justification, dict)
        and justification.get("decision") is True
        and justification.get("useful_operating_region") is True
        and justification.get("uncertainty_separates_errors") is True
        and justification.get("calibration_usable") is True
    )


def final_status(
    frozen_best: dict[str, Any] | None,
    adapted_best: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Select the public outcome using predeclared system-level promotion gates."""
    validate_config(config)
    if adapted_best is None:
        if frozen_best is not None:
            return "FROZEN_DECIDER_WINNER", "adapted candidate found no qualifying operating point"
        return "NO_WINNER", "neither frozen nor adapted cascade qualified"
    if frozen_best is None:
        return "ADAPTED_FNX_WINNER", "adapted candidate created the only qualifying operating point"
    if adapted_best.get("false_allow_rate", 1.0) > frozen_best.get("false_allow_rate", 1.0):
        return "FROZEN_DECIDER_WINNER", "adaptation regressed the security false-allow gate"
    if adapted_best.get("false_autonomy_rate", 1.0) > frozen_best.get(
        "false_autonomy_rate", 1.0
    ):
        return "FROZEN_DECIDER_WINNER", "adaptation regressed the false-autonomy gate"
    accuracy_gain = adapted_best["system_accuracy"] - frozen_best["system_accuracy"]
    compute_gain = (
        adapted_best["normalized_compute_reduction_vs_0pct"]
        - frozen_best["normalized_compute_reduction_vs_0pct"]
    )
    accuracy_win = accuracy_gain >= config["minimum_accuracy_gain"]
    efficiency_win = (
        compute_gain >= config["minimum_compute_reduction_gain"]
        and accuracy_gain >= -config["maximum_accuracy_regression_for_efficiency_win"]
    )
    if accuracy_win or efficiency_win:
        reason = "material system accuracy gain" if accuracy_win else "material compute gain at retained quality"
        return "ADAPTED_FNX_WINNER", reason
    return "FROZEN_DECIDER_WINNER", "adaptation did not clear a material system-level gain"
