#!/usr/bin/env python3
"""Run the predeclared optional FNX-R002 Decider LoRA adaptation stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_ROOT))

from src.r002_adaptation import final_status, training_is_required, validate_config

REQUIRED_SOURCE_FILES = (
    "scripts/run_r002_experiment.py",
    "configs/eval/r002.json",
    "reports/R002_RESULTS.json",
)
REQUIRED_ARTIFACT_FILES = (
    "calibration.json",
    "strong-heldout.json",
    "reflex-heldout.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--decider-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKER_ROOT / "configs" / "r002_adaptation.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_prerequisites(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    missing = [
        str(args.source_dir / relative)
        for relative in REQUIRED_SOURCE_FILES
        if not (args.source_dir / relative).is_file()
    ]
    missing.extend(
        str(args.artifact_dir / relative)
        for relative in REQUIRED_ARTIFACT_FILES
        if not (args.artifact_dir / relative).is_file()
    )
    if not (args.decider_dir / "decider" / "model.py").is_file():
        missing.append(str(args.decider_dir / "decider" / "model.py"))
    if missing:
        raise FileNotFoundError("missing R002 adaptation prerequisites: " + ", ".join(missing))
    frozen = load_json(args.source_dir / "reports" / "R002_RESULTS.json")
    if frozen.get("experiment_id") != "FNX-R002":
        raise ValueError("frozen result is not FNX-R002")
    return frozen


def import_runner(source_dir: Path) -> Any:
    sys.path.insert(0, str(source_dir / "src"))
    path = source_dir / "scripts" / "run_r002_experiment.py"
    spec = importlib.util.spec_from_file_location("fnx_r002_frozen_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the frozen R002 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def examples_for(rows: list[Any], runner: Any, Example: Any, Q: Any) -> list[Any]:
    examples = []
    for row in rows:
        question = runner.question_for(row)
        labels = runner.labels_for(row)
        target = runner.target_for(row)
        examples.append(
            Example(
                json.dumps(row.context, sort_keys=True, ensure_ascii=False),
                [Q(question["question"], question["options"], labels.index(target))],
                "fnx-r002",
            )
        )
    return examples


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def gpu_snapshot() -> dict[str, float]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rows = [
            [float(value.strip()) for value in line.split(",")]
            for line in output.splitlines()
            if line.strip()
        ]
        return {
            "gpu_utilization": sum(row[0] for row in rows) / len(rows),
            "vram_mib": max(row[1] for row in rows),
        }
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        return {}


def validation_metrics(model: Any, examples: list[Any], max_context: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    from decider.model import collate
    from decider.prompt import build

    class NoShuffle:
        def shuffle(self, values: list[Any]) -> None:
            return None

        def sample(self, values: list[Any], count: int) -> list[Any]:
            return values[:count]

    items = [
        build(example, model.tok, NoShuffle(), max_options=10, max_ctx_tokens=max_context)
        for example in examples
    ]
    items.sort(key=lambda item: len(item["ids"]))
    total_loss = 0.0
    total_correct = 0
    total_questions = 0
    model.eval()
    with torch.no_grad():
        for item in items:
            batch = move_batch(collate([item], model.tok.pad_token_id), "cuda")
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["golds"])
            total_loss += float(loss) * len(batch["golds"])
            total_correct += int((logits.argmax(-1) == batch["golds"]).sum())
            total_questions += len(batch["golds"])
    model.train()
    return {
        "loss": total_loss / total_questions,
        "accuracy": total_correct / total_questions,
        "questions": float(total_questions),
    }


def train_candidate(
    config: dict[str, Any],
    runner: Any,
    decider_dir: Path,
    artifact_dir: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    sys.path.insert(0, str(decider_dir))
    from decider.infer import Example, Q
    from decider.model import DecisionModel, collate
    from decider.train import batches_by_tokens, make_items

    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R002 adaptation")
    snapshot = snapshot_download(config["base_model"], revision=config["base_revision"])
    model = DecisionModel(snapshot, dtype=torch.float16, grad_ckpt=True)
    model.lm.config.use_cache = False
    model.lm = get_peft_model(
        model.lm,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        ),
    )
    model = model.cuda()
    train_rows, train_hash = runner.read_split("train")
    validation_rows, validation_hash = runner.read_split("validation")
    train_examples = examples_for(train_rows, runner, Example, Q)
    validation_examples = examples_for(validation_rows, runner, Example, Q)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    started = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_rng = random.Random(config["seed"] + epoch)
        items = make_items(
            train_examples,
            model.tok,
            epoch_rng,
            config["max_context_tokens"],
            max_options=10,
        )
        batch_indexes = batches_by_tokens(
            items,
            config["max_tokens_per_microbatch"],
            epoch_rng,
        )
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        questions = 0
        for micro_step, indexes in enumerate(batch_indexes, 1):
            batch = move_batch(
                collate([items[index] for index in indexes], model.tok.pad_token_id),
                "cuda",
            )
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["golds"])
                scaled_loss = loss / config["gradient_accumulation_steps"]
            scaler.scale(scaled_loss).backward()
            epoch_loss += float(loss.detach()) * len(batch["golds"])
            questions += len(batch["golds"])
            should_step = (
                micro_step % config["gradient_accumulation_steps"] == 0
                or micro_step == len(batch_indexes)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation = validation_metrics(
            model,
            validation_examples,
            config["max_context_tokens"],
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / questions,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(record)
        runner.emit_live(
            artifact_dir,
            "TRAINING",
            training_epoch=epoch,
            training_epochs=config["epochs"],
            training_loss=record["train_loss"],
            validation_accuracy=record["validation_accuracy"],
            **gpu_snapshot(),
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model.lm).items()
            }
    if best_state is None:
        raise RuntimeError("adaptation produced no candidate checkpoint")
    set_peft_model_state_dict(model.lm, best_state)
    adapter_dir = artifact_dir / "private-candidate" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(adapter_dir, safe_serialization=True)
    model.tok.save_pretrained(adapter_dir)
    provenance = {
        "architecture": config["architecture"],
        "base_model": config["base_model"],
        "base_revision": config["base_revision"],
        "train_split_hash": train_hash,
        "validation_split_hash": validation_hash,
        "selected_epoch": best_epoch,
        "elapsed_seconds": perf_counter() - started,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(adapter_dir / "fnx-r002-provenance.json", provenance)
    del optimizer, scaler, best_state, train_examples, validation_examples
    torch.cuda.empty_cache()
    return model, history, provenance


class CandidateBackend:
    def __init__(self, model: Any, runner: Any) -> None:
        self.model = model
        self.runner = runner

    def score(self, row: Any) -> tuple[Any, float]:
        import numpy as np
        import torch
        from decider.infer import Example, Q
        from decider.model import collate
        from decider.prompt import build

        question = self.runner.question_for(row)
        labels = self.runner.labels_for(row)
        target = self.runner.target_for(row)
        example = Example(
            json.dumps(row.context, sort_keys=True, ensure_ascii=False),
            [Q(question["question"], question["options"], labels.index(target))],
            "fnx-r002",
        )

        class NoShuffle:
            def shuffle(self, values: list[Any]) -> None:
                return None

            def sample(self, values: list[Any], count: int) -> list[Any]:
                return values[:count]

        item = build(example, self.model.tok, NoShuffle(), max_options=10, max_ctx_tokens=2048)
        batch = move_batch(collate([item], self.model.tok.pad_token_id), "cuda")
        self.model.eval()
        started = perf_counter()
        with torch.no_grad():
            probabilities = torch.softmax(self.model(batch), -1)[0, : len(labels)].float().cpu().numpy()
        latency = (perf_counter() - started) * 1000
        return np.asarray(probabilities, dtype=np.float64), latency


def calibrate_candidate(model: Any, runner: Any, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    import numpy as np

    backend = CandidateBackend(model, runner)
    calibration_rows, calibration_hash = runner.read_split("calibration")
    validation_rows, validation_hash = runner.read_split("validation")
    calibration_raw = runner.raw_decider_records(backend, calibration_rows)
    validation_raw = runner.raw_decider_records(backend, validation_rows)
    model_hash = hashlib.sha256(
        f"{config['base_model']}@{config['base_revision']}+FNX-R002-LoRA".encode()
    ).hexdigest()
    global_scaler = runner.fit_scaler(calibration_raw, calibration_hash, model_hash)
    global_temperatures = {"all": global_scaler.temperature}
    primitive_scalers = {}
    for primitive in ("select", "binary", "ordinal"):
        subset = [item for item in calibration_raw if item["primitive"] == primitive]
        if subset:
            primitive_scalers[primitive] = runner.fit_scaler(subset, calibration_hash, model_hash)
    primitive_temperatures = {
        "all": global_scaler.temperature,
        **{key: value.temperature for key, value in primitive_scalers.items()},
    }
    global_validation = runner.calibrated_records(validation_raw, global_temperatures)
    primitive_validation = runner.calibrated_records(validation_raw, primitive_temperatures)
    global_metrics = runner.classification_from_records(global_validation)
    primitive_metrics = runner.classification_from_records(primitive_validation)
    use_primitive = primitive_metrics["nll"] + 0.002 < global_metrics["nll"]
    temperatures = primitive_temperatures if use_primitive else global_temperatures
    validation = runner.calibrated_records(validation_raw, temperatures)
    calibration = runner.calibrated_records(calibration_raw, temperatures)
    correct = np.asarray([item["prediction"] == item["target"] for item in validation])
    uncertainty = {}
    for method in ("maximum_probability", "top1_top2_margin", "normalized_entropy"):
        scores = runner.uncertainty_scores(validation, method)
        uncertainty[method] = {
            "aurc": runner.aurc(scores, correct),
            "mean_score": float(scores.mean()),
        }
    method = min(uncertainty, key=lambda name: uncertainty[name]["aurc"])
    threshold_map = runner.thresholds(
        runner.uncertainty_scores(calibration, method),
        runner.config()["coverage_targets"],
    )
    report = {
        "experiment_id": "FNX-R002-ADAPTED-CALIBRATION",
        "calibration_split_hash": calibration_hash,
        "validation_split_hash": validation_hash,
        "temperatures": temperatures,
        "selected": "primitive_specific" if use_primitive else "global",
        "selected_uncertainty_method": method,
        "uncertainty_methods": uncertainty,
        "thresholds_from_calibration": threshold_map,
        "validation": runner.classification_from_records(validation),
        "production_status": "NOT_PRODUCTION_CERTIFIED",
    }
    return report, backend


def evaluate_candidate(
    backend: CandidateBackend,
    calibration: dict[str, Any],
    runner: Any,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    import numpy as np

    strong_payload = load_json(artifact_dir / "strong-heldout.json")
    strong_by_id = {item["id"]: item for item in strong_payload["records"]}
    raw = []
    split_hashes = {}
    for split in runner.HELDOUT_SPLITS:
        rows, split_hash = runner.read_split(split)
        split_hashes[split.upper()] = split_hash
        raw.extend(runner.raw_decider_records(backend, rows))
    reflex = runner.calibrated_records(raw, calibration["temperatures"])
    method = calibration["selected_uncertainty_method"]
    experiment_config = runner.config()
    points = []
    for target in experiment_config["coverage_targets"]:
        threshold = calibration["thresholds_from_calibration"][f"{target:.1f}"]
        points.append(
            runner.operating_point(
                target,
                threshold,
                reflex,
                strong_by_id,
                method,
                experiment_config,
            )
        )
    baseline = points[0]
    for index, point in enumerate(points):
        point["strong_model_calls_avoided_per_1000"] = (
            baseline["strong_model_calls"] - point["strong_model_calls"]
        ) * 1000 / len(reflex)
        point["normalized_compute_reduction_vs_0pct"] = (
            1 - point["gpu_seconds_per_1000_decisions"] / baseline["gpu_seconds_per_1000_decisions"]
        )
        difference = np.asarray(point["correctness_vector"]) - np.asarray(
            baseline["correctness_vector"]
        )
        interval = runner.bootstrap_interval(
            difference,
            seed=experiment_config["seed"] + 1,
            samples=experiment_config["bootstrap_samples"],
        )
        point["accuracy_difference_vs_0pct"] = float(difference.mean())
        point["accuracy_difference_vs_0pct_95ci"] = interval
        point["quality_consistent_with_baseline"] = interval[0] >= -experiment_config["max_accuracy_loss"]
        point["hard_gates_pass"] = (
            point["false_allow_rate"] <= experiment_config["max_security_false_allow"]
            and point["false_autonomy_rate"] == 0
        )
        point["meaningful_resource_gain"] = (
            point["achieved_reflex_coverage"] >= experiment_config["min_meaningful_reflex_coverage"]
            and point["normalized_compute_reduction_vs_0pct"]
            >= experiment_config["min_normalized_compute_reduction"]
        )
        del point["correctness_vector"]
        runner.emit_live(
            artifact_dir,
            "FINAL_EVAL",
            reflex_coverage=point["achieved_reflex_coverage"],
            system_accuracy=point["system_accuracy"],
            baseline_accuracy=baseline["system_accuracy"],
            false_allow_rate=point["false_allow_rate"],
            false_autonomy_rate=point["false_autonomy_rate"],
            strong_model_escalation_rate=point["strong_model_escalation_rate"],
            human_escalation_rate=point["human_escalation_rate"],
            average_latency_ms=point["average_latency_ms"],
            p50_latency_ms=point["p50_latency_ms"],
            p95_latency_ms=point["p95_latency_ms"],
            decisions_per_second=point["decisions_per_second"],
            **gpu_snapshot(),
            coverage_curve=[
                {
                    "reflex_coverage": prior["achieved_reflex_coverage"],
                    "system_accuracy": prior["system_accuracy"],
                    "average_latency_ms": prior["average_latency_ms"],
                    "strong_model_escalation_rate": prior["strong_model_escalation_rate"],
                }
                for prior in points[: index + 1]
            ],
        )
    useful = [
        point
        for point in points[1:]
        if point["quality_consistent_with_baseline"]
        and point["hard_gates_pass"]
        and point["meaningful_resource_gain"]
    ]
    best = min(useful, key=lambda item: item["gpu_seconds_per_1000_decisions"]) if useful else None
    return points, best, reflex


def write_curve_csv(path: Path, points: list[dict[str, Any]]) -> None:
    fields = [key for key, value in points[0].items() if not isinstance(value, (dict, list))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)


def selected_metrics(result: dict[str, Any], adapted_best: dict[str, Any] | None) -> dict[str, Any]:
    if result["r002_status"] == "ADAPTED_FNX_WINNER" and adapted_best is not None:
        return adapted_best
    return result.get("best_operating_point") or result["baseline_operating_point"]


def run_actual(args: argparse.Namespace, config: dict[str, Any], frozen: dict[str, Any]) -> int:
    if not training_is_required(frozen):
        print("FNX-R002 ADAPTATION: NOT_REQUIRED")
        return 0
    runner = import_runner(args.source_dir)
    runner.emit_live(args.artifact_dir, "TRAINING")
    model, history, provenance = train_candidate(config, runner, args.decider_dir, args.artifact_dir)
    calibration, backend = calibrate_candidate(model, runner, config)
    points, adapted_best, adapted_records = evaluate_candidate(
        backend, calibration, runner, args.artifact_dir
    )
    status, reason = final_status(frozen.get("best_operating_point"), adapted_best, config)
    result = dict(frozen)
    result.update(
        {
            "technical_status": "PASS",
            "r002_status": status,
            "production_status": "NOT_PRODUCTION_CERTIFIED",
            "training_performed": True,
            "adaptation": {
                "architecture": config["architecture"],
                "base_model": config["base_model"],
                "base_revision": config["base_revision"],
                "selected_epoch": provenance["selected_epoch"],
                "promotion_reason": reason,
                "adapted_best_operating_point": adapted_best,
            },
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    reports = args.source_dir / "reports"
    adaptation_report = {
        "experiment_id": "FNX-R002-ADAPTATION",
        "configuration": config,
        "history": history,
        "provenance": provenance,
        "calibration": calibration,
        "best_operating_point": adapted_best,
        "promotion_status": status,
        "promotion_reason": reason,
        "production_status": "NOT_PRODUCTION_CERTIFIED",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    write_json(reports / "R002_ADAPTATION.json", adaptation_report)
    write_json(args.artifact_dir / "adaptation.json", adaptation_report)
    write_json(
        reports / "R002_ADAPTED_COVERAGE_CURVE.json",
        {
            "experiment_id": "FNX-R002-ADAPTED-COVERAGE",
            "points": points,
            "production_status": "NOT_PRODUCTION_CERTIFIED",
        },
    )
    write_curve_csv(reports / "R002_ADAPTED_COVERAGE_CURVE.csv", points)
    write_json(
        args.artifact_dir / "adapted-heldout.json",
        {"records": adapted_records, "production_status": "NOT_PRODUCTION_CERTIFIED"},
    )
    write_json(reports / "R002_RESULTS.json", result)
    selected = selected_metrics(result, adapted_best)
    (args.source_dir / "docs" / "R002_RESULTS.md").write_text(
        "# FNX-R002 — Selective Reflex Cascade results\n\n"
        f"Technical status: **{result['technical_status']}**  \n"
        f"R002 status: **{status}**  \n"
        "Production status: **NOT_PRODUCTION_CERTIFIED**\n\n"
        f"- Promotion reason: {reason}\n"
        f"- Selected system accuracy: {100 * selected['system_accuracy']:.2f}%\n"
        f"- 0% Reflex baseline accuracy: {100 * result['baseline_operating_point']['system_accuracy']:.2f}%\n"
        f"- Selected Reflex coverage: {100 * selected['achieved_reflex_coverage']:.2f}%\n"
        f"- False allow: {100 * selected['false_allow_rate']:.2f}%\n"
        f"- False autonomy: {100 * selected['false_autonomy_rate']:.2f}%\n"
        f"- Strong-model escalation: {100 * selected['strong_model_escalation_rate']:.2f}%\n"
        f"- Human escalation: {100 * selected['human_escalation_rate']:.2f}%\n"
        f"- Average latency: {selected['average_latency_ms']:.2f} ms\n"
        f"- p50 / p95 latency: {selected['p50_latency_ms']:.2f} / {selected['p95_latency_ms']:.2f} ms\n"
        f"- Throughput: {selected['decisions_per_second']:.3f} decisions/sec\n"
        "- Training performed: YES\n",
        encoding="utf-8",
    )
    runner.emit_live(
        args.artifact_dir,
        "COMPLETE",
        reflex_coverage=selected["achieved_reflex_coverage"],
        system_accuracy=selected["system_accuracy"],
        baseline_accuracy=result["baseline_operating_point"]["system_accuracy"],
        false_allow_rate=selected["false_allow_rate"],
        false_autonomy_rate=selected["false_autonomy_rate"],
        strong_model_escalation_rate=selected["strong_model_escalation_rate"],
        human_escalation_rate=selected["human_escalation_rate"],
        average_latency_ms=selected["average_latency_ms"],
        p50_latency_ms=selected["p50_latency_ms"],
        p95_latency_ms=selected["p95_latency_ms"],
        decisions_per_second=selected["decisions_per_second"],
        **gpu_snapshot(),
    )
    print(
        "FNX_R002_ADAPTATION_FINAL="
        + json.dumps({"status": status, "promotion_reason": reason}, sort_keys=True)
    )
    return 0


def record_failure(args: argparse.Namespace, exc: BaseException) -> None:
    failure = {
        "experiment_id": "FNX-R002-ADAPTATION",
        "technical_status": "FAIL",
        "r002_status": "NO_WINNER",
        "error_type": type(exc).__name__,
        "production_status": "NOT_PRODUCTION_CERTIFIED",
        "failed_at": datetime.now(UTC).isoformat(),
    }
    write_json(args.artifact_dir / "adaptation-failure.json", failure)
    result_path = args.source_dir / "reports" / "R002_RESULTS.json"
    if result_path.is_file():
        result = load_json(result_path)
        result.update(
            {
                "technical_status": "FAIL",
                "r002_status": "NO_WINNER",
                "production_status": "NOT_PRODUCTION_CERTIFIED",
                "adaptation_error_type": type(exc).__name__,
                "completed_at": failure["failed_at"],
            }
        )
        write_json(result_path, result)
    try:
        import_runner(args.source_dir).emit_live(
            args.artifact_dir, "FAILED", error_type=type(exc).__name__
        )
    except Exception as telemetry_error:  # noqa: BLE001 - telemetry is explicitly best-effort
        print(
            f"FNX-R002 failure telemetry unavailable: {type(telemetry_error).__name__}",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    frozen = validate_prerequisites(args, config)
    if args.dry_run:
        print(
            "FNX-R002 ADAPTATION DRY RUN: READY "
            f"(training_required={str(training_is_required(frozen)).lower()})"
        )
        return 0
    try:
        return run_actual(args, config, frozen)
    except Exception as exc:  # noqa: BLE001 - all operational failures must persist evidence
        record_failure(args, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
