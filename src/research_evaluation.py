"""Reproducible model comparison for synthetic professional discussions.

The evaluation uses identical cases, prompts, schemas, and scoring across all
supported models. It never downloads a model implicitly and never converts a
malformed response into a successful finding.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import httpx

from src.benchmark import run_benchmark, unload_model
from src.config import SUPPORTED_MODELS, Settings
from src.research_contracts import CONTRACTS, StructuredOutputError, validate_instance


DEFAULT_CASES = Path(__file__).resolve().parents[1] / "benchmarks" / "research_cases.json"
SYSTEM_PROMPT = (
    "Analyze only the supplied synthetic public professional discussion. "
    "Do not add unsupported facts. Return JSON matching the supplied schema."
)


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Evaluation dataset is unavailable or malformed: {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != "forgenovax-synthetic-research-cases-v1":
        raise ValueError("Unsupported evaluation dataset schema")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain cases")
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every evaluation case must be an object")
        identifier = case.get("id")
        task = case.get("task")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Evaluation case IDs must be unique non-empty strings")
        if task not in CONTRACTS:
            raise ValueError(f"Unknown evaluation task: {task}")
        if not isinstance(case.get("discussion"), str) or not case["discussion"].strip():
            raise ValueError(f"Evaluation case {identifier} has no discussion")
        if not isinstance(case.get("expected"), dict):
            raise ValueError(f"Evaluation case {identifier} has no expected result")
        identifiers.add(identifier)
    return cases


def _contains(value: object, keywords: object, *, require_all: bool) -> bool:
    if isinstance(value, list):
        text = " ".join(str(item) for item in value).lower()
    else:
        text = str(value).lower()
    if not isinstance(keywords, list) or not keywords:
        return False
    matches = [str(keyword).lower() in text for keyword in keywords]
    return all(matches) if require_all else any(matches)


def score_output(case: dict[str, Any], output: dict[str, Any]) -> float:
    expected = case["expected"]
    if case["task"] == "post_classification":
        checks = [
            output.get("category") == expected.get("category"),
            output.get("additional_research_justified")
            is expected.get("additional_research_justified"),
            isinstance(output.get("relevance_score"), int)
            and expected.get("relevance_score_min", 0)
            <= output["relevance_score"]
            <= expected.get("relevance_score_max", 100),
            _contains(
                output.get("reasoning_summary"),
                expected.get("reasoning_keywords"),
                require_all=False,
            ),
        ]
    else:
        checks = [
            _contains(
                output.get("central_argument"),
                expected.get("central_argument_keywords"),
                require_all=True,
            ),
            _contains(
                output.get("unanswered_questions"),
                expected.get("unanswered_question_keywords"),
                require_all=False,
            ),
            output.get("discussion_quality") == expected.get("discussion_quality"),
            _contains(
                output.get("explicitly_supported_business_needs"),
                expected.get("business_need_keywords"),
                require_all=False,
            ),
            output.get("source_grounded") is expected.get("source_grounded"),
        ]
    return round(sum(checks) / len(checks), 4)


def evaluate_case(
    client: httpx.Client,
    *,
    model: str,
    context: int,
    case: dict[str, Any],
) -> dict[str, Any]:
    schema = CONTRACTS[case["task"]]
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Task: {case['task']}\nDiscussion:\n{case['discussion']}",
            },
        ],
        "stream": False,
        "think": False,
        "temperature": 0,
        "max_tokens": 512,
        "options": {"num_ctx": context},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": case["task"],
                "strict": True,
                "schema": schema,
            },
        },
    }
    try:
        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        completion = response.json()
        content = completion["choices"][0]["message"]["content"]
        output = json.loads(content)
        if not isinstance(output, dict):
            raise StructuredOutputError("Structured output must be an object")
        validate_instance(output, schema)
        usage = completion.get("usage")
        return {
            "case_id": case["id"],
            "task": case["task"],
            "success": True,
            "latency_seconds": round(time.monotonic() - started, 3),
            "quality_score": score_output(case, output),
            "usage_metadata_present": isinstance(usage, dict),
            "usage": usage if isinstance(usage, dict) else None,
            "output": output,
            "error": None,
        }
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, StructuredOutputError) as exc:
        return {
            "case_id": case["id"],
            "task": case["task"],
            "success": False,
            "latency_seconds": round(time.monotonic() - started, 3),
            "quality_score": 0.0,
            "usage_metadata_present": False,
            "usage": None,
            "output": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def summarize_results(results: list[dict[str, Any]]) -> dict[str, object]:
    successful = [result for result in results if result["success"]]
    classifications = [
        float(result["quality_score"])
        for result in successful
        if result["task"] == "post_classification"
    ]
    analyses = [
        float(result["quality_score"])
        for result in successful
        if result["task"] == "discussion_analysis"
    ]
    return {
        "case_count": len(results),
        "structured_output_success_rate": round(len(successful) / len(results), 4)
        if results
        else 0.0,
        "relevance_classification_quality": _mean(classifications),
        "factual_extraction_accuracy": _mean(analyses),
        "overall_quality": _mean(
            [float(result["quality_score"]) for result in successful]
        ),
        "usage_metadata_rate": round(
            sum(bool(result["usage_metadata_present"]) for result in results) / len(results),
            4,
        )
        if results
        else 0.0,
        "median_latency_seconds": round(
            statistics.median(float(result["latency_seconds"]) for result in results), 3
        )
        if results
        else None,
    }


def concurrent_probe(
    client: httpx.Client,
    *,
    model: str,
    context: int,
    case: dict[str, Any],
    concurrency: int,
) -> dict[str, object]:
    if not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between 1 and 4")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(evaluate_case, client, model=model, context=context, case=case)
            for _ in range(concurrency)
        ]
        results = [future.result() for future in futures]
    return {
        "concurrency": concurrency,
        "wall_seconds": round(time.monotonic() - started, 3),
        "success_count": sum(bool(result["success"]) for result in results),
        "request_latencies_seconds": [result["latency_seconds"] for result in results],
    }


def recommend_default(trials: list[dict[str, Any]]) -> dict[str, object]:
    available_models = {str(trial["model"]) for trial in trials if trial.get("status") == "measured"}
    missing = [model for model in SUPPORTED_MODELS if model not in available_models]
    if missing:
        return {
            "status": "pending_complete_measurement",
            "model": None,
            "reason": "No recommendation is valid until every supported model is measured.",
            "missing_models": missing,
        }
    eligible = [
        trial
        for trial in trials
        if trial["quality"]["structured_output_success_rate"] >= 0.90
    ]
    if not eligible:
        return {
            "status": "no_eligible_model",
            "model": None,
            "reason": "No model met the 90% structured-output reliability gate.",
            "missing_models": [],
        }
    ranked = sorted(
        eligible,
        key=lambda trial: (
            -(trial["quality"]["overall_quality"] or 0.0),
            trial["quality"]["median_latency_seconds"] or float("inf"),
            trial["performance"]["median_wall_seconds"],
        ),
    )
    winner = ranked[0]
    return {
        "status": "measured_recommendation",
        "model": winner["model"],
        "context_length": winner["context_length"],
        "reason": (
            "Highest schema-gated quality; latency breaks ties. "
            "Retain qwen3-coder:30b as the coding option regardless of this result."
        ),
        "missing_models": [],
    }


def installed_models(client: httpx.Client) -> set[str]:
    response = client.get("/api/tags")
    response.raise_for_status()
    payload = response.json()
    return {
        str(item.get("name", "")).removesuffix(":latest")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }


def load_or_create_report(
    path: Path | None, *, contexts: list[int], cases: Path
) -> dict[str, Any]:
    if path is not None and path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing comparison report is malformed: {path}") from exc
        if not isinstance(report, dict) or report.get("schema") != "forgenovax-research-model-comparison-v1":
            raise ValueError("Existing comparison report has an unsupported schema")
        existing_contexts = report.get("contexts")
        if existing_contexts != contexts:
            raise ValueError(
                f"Existing comparison contexts {existing_contexts} do not match {contexts}"
            )
        return report
    return {
        "schema": "forgenovax-research-model-comparison-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(SUPPORTED_MODELS),
        "contexts": contexts,
        "cases": str(cases),
        "trials": [],
    }


def merge_trials(
    existing: list[dict[str, Any]], additions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        (str(trial.get("model")), int(trial.get("context_length", 0))): trial
        for trial in existing
    }
    for trial in additions:
        by_key[(str(trial.get("model")), int(trial.get("context_length", 0)))] = trial
    return [by_key[key] for key in sorted(by_key)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS))
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/benchmarks"))
    parser.add_argument(
        "--report",
        type=Path,
        help="Append/replace model-context trials in one report for sequential low-disk runs",
    )
    args = parser.parse_args(argv)
    if any(model not in SUPPORTED_MODELS for model in args.models):
        parser.error("--models may contain only supported model identifiers")
    if any(not 512 <= context <= 131072 for context in args.contexts):
        parser.error("--contexts must be between 512 and 131072")

    settings = Settings.from_env()
    settings.validate_for_startup()
    cases = load_cases(args.cases)
    proxy_headers = {"Authorization": f"Bearer {settings.api_key}"}
    report = load_or_create_report(args.report, contexts=args.contexts, cases=args.cases)
    new_trials: list[dict[str, Any]] = []
    try:
        with (
            httpx.Client(base_url=settings.ollama_base_url, timeout=900) as ollama,
            httpx.Client(
                base_url=f"{settings.proxy_base_url}/v1",
                headers=proxy_headers,
                timeout=900,
            ) as proxy,
        ):
            present = installed_models(ollama)
            for model in args.models:
                for context in args.contexts:
                    if model.removesuffix(":latest") not in present:
                        new_trials.append(
                            {
                                "model": model,
                                "context_length": context,
                                "status": "missing_model",
                            }
                        )
                        continue
                    performance = run_benchmark(
                        ollama,
                        model=model,
                        context=context,
                        num_predict=128,
                        warmups=0,
                        runs=args.runs,
                        cold_start=True,
                        measure_first_token=True,
                    )
                    results = [
                        evaluate_case(proxy, model=model, context=context, case=case)
                        for case in cases
                    ]
                    concurrent = concurrent_probe(
                        proxy,
                        model=model,
                        context=context,
                        case=cases[0],
                        concurrency=args.concurrency,
                    )
                    new_trials.append(
                        {
                            "model": model,
                            "context_length": context,
                            "status": "measured",
                            "performance": performance,
                            "quality": summarize_results(results),
                            "case_results": results,
                            "concurrent_requests": concurrent,
                        }
                    )
                    unload_model(ollama, model)
    except (ValueError, OSError, httpx.HTTPError) as exc:
        print(f"EVALUATION_ERROR: {exc}", file=sys.stderr)
        return 1

    report["trials"] = merge_trials(report["trials"], new_trials)
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["recommendation"] = recommend_default(report["trials"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.report or args.output_dir / (
        "research-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Research comparison saved:", target)
    print(json.dumps(report["recommendation"], indent=2))
    return 0 if report["recommendation"]["status"] == "measured_recommendation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
