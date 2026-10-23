"""Measure real local Ollama throughput on the current Kaggle GPU runtime.

Synthetic prompts only. Results come from Ollama generation counters and
nvidia-smi; the script never estimates GPU placement or invents performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx

from src.config import Settings


PROMPT = (
    "Read this synthetic business discussion and return three short sentences "
    "identifying the main point, one unanswered question, and a constructive "
    "reply angle. Discussion: An operations leader says their team loses time "
    "switching between separate scheduling and reporting tools. A commenter "
    "asks how they could measure the time lost before selecting new software. "
    "Another commenter recommends establishing a baseline before automation."
)


def gpu_inventory() -> dict[str, object]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"available": False, "devices": []}
    devices = []
    for line in r.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            devices.append(dict(zip(
                ("index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent"),
                fields,
            )))
    return {"available": bool(devices), "devices": devices}


def _rate(tokens: int, duration_ns: int) -> float | None:
    return round(tokens * 1e9 / duration_ns, 2) if tokens > 0 and duration_ns > 0 else None


def one_run(client: httpx.Client, *, model: str, context: int, num_predict: int) -> dict[str, object]:
    started = time.monotonic()
    response = client.post("/api/generate", json={
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "think": False,
        "options": {"num_ctx": context, "num_predict": num_predict, "temperature": 0},
    })
    response.raise_for_status()
    result: Any = response.json()
    if not isinstance(result, dict) or not result.get("done") or not isinstance(result.get("response"), str):
        raise ValueError("Ollama returned an incomplete response")
    if not result["response"].strip():
        raise ValueError("Ollama generated empty output")
    fields = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
              "eval_count", "eval_duration")
    metrics: dict[str, int] = {}
    for name in fields:
        value = result.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Ollama did not report valid {name}")
        metrics[name] = value
    return {
        "wall_seconds": round(time.monotonic() - started, 3),
        "total_seconds": round(metrics["total_duration"] / 1e9, 3),
        "load_seconds": round(metrics["load_duration"] / 1e9, 3),
        "prompt_tokens": metrics["prompt_eval_count"],
        "output_tokens": metrics["eval_count"],
        "prompt_tokens_per_second": _rate(metrics["prompt_eval_count"], metrics["prompt_eval_duration"]),
        "generation_tokens_per_second": _rate(metrics["eval_count"], metrics["eval_duration"]),
    }


def run_benchmark(client: httpx.Client, *, model: str, context: int,
                  num_predict: int, warmups: int, runs: int) -> dict[str, object]:
    if min(context, num_predict, runs) < 1 or warmups < 0 or runs > 30 or warmups > 10:
        raise ValueError("Require positive context/tokens, 1-30 runs and 0-10 warmups")
    before = gpu_inventory()
    for _ in range(warmups):
        one_run(client, model=model, context=context, num_predict=num_predict)
    values = [one_run(client, model=model, context=context, num_predict=num_predict)
              for _ in range(runs)]
    after = gpu_inventory()
    speeds = [float(v["generation_tokens_per_second"])
              for v in values if v["generation_tokens_per_second"] is not None]
    return {
        "schema": "forgenovax-inference-benchmark-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "context_length": context,
        "max_output_tokens": num_predict,
        "warmups": warmups,
        "runs": values,
        "median_generation_tokens_per_second": round(statistics.median(speeds), 2) if speeds else None,
        "median_wall_seconds": round(statistics.median(float(v["wall_seconds"]) for v in values), 3),
        "gpu_before": before,
        "gpu_after": after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("MODEL", "qwen3-coder:30b"))
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", default=os.getenv("FORGENOVAX_WORK_DIR", "/kaggle/working") + "/benchmarks")
    args = parser.parse_args(argv)
    try:
        config = Settings.from_env()
        with httpx.Client(base_url=config.ollama_base_url, timeout=httpx.Timeout(900)) as client:
            result = run_benchmark(client, model=args.model, context=args.context,
                num_predict=args.max_output_tokens, warmups=args.warmups, runs=args.runs)
        target_dir = Path(args.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / ("benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Benchmark saved:", target)
        print("Model:", result["model"])
        print("Median generation tokens/sec:", result["median_generation_tokens_per_second"])
        print("Median wall seconds:", result["median_wall_seconds"])
        print("GPU inventory recorded:", result["gpu_before"]["available"], result["gpu_after"]["available"])
        return 0
    except (ValueError, httpx.HTTPError, OSError) as exc:
        print(f"BENCHMARK_ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
