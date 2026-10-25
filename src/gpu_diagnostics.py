"""Measured NVIDIA and Ollama placement diagnostics for Kaggle inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

from src.config import Settings


def _csv(command: list[str]) -> list[list[str]]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _integer(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def gpu_inventory() -> dict[str, object]:
    rows = _csv(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, object]] = []
    for row in rows:
        if len(row) != 7:
            continue
        index, uuid, name, used, free, total, utilization = row
        devices.append(
            {
                "index": _integer(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": _integer(used),
                "memory_free_mib": _integer(free),
                "memory_total_mib": _integer(total),
                "utilization_percent": _integer(utilization),
            }
        )
    return {
        "available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
        "separate_memory_devices": True if len(devices) > 1 else None,
    }


def compute_processes(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    by_uuid = {str(device.get("uuid")): device.get("index") for device in devices}
    rows = _csv(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, object]] = []
    for row in rows:
        if len(row) != 4:
            continue
        uuid, pid, process_name, used = row
        processes.append(
            {
                "gpu_index": by_uuid.get(uuid),
                "gpu_uuid": uuid,
                "pid": _integer(pid),
                "process_name": process_name,
                "used_memory_mib": _integer(used),
            }
        )
    return processes


def ollama_cpu_percent() -> float | None:
    try:
        pids = subprocess.run(
            ["pgrep", "-x", "ollama"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if not pids:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", ",".join(pids), "-o", "%cpu="],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [float(value) for value in result.stdout.split()]
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return round(sum(values), 2)


def ollama_placement(client: httpx.Client) -> dict[str, object]:
    try:
        response = client.get("/api/ps")
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return {"available": False, "models": []}
    raw_models = payload.get("models", []) if isinstance(payload, dict) else []
    models: list[dict[str, object]] = []
    for value in raw_models:
        if not isinstance(value, dict):
            continue
        size = value.get("size")
        size_vram = value.get("size_vram")
        offload_ratio: float | None = None
        if (
            isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            and isinstance(size_vram, int)
            and not isinstance(size_vram, bool)
            and size_vram >= 0
        ):
            offload_ratio = round(max(0, size - size_vram) / size, 4)
        models.append(
            {
                "name": value.get("name") or value.get("model"),
                "size_bytes": size,
                "size_vram_bytes": size_vram,
                "cpu_offload_ratio": offload_ratio,
                "excessive_cpu_offload": (
                    offload_ratio is not None and offload_ratio > 0.20
                ),
                "processor": value.get("processor"),
                "context_length": value.get("context_length"),
            }
        )
    return {"available": True, "models": models}


def runtime_snapshot(client: httpx.Client | None = None) -> dict[str, object]:
    inventory = gpu_inventory()
    devices = inventory["devices"]
    result: dict[str, object] = {
        "gpu": inventory,
        "compute_processes": compute_processes(devices),
        "ollama_cpu_percent": ollama_cpu_percent(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    if client is not None:
        result["ollama_placement"] = ollama_placement(client)
    return result


class RuntimeMonitor:
    """Sample GPU memory/utilization and Ollama CPU while inference runs."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        inventory = gpu_inventory()
        self.samples.append(
            {
                "monotonic_seconds": round(time.monotonic(), 3),
                "devices": inventory["devices"],
                "ollama_cpu_percent": ollama_cpu_percent(),
            }
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "RuntimeMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._sample()

    def summary(self) -> dict[str, object]:
        by_index: dict[int, dict[str, int | None]] = {}
        cpu_values: list[float] = []
        for sample in self.samples:
            cpu = sample.get("ollama_cpu_percent")
            if isinstance(cpu, (int, float)) and not isinstance(cpu, bool):
                cpu_values.append(float(cpu))
            for device in sample.get("devices", []):
                if not isinstance(device, dict) or not isinstance(device.get("index"), int):
                    continue
                index = int(device["index"])
                target = by_index.setdefault(
                    index,
                    {"max_memory_used_mib": None, "max_utilization_percent": None},
                )
                for source, destination in (
                    ("memory_used_mib", "max_memory_used_mib"),
                    ("utilization_percent", "max_utilization_percent"),
                ):
                    value = device.get(source)
                    if isinstance(value, int):
                        previous = target[destination]
                        target[destination] = value if previous is None else max(previous, value)
        return {
            "sample_count": len(self.samples),
            "per_device": [
                {"index": index, **values} for index, values in sorted(by_index.items())
            ],
            "max_ollama_cpu_percent": max(cpu_values) if cpu_values else None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    with httpx.Client(base_url=settings.ollama_base_url, timeout=15) as client:
        report = runtime_snapshot(client)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["gpu"]["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
