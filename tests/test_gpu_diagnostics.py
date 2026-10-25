import httpx

from src import gpu_diagnostics


def test_gpu_inventory_and_per_device_process_memory(monkeypatch) -> None:
    outputs = {
        "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu": [
            ["0", "GPU-a", "Tesla T4", "100", "14900", "15360", "55"],
            ["1", "GPU-b", "Tesla T4", "200", "14800", "15360", "65"],
        ],
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory": [
            ["GPU-a", "123", "ollama_llama_server", "90"],
            ["GPU-b", "123", "ollama_llama_server", "180"],
        ],
    }
    monkeypatch.setattr(
        gpu_diagnostics,
        "_csv",
        lambda command: next(value for key, value in outputs.items() if key in command),
    )

    inventory = gpu_diagnostics.gpu_inventory()
    processes = gpu_diagnostics.compute_processes(inventory["devices"])

    assert inventory["device_count"] == 2
    assert inventory["separate_memory_devices"] is True
    assert processes[1]["gpu_index"] == 1
    assert processes[1]["used_memory_mib"] == 180


def test_ollama_placement_detects_excessive_cpu_offload() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:14b",
                        "size": 10_000,
                        "size_vram": 6_000,
                        "processor": "60% GPU",
                    }
                ]
            },
        )
    )
    with httpx.Client(base_url="http://test", transport=transport) as client:
        result = gpu_diagnostics.ollama_placement(client)

    assert result["models"][0]["cpu_offload_ratio"] == 0.4
    assert result["models"][0]["excessive_cpu_offload"] is True
