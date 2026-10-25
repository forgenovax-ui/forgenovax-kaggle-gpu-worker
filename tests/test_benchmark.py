import httpx
import pytest

from src import benchmark


def fake_generate(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    assert request.url.path == "/api/generate"
    data = __import__("json").loads(request.content)
    assert data["stream"] is False
    assert data["options"]["num_ctx"] == 4096
    assert "credentials" not in data["prompt"].lower()
    return httpx.Response(200, json={
        "done": True,
        "response": "Synthetic discussion: a baseline is needed.",
        "total_duration": 3_000_000_000,
        "load_duration": 10_000_000,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": 100,
        "eval_duration": 2_000_000_000,
    })


def test_benchmark_computes_real_reported_token_rates(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "gpu_inventory", lambda: {
        "available": True, "devices": [{"index": "0", "name": "Test T4"}]
    })
    with httpx.Client(base_url="http://127.0.0.1:11434",
                      transport=httpx.MockTransport(fake_generate)) as client:
        result = benchmark.run_benchmark(
            client, model="test:latest", context=4096, num_predict=128, warmups=1, runs=3
        )
    assert len(result["runs"]) == 3
    assert result["median_generation_tokens_per_second"] == 50.0
    assert result["runs"][0]["prompt_tokens_per_second"] == 20.0
    assert result["runs"][0]["total_seconds"] == 3.0
    assert result["gpu_after"]["available"] is True
    assert "runtime_monitor" in result


def test_incomplete_ollama_metrics_fail_instead_of_guessing() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"done": True, "response": "hello"})
    )
    with httpx.Client(base_url="http://127.0.0.1:11434", transport=transport) as client:
        with pytest.raises(ValueError, match="valid total_duration"):
            benchmark.one_run(client, model="test", context=1024, num_predict=32)


@pytest.mark.parametrize("runs,warmups", [(0, 0), (31, 0), (1, -1), (1, 11)])
def test_invalid_benchmark_limits(runs: int, warmups: int) -> None:
    with httpx.Client(base_url="http://127.0.0.1:11434",
                      transport=httpx.MockTransport(fake_generate)) as client:
        with pytest.raises(ValueError):
            benchmark.run_benchmark(
                client, model="test", context=4096, num_predict=128,
                warmups=warmups, runs=runs
            )
