# GPU optimization report

## Live baseline — 2026-10-25

Measured in the existing Kaggle notebook owned by `forgenovaxinc`:

- Accelerator setting: NVIDIA T4 x2
- Internet: enabled
- Persistence: none
- Environment: pinned to the notebook's original 2026-06-30 image
- GPU 0: Tesla T4, 15,360 MiB reported memory
- GPU 1: Tesla T4, 15,360 MiB reported memory
- Driver: 580.178.04
- CUDA reported by PyTorch: 12.8
- GPU compute capability: 7.5 on each device
- PyTorch: 2.10.0+cu128; CUDA available; two devices visible
- CPU: 4 logical processors
- RAM: 31.35 GiB total, 29.65 GiB available at baseline
- `/kaggle/working`: 19.52 GiB total, 19.50 GiB free at baseline
- Initial GPU allocation: 0 MiB and no compute process on either device
- Initial Ollama state: not installed in the fresh ephemeral session

The two GPUs are reported and evaluated as separate memory devices. No report
adds their VRAM and describes it as one automatically pooled 30/32 GB device.

## Runtime evidence collected by the worker

`python -m src.gpu_diagnostics` records typed GPU inventory, free/used memory,
utilization, compute-process memory by GPU UUID/index, `CUDA_VISIBLE_DEVICES`,
Ollama CPU percentage, and `/api/ps` model placement. The benchmark samples
these values during inference and reports maximum observed values per device.

Ollama's measured `size` and `size_vram` are used to calculate CPU-offload
ratio. More than 20% outside VRAM is flagged as excessive. A device is never
claimed to host model work merely because it is visible; process memory must be
measured on that device.

## Safe operating defaults

- `OLLAMA_NUM_PARALLEL=1` until concurrent-request evidence supports more.
- `OLLAMA_SCHED_SPREAD=1` distributes the active model across both T4s. This is
  now the Kaggle default because the controlled qwen3:14b comparison below
  measured higher throughput than Ollama's default single-GPU placement.
- Use 8,192 tokens as the measured default context; 32,768 remains opt-in.
- Use `qwen3-coder:30b` as the measured research and coding default.
- `FORGENOVAX_CUDA_VISIBLE_DEVICES` may explicitly select detected indexes.
  Invalid or absent indexes fail startup.
- GPU acceleration is required by default. Set `FORGENOVAX_REQUIRE_GPU=0` only
  for an intentional non-Kaggle diagnostic environment.

## Live scheduler comparison

Identical qwen3:14b steady-state runs used one warm-up, three measured runs,
8,192 context, first-token streaming, and the same synthetic prompt.

| Scheduler | Median generation tok/s | Median wall time | GPU 0 peak | GPU 1 peak |
|---|---:|---:|---:|---:|
| Ollama default | 20.43 | 2.992 s | 10,149 MiB / 99% | 3 MiB / 2% |
| `OLLAMA_SCHED_SPREAD=1` | 25.62 | 2.389 s | 5,493 MiB / 49% | 5,489 MiB / 52% |

Spread scheduling increased measured throughput by 25.4% and reduced wall time
by 20.2%. The first enhanced cold/warm run also recorded a 23.479 s cold wall
time (8.193 s first token) and a 2.889 s warm wall time (0.050 s first token).
The cold figure includes disk/model loading and is not used for the scheduler
percentage because the later spread run benefited from the operating-system
page cache.

## Live model placement

All three supported models were fully GPU-resident with zero measured CPU
offload. Values are sampled maxima from the model-comparison run.

| Model | Ollama size / VRAM bytes | GPU 0 peak | GPU 1 peak | Max Ollama CPU |
|---|---:|---:|---:|---:|
| `qwen3:14b` | 10,489,430,015 / 10,489,430,015 | 5,493 MiB / 49% | 5,489 MiB / 52% | 0.3% |
| `gpt-oss:20b` | 12,962,454,567 / 12,962,454,567 | 6,609 MiB / 58% | 6,699 MiB / 76% | 0.4% |
| `qwen3-coder:30b` | 19,793,570,692 / 19,793,570,692 | 9,841 MiB / 59% | 9,311 MiB / 41% | 39.6% |

The 30B download occupied about 17.3 GiB of the 19.5 GiB working filesystem,
so multiple large models and a duplicate cache snapshot cannot coexist safely.
Sequential model rotation is the supported evaluation path.
