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
- Compare 4,096 and 8,192-token research contexts before using 32,768.
- Keep `qwen3-coder:30b` available for coding, but do not assume the largest
  model is the best routine research model.
- `FORGENOVAX_CUDA_VISIBLE_DEVICES` may explicitly select detected indexes.
  Invalid or absent indexes fail startup.
- GPU acceleration is required by default. Set `FORGENOVAX_REQUIRE_GPU=0` only
  for an intentional non-Kaggle diagnostic environment.

Final model placement and performance tables belong in the generated
`research-comparison-*.json` report. Values must come from a live run; this
document does not substitute theoretical figures.
