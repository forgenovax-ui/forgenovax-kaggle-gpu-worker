# Kaggle Setup

## Before you begin

You need a Kaggle account with GPU access. Kaggle may ask you to verify the account or a phone number before the GPU option appears. The endpoint exists only while the notebook session is running.

## Exact setup steps

1. Sign in to Kaggle.
2. Verify your account or phone if Kaggle requires GPU verification.
3. Create a notebook or open the existing ForgeNovaX worker notebook.
4. Import `notebooks/forgenovax_kaggle_gpu_worker.ipynb`.
5. Open notebook **Settings**.
6. Select **Accelerator → GPU T4 x2**.
7. Enable **Internet**.
8. Add a Kaggle Secret named `FORGENOVAX_API_KEY`. Use a long random value; do not place it in a notebook cell.
9. Click **Run All**.
10. Wait for the final `STATUS: READY` block. The model download and first load can take time.
11. Copy the displayed **OPENAI BASE URL** (it ends in `/v1`).
12. Configure Hermes or another OpenAI-compatible client.

If no secret is available, the notebook creates a strong temporary key and prints it once. Save it immediately. It changes after a runtime reset.

## What the notebook validates

The first cell runs `nvidia-smi` and detects GPU count, names, memory per device, total visible VRAM, and driver. If GPU T4 x2 is not active, setup stops with:

```text
WARNING: Kaggle T4 x2 is not currently active.

Open:
Notebook Settings
→ Accelerator
→ GPU T4 x2

Then restart the session.
```

Afterward it installs services, pulls only the configured model, runs local model inference, checks both authentication rejection paths, verifies Ollama is loopback-only, creates the tunnel, and performs public health, auth, models, and inference tests. A URL alone is not treated as success.

## Re-running after a reset

Kaggle storage and processes are ephemeral. After a full runtime reset, use **Run All** again. The Quick Tunnel URL and any generated temporary key will change.

## Pitfalls seen in the real account (2026-10-25)

- **Importing this notebook (File → Import Notebook) replaces the draft's data
  sources.** Re-attach any datasets the draft used (for example the R002 source
  datasets) and re-check Accelerator and Internet afterwards. Saved versions are
  kept in Version history.
- The notebook clones `FORGENOVAX_REPO_REF` (default: the branch it ships on).
  Override it in the console before **Run All** only for an authorized branch.
- Secret values belong in Add-ons → Secrets → Edit secret → **Value**. Text pasted
  into the console box stays visible on screen; clear it and rotate the key if
  that happens.
- `/readyz` after an idle period loads the model first (measured ≈ 48 s); clients
  should allow at least two minutes for a readiness check.

