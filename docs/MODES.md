# Mutually exclusive worker modes

Set exactly one mode before dispatch:

```bash
MODE=inference bash scripts/run_mode.sh
MODE=training FNX_SOURCE_DIR=/kaggle/working/forgenovax-fnx bash scripts/run_mode.sh
```

`inference` preserves the existing Ollama → authenticated proxy → Cloudflare behavior.

`training` first stops Ollama, proxy, and tunnel; verifies termination; confirms two distinct T4 devices and less than 512 MiB residual allocation on each; configures Hugging Face cache; resolves an exact FNX commit when supplied; validates the dataset hashes; and starts the resumable experiment. It never treats T4 x2 as one 32 GB GPU.

FNX-R003 uses the stricter frozen runner:

```bash
FNX_R003_SOURCE_GIT_SHA=<exact-40-character-sha> \
FNX_R003_STAGE=pipeline \
bash scripts/run_r003_reflex.sh
```

The pipeline stage runs the heads-only and LoRA development comparisons and selects by
validation loss without accessing final held-out labels. After the main repository has
recorded `R003 PRE-TRAINING: READY`, run the same command with
`FNX_R003_STAGE=full`. The runner never starts the inference proxy, Cloudflare tunnel,
OBS, or a public broadcast. Its private archive is written to the Kaggle notebook
output area with a SHA-256 manifest.

The private FNX source must already be present in `FNX_SOURCE_DIR` or be reachable through an authorized `FNX_REPO_URL`. Credentials are never accepted as command-line values and are never logged. Training output goes to `/kaggle/working/fnx-artifacts/<experiment>` by default.
