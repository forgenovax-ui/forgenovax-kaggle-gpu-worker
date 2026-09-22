# Mutually exclusive worker modes

Set exactly one mode before dispatch:

```bash
MODE=inference bash scripts/run_mode.sh
MODE=training FNX_SOURCE_DIR=/kaggle/working/forgenovax-fnx bash scripts/run_mode.sh
```

`inference` preserves the existing Ollama → authenticated proxy → Cloudflare behavior.

`training` first stops Ollama, proxy, and tunnel; verifies termination; confirms two distinct T4 devices and less than 512 MiB residual allocation on each; configures Hugging Face cache; resolves an exact FNX commit when supplied; validates the dataset hashes; and starts the resumable experiment. It never treats T4 x2 as one 32 GB GPU.

The private FNX source must already be present in `FNX_SOURCE_DIR` or be reachable through an authorized `FNX_REPO_URL`. Credentials are never accepted as command-line values and are never logged. Training output goes to `/kaggle/working/fnx-artifacts/<experiment>` by default.

