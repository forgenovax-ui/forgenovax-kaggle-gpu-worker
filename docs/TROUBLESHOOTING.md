# Troubleshooting

## The notebook reports that T4 x2 is not active

Open **Notebook Settings → Accelerator → GPU T4 x2**, enable it, and restart the session. Do not continue with a false GPU pass.

## `ollama pull` fails

Confirm Kaggle Internet is enabled and storage remains available. Read `/kaggle/working/ollama.log`, then rerun the pull. Setup intentionally stops rather than marking an absent model ready.

## Ollama never becomes ready

Run `tail -n 100 /kaggle/working/ollama.log`. Confirm `OLLAMA_HOST` is exactly `127.0.0.1:11434`. Run `nvidia-smi` to check runtime health, then use `scripts/shutdown.sh` before restarting.

## Tunnel startup aborts on isolation

Run `ss -ltnp | grep 11434` (or `lsof -nP -iTCP:11434 -sTCP:LISTEN`). Only `127.0.0.1:11434` is accepted. Stop any existing Ollama process and restart it with `scripts/start_ollama.sh`.

## 401 from the public endpoint

The request key does not match the proxy's `FORGENOVAX_API_KEY`, or the `Bearer ` prefix is missing. Restart the proxy after changing a key. Do not expose or print authorization headers while debugging.

## 503 from the proxy

Ollama is unavailable or API authentication was not configured. Check `/kaggle/working/proxy.log`, `/kaggle/working/ollama.log`, and `curl http://127.0.0.1:11434/api/tags` inside the notebook.

## First request is slow

This is expected when the model is first loaded or has been idle longer than the keep-alive duration. Check `ollama ps`; allow the first request to finish before sending another because parallelism defaults to one.

## The URL stopped working

Quick Tunnel URLs are temporary. Confirm the Kaggle session is still running, check `/kaggle/working/cloudflared.log`, and rerun tunnel startup. A new hostname must be copied into the client.

## Health says model installed but not loaded

Installation and loading are different checks. Send one inference request to load the model, then rerun `scripts/status.sh`.

## Diagnostics

Use the notebook's diagnostics cell. It prints recent lines from all three logs and redacts bearer tokens, authorization headers, and API-key assignments. Avoid sharing raw logs until you have separately inspected them for prompt-derived sensitive text.
