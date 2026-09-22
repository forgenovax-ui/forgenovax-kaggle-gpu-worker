# Security

## Non-negotiable boundary

**Ollama is never exposed directly to the public internet.**

The only approved public path is:

```text
Cloudflare
→ authenticated proxy
→ Ollama
```

Both Ollama and the proxy bind to `127.0.0.1`. The tunnel targets the proxy on port 8000. Startup aborts if Ollama is exposed on a wildcard address or if missing/invalid credentials are not rejected.

## Data boundary

This worker is not approved for storing or processing regulated,
customer-sensitive, confidential production data unless a future
ForgeNovaX security review explicitly authorizes such use.

It may be used for engineering, code generation, repository analysis, testing, QA, batch inference, model evaluation, synthetic data, Nova Edge experiments, ForgeQA reasoning, and Company OS batch processing.

It must not be used for Nova Voice production traffic, live trading execution, production authentication, payments, regulated data, customer secrets, persistent customer APIs, 24/7 production workloads, or ForgeNovaX Connect production traffic.

## Threat model and controls

- **Internet scanning or URL discovery:** all `/v1/*` requests require the bearer key.
- **Credential forwarding:** the gateway removes `Authorization` before contacting Ollama.
- **Raw Ollama exposure:** explicit loopback binding and listener verification prevent tunneling port 11434.
- **Secret disclosure in source:** real `.env` files, runtime files, logs, and PIDs are ignored; examples use placeholders.
- **Secret disclosure in access logs:** uvicorn access logging is disabled. Diagnostics redact bearer/header/key-shaped values.
- **Upstream error leakage:** connection errors are converted to generic gateway responses.
- **Stale endpoint use:** Quick Tunnel URLs and processes disappear with the Kaggle runtime; clients must not treat them as stable.

Bearer authentication protects access but does not add per-user authorization, quotas, revocation lists, or audit identities. Anyone holding the current key has worker access. Use a long random key, share it through an approved secret channel, rotate it by restarting the proxy with a new value, and never paste it into source or a committed notebook.

## Ephemeral runtime

Kaggle runtime storage, generated temporary keys, process state, and Quick Tunnel URLs are ephemeral. A reset requires setup again. Cloudflare Quick Tunnels offer no hostname reservation, SLA, access policy, or production availability guarantee.

## Logs

Runtime logs are local to `/kaggle/working`. They must not contain bearer values. The notebook diagnostics view performs a defensive redaction, but prompts and model errors may still contain input-derived text. Do not put sensitive data into prompts.
