# Hermes Setup

Wait for the notebook to show `STATUS: READY`, then configure Hermes as follows:

```text
Provider:
Custom/OpenAI-compatible endpoint

Base URL:
https://xxxxx.trycloudflare.com/v1

API Key:
FORGENOVAX_API_KEY

Model:
qwen3-coder:30b
```

Replace the example hostname with the current URL printed by the notebook. In the API Key field, enter the value stored in the Kaggle secret—not the literal secret name. If the notebook generated a temporary key, use the one-time displayed value.

## Verification

Send this prompt:

```text
You are running through the ForgeNovaX Kaggle GPU Worker.
Reply only with: HERMES_CONNECTED
```

Expected response:

```text
HERMES_CONNECTED
```

The first request after startup or a period of inactivity may be noticeably slower while Ollama loads the model into GPU memory. A 401 response means the key does not match. A connection failure usually means the Kaggle session or Quick Tunnel stopped.
