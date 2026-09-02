# Private STT API server

Host-laptop speech-to-text API. The Whisper model stays on this machine. Clients only need:

```env
STT_API_URL=http://127.0.0.1:8000/v1/audio/transcriptions
STT_API_KEY=stt_live_...
```

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH` (faster-whisper uses it for mp3/m4a/ogg/webm)
- NVIDIA GPU + CUDA for `large-v3-turbo` at useful speed (CPU works, but is slow)
- Optional: [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) to expose localhost over HTTPS

## Setup

```powershell
cd C:\Coding\stt-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Generate an API key

Plaintext is printed **once**. Only a SHA-256 hash is stored in `keys.json`.

```powershell
python scripts/generate_api_key.py --name "dev-a"
```

Example output:

```text
Save this key now; it will not be shown again.
id:          dev-a
name:        dev-a
prefix:      stt_live_AbCd
keys file:   keys.json
STT_API_KEY=stt_live_AbCd...
```

Give the caller `STT_API_URL` and `STT_API_KEY`. To revoke a key, set `"revoked": true` on that entry in `keys.json`. The server reloads keys per request, so no restart is required.

## Run the server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The process binds to localhost only. Use a tunnel if you need a public URL.

First start downloads `large-v3-turbo` via faster-whisper (CTranslate2). Later starts reuse the Hugging Face cache.

## API

### `GET /health` (no auth)

```json
{ "status": "ok", "model_loaded": true, "device": "cuda" }
```

### `GET /v1/models`

Header: `Authorization: Bearer stt_live_...`

```json
{ "models": [{ "id": "whisper-large-v3-turbo", "type": "stt" }] }
```

### `POST /v1/audio/transcriptions`

Header: `Authorization: Bearer stt_live_...`

`multipart/form-data`:

- `file` (required): wav, mp3, m4a, ogg, flac, or webm (max 25 MB)
- `language` (optional): e.g. `en`

```json
{
  "text": "What time is the meeting tomorrow?",
  "language": "en",
  "language_probability": 0.98,
  "duration": 2.74,
  "processing_ms": 386,
  "rtf": 0.141,
  "device": "cuda",
  "model": "whisper-large-v3-turbo"
}
```

Timing fields:

- `processing_ms` — server inference time (not Cloudflare network RTT)
- `duration` — audio length in seconds
- `rtf` — real-time factor (`processing_seconds / duration`). Below 1.0 means faster than real time
- `language_probability` — Whisper language-detection confidence (0–1)

Total client latency is still measured on their side (`time around requests.post`), because that includes upload + tunnel + download.

### curl

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" http://127.0.0.1:8000/v1/models

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -F "file=@test.wav;type=audio/wav" `
  -F "language=en" `
  http://127.0.0.1:8000/v1/audio/transcriptions
```

## Config

Copied from `.env.example`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper model name |
| `STT_DEVICE` | `auto` | `cuda` if a GPU is visible, else `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | `float16` on CUDA, `int8` on CPU |
| `STT_KEYS_FILE` | `keys.json` | hashed key store (gitignored) |
| `STT_MAX_UPLOAD_MB` | `25` | upload size limit |
| `STT_HOST` | `127.0.0.1` | bind address |
| `STT_PORT` | `8000` | bind port |

## Cloudflare Tunnel

The API must work on `http://127.0.0.1:8000` first. Then optionally:

**Quick tunnel** (random `*.trycloudflare.com` URL):

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

**Named tunnel** later, after you have a Cloudflare zone:

```yaml
# ~/.cloudflared/config.yml
tunnel: <TUNNEL_ID>
credentials-file: C:\Users\<you>\.cloudflared\<TUNNEL_ID>.json

ingress:
  - hostname: stt.example.com
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Clients then use:

```env
STT_API_URL=https://stt.example.com/v1/audio/transcriptions
STT_API_KEY=stt_live_...
```

## Tests

```powershell
pytest
```

API tests mock Whisper so they do not need a GPU or model download.
