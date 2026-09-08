# Private STT API server

Host-laptop speech-to-text, TTS, rerank, and embedding API. Whisper, SraVaani, rumik-oss, `BAAI/bge-reranker-v2-m3`, and `BAAI/bge-m3` stay on this machine. Clients only need:

```env
STT_API_URL=http://127.0.0.1:8000/v1/audio/transcriptions
TTS_API_URL=http://127.0.0.1:8000/v1/audio/speech
RERANK_API_URL=http://127.0.0.1:8000/v1/rerank
EMBED_API_URL=http://127.0.0.1:8000/v1/embeddings
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
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
copy .env.example .env
```

`pip install torch` from PyPI is CPU-only. The second command installs matching CUDA 12.8 wheels (`torch`, `torchvision`, `torchaudio`) so rerank, embeddings, SraVaani, and rumik-oss can use the GPU. Those three packages must come from the same index; a mismatched `torchvision` crashes Transformers with `operator torchvision::nms does not exist`. Whisper uses CTranslate2 and does not depend on that Torch build.

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

Each request is logged in IST with key id, client IP (Cloudflare `CF-Connecting-IP` when tunneled), country, method, path, status, and duration:

```text
2026-09-08 16:41:12 IST  key=dev-a  ip=49.36.11.20  IN  POST /v1/embeddings  200  54ms
```

First start downloads `large-v3-turbo` via faster-whisper (CTranslate2), plus `BAAI/bge-reranker-v2-m3` and `BAAI/bge-m3` via Hugging Face. SraVaani and rumik-oss stay **lazy**: they download and occupy the GPU only on first use, and they swap with each other so both are not in VRAM at once.

SraVaani is gated. Accept the license at [ARTPARK-IISc/SraVaani-1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) and set `HF_TOKEN` in `.env`. rumik-oss 1 is CC-BY-NC (research / non-commercial); see [the model card](https://huggingface.co/rumik-ai/rumik-oss-1).

## API

### `GET /health` (no auth)

```json
{ "status": "ok", "model_loaded": true, "reranker_loaded": true, "embedder_loaded": true, "sravaani_loaded": false, "tts_loaded": false, "device": "cuda" }
```

### `GET /v1/models`

Header: `Authorization: Bearer stt_live_...`

```json
{
  "models": [
    { "id": "whisper-large-v3-turbo", "type": "stt" },
    { "id": "sravaani-1.0", "type": "stt" },
    { "id": "BAAI/bge-reranker-v2-m3", "type": "rerank" },
    { "id": "BAAI/bge-m3", "type": "embedding" },
    { "id": "rumik-oss-1", "type": "tts" }
  ]
}
```

### `POST /v1/audio/transcriptions`

Header: `Authorization: Bearer stt_live_...`

`multipart/form-data`:

- `file` (required): wav, mp3, m4a, ogg, flac, or webm (max 25 MB)
- `language` (optional): e.g. `en` or `hi`
- `model` (optional): `whisper-large-v3-turbo` (default) or `sravaani-1.0`

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

### `POST /v1/rerank`

Header: `Authorization: Bearer stt_live_...`

JSON body:

- `query` (required): search or ranking query
- `documents` (required): 1 to `RERANK_MAX_DOCS` strings
- `top_k` (optional): return only the top N results

```json
{
  "results": [
    { "index": 1, "score": 0.87, "document": "doc b" }
  ],
  "processing_ms": 12,
  "device": "cuda",
  "model": "BAAI/bge-reranker-v2-m3"
}
```

`index` is the original position in `documents`. Scores are sigmoid-normalized 0–1. Results are sorted high-to-low.

### `POST /v1/embeddings`

Header: `Authorization: Bearer stt_live_...`

JSON body:

- `input` (required): a string or a list of 1 to `EMBED_MAX_TEXTS` strings

```json
{
  "embeddings": [[0.01, -0.02]],
  "dim": 1024,
  "processing_ms": 18,
  "device": "cuda",
  "model": "BAAI/bge-m3"
}
```

Vectors are L2-normalized dense embeddings (BGE-M3 CLS pooling). Compare them with cosine similarity.

### `POST /v1/audio/speech`

Header: `Authorization: Bearer stt_live_...`

JSON body (returns `audio/wav`, 24 kHz mono):

- `input` (required): text to speak (alias: `text`, max 2000 chars)
- `speaker` (optional): `Ira`, `Aisha`, `Siya`, or `Zoya` (alias: `voice`)
- `tone` (optional): `happy`, `sad`, `angry`, `excited`, `professional`
- `accent` (optional): `Hindi`, `Telugu`, `Tamil`, `Kannada`, `Bengali`, `Punjabi`, `Indian English`
- `pace` (optional): `slow`, `fast`, `steady`

You can also put a raw rumik description in `input`:

```text
<description="happy, Hindi accent, steady pace"> नमस्ते, आज आपका दिन कैसा रहा?
```

Inline vocalizations supported by the model: `<laugh>`, `<chuckle>`, `<sigh>`.

Response headers: `X-Processing-Ms`, `X-Model`, `X-Device`, `X-Speaker`.

rumik-oss is about 3B parameters. On a 12 GB GPU it is swapped with SraVaani so they do not sit in VRAM together. Do not set both `SRAVAANI_LOAD_ON_STARTUP` and `TTS_LOAD_ON_STARTUP`.

### curl

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" http://127.0.0.1:8000/v1/models

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -F "file=@test.wav;type=audio/wav" `
  -F "language=en" `
  http://127.0.0.1:8000/v1/audio/transcriptions

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -F "file=@hindi.mp3;type=audio/mpeg" `
  -F "model=sravaani-1.0" `
  -F "language=hi" `
  http://127.0.0.1:8000/v1/audio/transcriptions

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{ "input": "Namaste", "speaker": "Ira", "tone": "happy", "accent": "Hindi", "pace": "steady" }' `
  http://127.0.0.1:8000/v1/audio/speech `
  --output speech.wav

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{ "query": "meeting notes", "documents": ["doc a", "doc b"] }' `
  http://127.0.0.1:8000/v1/rerank

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{ "input": ["meeting notes", "standup is at 10am"] }' `
  http://127.0.0.1:8000/v1/embeddings
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
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Hugging Face reranker id |
| `RERANK_DEVICE` | `auto` | PyTorch `cuda` if a GPU is visible, else `cpu` |
| `RERANK_MAX_DOCS` | `64` | max documents per rerank request |
| `EMBED_MODEL` | `BAAI/bge-m3` | Hugging Face embedding id |
| `EMBED_DEVICE` | `auto` | PyTorch `cuda` if a GPU is visible, else `cpu` |
| `EMBED_MAX_TEXTS` | `64` | max strings per embeddings request |
| `EMBED_MAX_LENGTH` | `8192` | tokenizer max tokens per string |
| `SRAVAANI_ENABLED` | `true` | expose `sravaani-1.0` on transcriptions |
| `SRAVAANI_MODEL` | `ARTPARK-IISc/SraVaani-1.0` | Hugging Face id (gated; needs `HF_TOKEN`) |
| `SRAVAANI_DEVICE` | `auto` | PyTorch device for SraVaani |
| `SRAVAANI_LOAD_ON_STARTUP` | `false` | load SraVaani at boot instead of first request |
| `TTS_ENABLED` | `true` | expose rumik-oss on `/v1/audio/speech` |
| `TTS_MODEL` | `rumik-ai/rumik-oss-1` | Hugging Face id |
| `TTS_DEVICE` | `auto` | PyTorch device for rumik-oss |
| `TTS_LOAD_ON_STARTUP` | `false` | load rumik-oss at boot instead of first request |
| `TTS_MAX_CHARS` | `2000` | max TTS input length |
| `TTS_DEFAULT_SPEAKER` | `Ira` | default rumik voice |
| `HF_TOKEN` | empty | Hugging Face token for gated SraVaani |

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

API tests mock Whisper, SraVaani, rumik-oss, the reranker, and the embedder so they do not need a GPU or model download.
