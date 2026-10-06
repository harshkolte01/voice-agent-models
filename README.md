# Private STT API server

Host-laptop speech-to-text, TTS, rerank, and embedding API. STT is Whisper `large-v3-turbo`, NVIDIA Parakeet Unified EN 0.6B, or official Fermion Phonon-2 (chosen by `STT_MODEL`), plus Kokoro-82M, `BAAI/bge-reranker-v2-m3`, and `BAAI/bge-m3` on this machine. Clients only need:

```env
STT_API_URL=http://127.0.0.1:8000/v1/audio/transcriptions
TTS_API_URL=http://127.0.0.1:8000/v1/audio/speech
RERANK_API_URL=http://127.0.0.1:8000/v1/rerank
EMBED_API_URL=http://127.0.0.1:8000/v1/embeddings
STT_API_KEY=stt_live_...
```

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) on `PATH` (Whisper, Parakeet, and Phonon-2 use it for mp3/m4a/ogg/webm; Parakeet and Phonon-2 also need 16 kHz mono wav)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) on `PATH` (Kokoro uses it for phonemes; typical Windows install is `C:\Program Files\eSpeak NG\`)
- NVIDIA GPU + CUDA for Whisper / Parakeet / Phonon-2 at useful speed (CPU works, but is slow). Official `fermion-research` pip is a CPU engine; this server relocates Phonon-2's dense torch graph onto CUDA when `STT_DEVICE=auto|cuda`
- For Parakeet: `pip install "nemo_toolkit[asr]>=2.7.3"` (see `requirements.txt`)
- For Phonon-2: `pip install fermion-research --no-deps` then `pip install soundfile scipy zstandard` (keep transformers 4.x for Kokoro/BGE)
- Optional: [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) to expose localhost over HTTPS

## Setup

```powershell
cd F:\stt-model
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install fermion-research --no-deps
pip install soundfile scipy zstandard
copy .env.example .env
```

`pip install torch` from PyPI is CPU-only. The second command installs matching CUDA 12.8 wheels (`torch`, `torchvision`, `torchaudio`) so rerank, embeddings, and Kokoro can use the GPU. Those three packages must come from the same index; a mismatched `torchvision` crashes Transformers with `operator torchvision::nms does not exist`. Whisper uses CTranslate2 and does not depend on that Torch build.

### Choose Whisper, Parakeet, or Phonon-2

Only one STT engine loads. Set `STT_MODEL` in `.env`, then **restart** uvicorn:

```env
# Whisper (faster-whisper large-v3-turbo) — default in .env.example
STT_MODEL=large-v3-turbo

# NVIDIA Parakeet Unified EN 0.6B (NeMo) — English only, typically faster on long audio
STT_MODEL=parakeet

# Official Fermion Phonon-2 (FermionResearch/Phonon-2 via fermion-research) — English only
STT_MODEL=phonon-2
```

Accepted aliases:

| Backend | `STT_MODEL` values | Public id in `/v1/models` |
| --- | --- | --- |
| Whisper | `whisper`, `large-v3-turbo`, `whisper-large-v3-turbo` | `whisper-large-v3-turbo` |
| Parakeet | `parakeet`, `parakeet-unified-en-0.6b`, `nvidia/parakeet-unified-en-0.6b` | `parakeet-unified-en-0.6b` |
| Phonon-2 | `phonon`, `phonon-2`, `phonon2`, `FermionResearch/Phonon-2` | `phonon-2` |

Parakeet needs ffmpeg on `PATH` (converts uploads to 16 kHz mono wav) and `nemo_toolkit[asr]` from `requirements.txt`. On this host, Parakeet was ~3–4× faster than Whisper on 1–5 minute clips with similar keyword accuracy on clean English TTS.

Phonon-2 loads the **official** 164 MB `FermionResearch/Phonon-2` weights (CC-BY-4.0), not a community export. Long files are windowed inside the Fermion engine (~35 s). On this host, dense fp32 weights are moved onto **CUDA** (Fermion's pip package itself is CPU-only; NVIDIA's official CUDA image is Docker).

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

### Ops dashboard

Live request board at `/ops` (localhost or the Cloudflare URL). It shows key id, path, status, total latency, inference time, model, device, IP, country, and a system strip (model load, CPU/RAM/GPU).

Set a dedicated operator token in `.env` (not an API client key):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

```env
STT_OPS_TOKEN=paste-the-token-here
```

Restart the API process, then open:

- `http://127.0.0.1:8000/ops`
- `https://<your-tunnel>.trycloudflare.com/ops`

Log in with `STT_OPS_TOKEN`. Without that env var, `/ops` stays off (404). The board is cookie-gated so the tunnel cannot be scraped by callers who only have `STT_API_KEY`.

First start loads the STT engine chosen by `STT_MODEL` (`large-v3-turbo` via faster-whisper / CTranslate2, `parakeet` via NeMo `nvidia/parakeet-unified-en-0.6b`, or `phonon-2` via official `FermionResearch/Phonon-2`), plus `BAAI/bge-reranker-v2-m3`, `BAAI/bge-m3`, and Kokoro-82M (`hexgrad/Kokoro-82M`, ~200 MB). Only one STT backend is resident. Kokoro is Apache-2.0; Parakeet is under the NVIDIA Open Model License; Phonon-2 weights are CC-BY-4.0. On Windows without Developer Mode, the server forces Hugging Face Hub to copy cache files (no symlinks) to avoid WinError 1314.

## API

### `GET /health` (no auth)

```json
{ "status": "ok", "model_loaded": true, "reranker_loaded": true, "embedder_loaded": true, "tts_loaded": true, "device": "cuda" }
```

### `GET /v1/models`

Header: `Authorization: Bearer stt_live_...`

```json
{
  "models": [
    { "id": "whisper-large-v3-turbo", "type": "stt" },
    { "id": "BAAI/bge-reranker-v2-m3", "type": "rerank" },
    { "id": "BAAI/bge-m3", "type": "embedding" },
    { "id": "kokoro-82m", "type": "tts" }
  ]
}
```

With `STT_MODEL=parakeet`, the STT entry is `{ "id": "parakeet-unified-en-0.6b", "type": "stt" }`. With `STT_MODEL=phonon-2`, it is `{ "id": "phonon-2", "type": "stt" }`.

### `POST /v1/audio/transcriptions`

Header: `Authorization: Bearer stt_live_...`

`multipart/form-data`:

- `file` (required): wav, mp3, m4a, ogg, flac, or webm (max 25 MB)
- `language` (optional): e.g. `en` or `hi` (Whisper). Parakeet and Phonon-2 are English-only and always return `en`
- `model` (optional): must match the loaded STT public id (`whisper-large-v3-turbo`, `parakeet-unified-en-0.6b`, or `phonon-2`)

Returned `text` is normalized for all STT engines: dotted `a.m.`/`p.m.` become `AM`/`PM` (so naive sentence splitters do not cut `four p.m.`), spoken hours before AM/PM become digits (`four p.m.` → `4 PM`), and letter/digit gluing like `at10` is spaced.

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
- `language_probability` — Whisper language-detection confidence (0–1); Parakeet and Phonon-2 return `1.0`

Total client latency is still measured on their side (`time around requests.post`), because that includes upload + tunnel + download.

To switch engines, see [Choose Whisper, Parakeet, or Phonon-2](#choose-whisper-parakeet-or-phonon-2).

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

- `input` (required): text to speak (alias: `text`, max 2000 chars). Markdown is stripped on this server before Kokoro, so `*`, `#`, links, and similar markup are not spoken. Cloudflare, LAN, and local callers all hit the same path.
- `model` (optional): `kokoro-82m` (default)
- `speaker` (optional, alias: `voice`): Kokoro voice id such as `af_heart` (default). Short names `Ira`, `Aisha`, `Siya`, and `Zoya` still map onto Kokoro voices
- `pace` (optional): `slow`, `fast`, `steady`

Response headers: `X-Processing-Ms`, `X-Model`, `X-Device`, `X-Speaker`.

### curl

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" http://127.0.0.1:8000/v1/models

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -F "file=@test.wav;type=audio/wav" `
  -F "language=en" `
  http://127.0.0.1:8000/v1/audio/transcriptions

curl.exe -H "Authorization: Bearer $env:STT_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{ "input": "Namaste", "speaker": "af_heart" }' `
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
| `STT_MODEL` | `large-v3-turbo` | STT backend at startup: Whisper (`large-v3-turbo` / `whisper` / `whisper-large-v3-turbo`), Parakeet (`parakeet` / `parakeet-unified-en-0.6b` / `nvidia/parakeet-unified-en-0.6b`), or official Phonon-2 (`phonon-2` / `phonon` / `FermionResearch/Phonon-2`). Restart required to switch |
| `STT_DEVICE` | `auto` | `cuda` if a GPU is visible, else `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | Whisper only: `float16` on CUDA, `int8` on CPU (ignored for Parakeet and Phonon-2) |
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
| `TTS_ENABLED` | `true` | expose Kokoro on `/v1/audio/speech` |
| `TTS_DEVICE` | `auto` | PyTorch device for Kokoro |
| `TTS_MAX_CHARS` | `2000` | max TTS input length |
| `TTS_KOKORO_VOICE` | `af_heart` | default Kokoro voice |
| `TTS_KOKORO_LANG` | `a` | fallback Kokoro lang code (`a` = American English) |
| `STT_OPS_TOKEN` | empty | operator login for `/ops` (required for Cloudflare) |
| `STT_OPS_STORE_SIZE` | `2000` | in-memory request history cap |

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

API tests mock Whisper / Parakeet / Phonon-2, Kokoro, the reranker, and the embedder so they do not need a GPU or model download.
