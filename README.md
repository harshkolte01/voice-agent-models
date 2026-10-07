# Private STT API server

Host-laptop speech-to-text, TTS, rerank, and embedding API. One process loads **one** STT engine (Whisper, NVIDIA Parakeet, or official Fermion Phonon-2), plus Kokoro-82M TTS, `BAAI/bge-reranker-v2-m3`, and `BAAI/bge-m3`.

Callers only need:

```env
STT_API_URL=http://127.0.0.1:8000/v1/audio/transcriptions
TTS_API_URL=http://127.0.0.1:8000/v1/audio/speech
RERANK_API_URL=http://127.0.0.1:8000/v1/rerank
EMBED_API_URL=http://127.0.0.1:8000/v1/embeddings
STT_API_KEY=stt_live_...
```

---

## Local PC setup (Windows)

Do this once on a new machine. Commands are PowerShell. Run them from the repo root.

### 1. Prerequisites

Install these **before** Python packages:

| Need | Why | Check |
| --- | --- | --- |
| [Python 3.11+](https://www.python.org/downloads/) | Runtime | `python --version` |
| [ffmpeg](https://ffmpeg.org/download.html) on `PATH` | Decode mp3/m4a/ogg/webm; Parakeet/Phonon also convert to 16 kHz mono wav | `ffmpeg -version` |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) on `PATH` | Kokoro phonemes. Typical Windows path: `C:\Program Files\eSpeak NG\` | `espeak-ng --version` |
| NVIDIA GPU + CUDA (optional but recommended) | Useful STT / TTS / BGE speed. CPU works, but is slow | `nvidia-smi` |
| [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) (optional) | Public HTTPS URL for this laptop | `cloudflared --version` |

Restart the terminal after adding ffmpeg / espeak-ng to `PATH`.

### 2. Create a virtualenv and install packages

```powershell
cd <path-to-stt-model>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

PyPI `torch` is **CPU-only**. Install matching CUDA 12.8 wheels so rerank, embeddings, Kokoro, and Phonon-2 GPU can use the card. These three packages must come from the **same** index (`cu128`). A mismatched `torchvision` crashes Transformers with `operator torchvision::nms does not exist`.

```powershell
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Phonon-2 extra (only needed if you will set `STT_MODEL=phonon-2`). Install **`--no-deps`** so it does not upgrade `transformers` to 5.x (Kokoro / BGE stay on 4.x):

```powershell
pip install fermion-research --no-deps
pip install soundfile scipy zstandard
```

`nemo_toolkit[asr]` (Parakeet) and `faster-whisper` (Whisper) are already in `requirements.txt`.

Linux / macOS: use `source .venv/bin/activate` and `cp .env.example .env` instead of the PowerShell copies below.

### 3. Create `.env`

The server reads **`.env` in the repo root** (not the example file). Copy it, then edit.

```powershell
copy .env.example .env
notepad .env
```

Minimum you must set for a usable local box:

1. `STT_MODEL` — which speech engine to load (see [step 4](#4-choose-the-stt-engine)).
2. `STT_OPS_TOKEN` — login for `/ops` (generate one; do not reuse an API key).
3. Optional `HF_TOKEN` — Hugging Face token if Hub downloads are rate-limited.

Generate an ops token:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Paste it into `.env` as `STT_OPS_TOKEN=...`.

#### `.env` reference (same keys as `.env.example`)

```env
# --- STT (exactly one engine loads; restart uvicorn after changing this) ---
# Whisper:   large-v3-turbo | whisper | whisper-large-v3-turbo
# Parakeet:  parakeet | parakeet-unified-en-0.6b | nvidia/parakeet-unified-en-0.6b
# Phonon-2:  phonon-2 | phonon | phonon2 | FermionResearch/Phonon-2
STT_MODEL=large-v3-turbo
STT_DEVICE=auto
STT_COMPUTE_TYPE=auto
STT_KEYS_FILE=keys.json
STT_MAX_UPLOAD_MB=25
STT_HOST=127.0.0.1
STT_PORT=8000

# --- Rerank / embeddings (always loaded) ---
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=auto
RERANK_MAX_DOCS=64
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=auto
EMBED_MAX_TEXTS=64
EMBED_MAX_LENGTH=8192

# --- TTS (Kokoro) ---
TTS_ENABLED=true
TTS_DEVICE=auto
TTS_MAX_CHARS=2000
TTS_KOKORO_VOICE=af_heart
TTS_KOKORO_LANG=a

# --- Ops board ---
STT_OPS_TOKEN=
STT_OPS_STORE_SIZE=2000
```

Optional Hub token (not in Settings; Hugging Face reads it from the process env):

```env
HF_TOKEN=hf_...
```

Do not commit `.env` or `keys.json`. Save the file after editing. If the editor still shows `parakeet` but the disk file is `whisper`, the process will load Whisper.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STT_MODEL` | `large-v3-turbo` | STT backend at startup. Restart required to switch |
| `STT_DEVICE` | `auto` | `cuda` if a GPU is visible, else `cpu` |
| `STT_COMPUTE_TYPE` | `auto` | Whisper only: `float16` on CUDA, `int8` on CPU (ignored for Parakeet / Phonon-2) |
| `STT_KEYS_FILE` | `keys.json` | hashed API key store (gitignored) |
| `STT_MAX_UPLOAD_MB` | `25` | upload size limit |
| `STT_HOST` | `127.0.0.1` | bind address when you start with `python -m app.main` |
| `STT_PORT` | `8000` | bind port when you start with `python -m app.main` |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Hugging Face reranker id |
| `RERANK_DEVICE` | `auto` | PyTorch device |
| `RERANK_MAX_DOCS` | `64` | max documents per rerank request |
| `EMBED_MODEL` | `BAAI/bge-m3` | Hugging Face embedding id |
| `EMBED_DEVICE` | `auto` | PyTorch device |
| `EMBED_MAX_TEXTS` | `64` | max strings per embeddings request |
| `EMBED_MAX_LENGTH` | `8192` | tokenizer max tokens per string |
| `TTS_ENABLED` | `true` | expose Kokoro on `/v1/audio/speech` |
| `TTS_DEVICE` | `auto` | PyTorch device for Kokoro |
| `TTS_MAX_CHARS` | `2000` | max TTS input length |
| `TTS_KOKORO_VOICE` | `af_heart` | default Kokoro voice |
| `TTS_KOKORO_LANG` | `a` | fallback Kokoro lang (`a` = American English) |
| `STT_OPS_TOKEN` | empty | operator login for `/ops`. Empty = `/ops` is 404 |
| `STT_OPS_STORE_SIZE` | `2000` | in-memory request history cap |

### 4. Choose the STT engine

Only one STT model is resident. Set `STT_MODEL`, save `.env`, **restart** uvicorn.

| Backend | Set `STT_MODEL` to | `/v1/models` STT id | Notes |
| --- | --- | --- | --- |
| Whisper | `large-v3-turbo` (or `whisper`) | `whisper-large-v3-turbo` | multilingual; `language` form field works |
| Parakeet | `parakeet` | `parakeet-unified-en-0.6b` | English only; needs ffmpeg |
| Phonon-2 | `phonon-2` | `phonon-2` | Official `FermionResearch/Phonon-2` (~164 MB). English only. Pip runtime is CPU; this server moves the dense torch graph to CUDA when `STT_DEVICE=auto` and a GPU is present |

First start downloads model weights from Hugging Face (Whisper / Parakeet / Phonon-2, BGE, Kokoro). That can take several minutes. On Windows without Developer Mode, the server copies Hub cache files (no symlinks) to avoid WinError 1314.

### 5. Create an API key

Plaintext is printed **once**. Only a SHA-256 hash is stored in `keys.json`.

```powershell
python scripts/generate_api_key.py --name "dev-a"
```

Example:

```text
Save this key now; it will not be shown again.
id:          dev-a
name:        dev-a
prefix:      stt_live_AbCd
keys file:   keys.json
STT_API_KEY=stt_live_AbCd...
```

Give callers `STT_API_URL` and `STT_API_KEY`. To revoke a key, set `"revoked": true` on that entry in `keys.json`. Keys reload per request (no restart).

```powershell
$env:STT_API_KEY="stt_live_..."   # paste the value from generate_api_key.py
```

### 6. Start the server

Keep the venv activated. Either start command is fine:

```powershell
python -m app.main
```

That binds `STT_HOST` / `STT_PORT` from `.env` (default `127.0.0.1:8000`). To override on the command line:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait until the log shows `Application startup complete`. Default bind is localhost only.

First-start loads the STT engine from `STT_MODEL`, plus BGE rerank/embed and Kokoro (~200 MB). Licenses: Kokoro Apache-2.0; Parakeet NVIDIA Open Model License; Phonon-2 weights CC-BY-4.0.

### 7. Smoke test

Other terminal (venv optional for curl):

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expect `"status":"ok"` and `"model_loaded":true`. `device` is `cuda` or `cpu`.

```powershell
curl.exe -H "Authorization: Bearer $env:STT_API_KEY" http://127.0.0.1:8000/v1/models
```

Ops board (after `STT_OPS_TOKEN` is set and the process was started with that env):

- `http://127.0.0.1:8000/ops`

Log in with `STT_OPS_TOKEN`, not the API key. The board is cookie-gated so a tunnel cannot be scraped by callers who only have `STT_API_KEY`.

Each request is logged in IST (key id, client IP, Cloudflare `CF-Connecting-IP` when tunneled, country, method, path, status, duration):

```text
2026-09-08 16:41:12 IST  key=dev-a  ip=49.36.11.20  IN  POST /v1/embeddings  200  54ms
```

---

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
- `model` (optional): must match the loaded STT public id

Returned `text` is normalized for all STT engines: dotted `a.m.`/`p.m.` become `AM`/`PM`, spoken hours before AM/PM become digits (`four p.m.` → `4 PM`), and letter/digit gluing like `at10` is spaced.

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

- `processing_ms` — server inference time (not Cloudflare RTT)
- `duration` — audio length in seconds
- `rtf` — `processing_seconds / duration` (below 1.0 is faster than real time)
- `language_probability` — Whisper confidence (0–1); Parakeet and Phonon-2 return `1.0`

Client latency still includes upload + tunnel + download.

### `POST /v1/rerank`

Header: `Authorization: Bearer stt_live_...`

JSON: `query`, `documents` (1 to `RERANK_MAX_DOCS`), optional `top_k`.

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

`index` is the original position in `documents`. Scores are sigmoid-normalized 0–1, high-to-low.

### `POST /v1/embeddings`

Header: `Authorization: Bearer stt_live_...`

JSON: `input` — a string or a list of 1 to `EMBED_MAX_TEXTS` strings.

```json
{
  "embeddings": [[0.01, -0.02]],
  "dim": 1024,
  "processing_ms": 18,
  "device": "cuda",
  "model": "BAAI/bge-m3"
}
```

L2-normalized dense vectors (BGE-M3 CLS pooling). Compare with cosine similarity.

### `POST /v1/audio/speech`

Header: `Authorization: Bearer stt_live_...`

JSON (returns `audio/wav`, 24 kHz mono):

- `input` (required; alias `text`, max 2000 chars). Markdown is stripped before Kokoro
- `model` (optional): `kokoro-82m`
- `speaker` (optional, alias `voice`): e.g. `af_heart`. Short names `Ira`, `Aisha`, `Siya`, `Zoya` map onto Kokoro voices
- `pace` (optional): `slow`, `fast`, `steady`

Headers: `X-Processing-Ms`, `X-Model`, `X-Device`, `X-Speaker`.

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

## Cloudflare Tunnel

Confirm `http://127.0.0.1:8000/health` first.

**Quick tunnel** (random `*.trycloudflare.com` URL):

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

**Named tunnel** after you have a Cloudflare zone:

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

Set `STT_OPS_TOKEN` before exposing `/ops` on a tunnel.

## Tests

```powershell
pytest
```

API tests mock Whisper / Parakeet / Phonon-2, Kokoro, the reranker, and the embedder (no GPU or model download).
