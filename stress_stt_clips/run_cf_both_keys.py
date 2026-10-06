"""Smoke-test STT via Cloudflare with every key in api-keys.txt."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import error, request

BASE = "https://viewing-efforts-interview-gonna.trycloudflare.com"
AUDIO = Path("stress_stt_clips/base_script_16k.wav")
KEYWORDS = [
    "quarterly",
    "ten",
    "meeting",
    "conference",
    "budget",
    "feedback",
    "roadmap",
    "hiring",
    "engineering",
]


def load_keys() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in Path("api-keys.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        pairs.append((name.strip(), value.strip()))
    return pairs


def http_json(method: str, url: str, headers: dict, data: bytes | None = None) -> tuple[int, dict | str]:
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=180) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body.decode("utf-8", "replace")[:400]
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw[:400]


def transcribe(key: str) -> dict:
    boundary = "----cfTestBoundary"
    raw = AUDIO.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{AUDIO.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + raw + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"en\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    t0 = time.perf_counter()
    status, payload = http_json(
        "POST",
        f"{BASE}/v1/audio/transcriptions",
        headers,
        body,
    )
    client_ms = round((time.perf_counter() - t0) * 1000, 1)
    if not isinstance(payload, dict):
        return {"ok": False, "status": status, "client_ms": client_ms, "err": payload}
    text = str(payload.get("text") or "")
    found = [k for k in KEYWORDS if k in text.lower()]
    return {
        "ok": status == 200,
        "status": status,
        "client_ms": client_ms,
        "processing_ms": payload.get("processing_ms"),
        "duration": payload.get("duration"),
        "rtf": payload.get("rtf"),
        "device": payload.get("device"),
        "model": payload.get("model"),
        "language": payload.get("language"),
        "keyword_hit_rate": round(len(found) / len(KEYWORDS), 3),
        "keywords_found": found,
        "text_preview": text[:180],
    }


def main() -> None:
    print("BASE", BASE)
    print("AUDIO", AUDIO, "exists=", AUDIO.exists(), "bytes=", AUDIO.stat().st_size if AUDIO.exists() else 0)
    health_status, health = http_json("GET", f"{BASE}/health", {})
    print("HEALTH", health_status, health)

    results = []
    for name, key in load_keys():
        print(f"\n=== key={name} ===")
        m_status, models = http_json(
            "GET",
            f"{BASE}/v1/models",
            {"Authorization": f"Bearer {key}"},
        )
        print("MODELS", m_status, models)
        tr = transcribe(key)
        print("TRANSCRIBE", json.dumps(tr, indent=2))
        results.append({"key": name, "models_status": m_status, "models": models, "transcribe": tr})

    out = Path("stress_stt_clips/cf_both_keys_result.json")
    out.write_text(
        json.dumps({"url": BASE, "health": health, "results": results}, indent=2),
        encoding="utf-8",
    )
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
