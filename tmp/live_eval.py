from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

BASE = "https://notify-towers-shipment-bookstore.trycloudflare.com"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp" / "live_eval"
PHRASE = "The meeting is at ten in the morning."
VOICES = ("af_heart", "am_liam", "bf_emma")


def load_keys() -> dict[str, str]:
    keys = {}
    for line in (ROOT / "api-keys.txt").read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, key = line.split("=", 1)
        keys[name.strip()] = key.strip()
    return keys


def words(text: str) -> list[str]:
    return [part.lower() for part in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if part]


def wer(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    rows = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        prev = rows[0]
        rows[0] = i
        for j, hyp_word in enumerate(hyp, start=1):
            insert = rows[j] + 1
            delete = rows[j - 1] + 1
            sub = prev + (0 if ref_word == hyp_word else 1)
            prev = rows[j]
            rows[j] = min(insert, delete, sub)
    return rows[-1] / len(ref)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = load_keys()
    client = httpx.Client(timeout=180.0, follow_redirects=True)
    rows: list[dict] = []

    def timed(method: str, path: str, **kwargs):
        started = time.perf_counter()
        response = client.request(method, f"{BASE}{path}", **kwargs)
        client_ms = int((time.perf_counter() - started) * 1000)
        return response, client_ms

    health, health_ms = timed("GET", "/health")
    rows.append(
        {
            "group": "health",
            "owner": "-",
            "check": "GET /health",
            "status": health.status_code,
            "ok": health.status_code == 200,
            "client_ms": health_ms,
            "server_ms": None,
            "accuracy": None,
            "detail": health.json() if health.status_code == 200 else health.text[:200],
        }
    )

    bad, bad_ms = timed("GET", "/v1/models", headers={"Authorization": "Bearer stt_live_invalid"})
    rows.append(
        {
            "group": "auth",
            "owner": "invalid",
            "check": "GET /v1/models bad key",
            "status": bad.status_code,
            "ok": bad.status_code == 401,
            "client_ms": bad_ms,
            "server_ms": None,
            "accuracy": None,
            "detail": "rejected",
        }
    )

    for owner, key in keys.items():
        headers = {"Authorization": f"Bearer {key}"}
        models, models_ms = timed("GET", "/v1/models", headers=headers)
        body = models.json() if models.headers.get("content-type", "").startswith("application/json") else {}
        ids = [item["id"] for item in body.get("models", [])] if models.status_code == 200 else []
        rows.append(
            {
                "group": "auth",
                "owner": owner,
                "check": "GET /v1/models",
                "status": models.status_code,
                "ok": models.status_code == 200 and "whisper-large-v3-turbo" in ids and "kokoro-82m" in ids,
                "client_ms": models_ms,
                "server_ms": None,
                "accuracy": None,
                "detail": ids,
            }
        )

        embed, embed_ms = timed(
            "POST",
            "/v1/embeddings",
            headers=headers,
            json={
                "input": [
                    "standup is at 10am",
                    "the daily standup starts at 10 in the morning",
                    "pizza toppings and dessert recipes",
                ]
            },
        )
        embed_body = embed.json() if embed.status_code == 200 else {}
        vectors = embed_body.get("embeddings", [])
        related = cosine(vectors[0], vectors[1]) if len(vectors) == 3 else None
        unrelated = cosine(vectors[0], vectors[2]) if len(vectors) == 3 else None
        embed_ok = (
            embed.status_code == 200
            and related is not None
            and unrelated is not None
            and related > unrelated
        )
        rows.append(
            {
                "group": "embeddings",
                "owner": owner,
                "check": "POST /v1/embeddings",
                "status": embed.status_code,
                "ok": embed_ok,
                "client_ms": embed_ms,
                "server_ms": embed_body.get("processing_ms"),
                "accuracy": None if related is None else round(related - unrelated, 4),
                "detail": {
                    "related_cosine": None if related is None else round(related, 4),
                    "unrelated_cosine": None if unrelated is None else round(unrelated, 4),
                    "dim": embed_body.get("dim"),
                    "model": embed_body.get("model"),
                },
            }
        )

        rerank, rerank_ms = timed(
            "POST",
            "/v1/rerank",
            headers=headers,
            json={
                "query": "when is standup",
                "documents": [
                    "standup is at 10am",
                    "the office wifi password is on the fridge",
                    "quarterly budget spreadsheet",
                ],
            },
        )
        rerank_body = rerank.json() if rerank.status_code == 200 else {}
        results = rerank_body.get("results", [])
        top_ok = bool(results) and results[0].get("index") == 0
        rows.append(
            {
                "group": "rerank",
                "owner": owner,
                "check": "POST /v1/rerank",
                "status": rerank.status_code,
                "ok": rerank.status_code == 200 and top_ok,
                "client_ms": rerank_ms,
                "server_ms": rerank_body.get("processing_ms"),
                "accuracy": None if not results else round(float(results[0]["score"]), 4),
                "detail": {
                    "top_index": None if not results else results[0].get("index"),
                    "top_doc": None if not results else results[0].get("document"),
                    "model": rerank_body.get("model"),
                },
            }
        )

        for voice in VOICES:
            tts, tts_ms = timed(
                "POST",
                "/v1/audio/speech",
                headers={**headers, "Content-Type": "application/json"},
                json={"input": PHRASE, "speaker": voice},
            )
            wav_path = OUT_DIR / f"{owner}_{voice}.wav"
            if tts.status_code == 200:
                wav_path.write_bytes(tts.content)
            tts_error = None
            if tts.status_code != 200:
                try:
                    tts_error = tts.json()
                except Exception:
                    tts_error = tts.text[:400]
            rows.append(
                {
                    "group": "tts",
                    "owner": owner,
                    "check": f"TTS {voice}",
                    "status": tts.status_code,
                    "ok": tts.status_code == 200 and tts.headers.get("x-model") == "kokoro-82m",
                    "client_ms": tts_ms,
                    "server_ms": int(tts.headers["x-processing-ms"]) if "x-processing-ms" in tts.headers else None,
                    "accuracy": None,
                    "detail": {
                        "x_model": tts.headers.get("x-model"),
                        "x_speaker": tts.headers.get("x-speaker"),
                        "bytes": len(tts.content),
                        "error": tts_error,
                    },
                }
            )
            if tts.status_code != 200:
                continue
            with wav_path.open("rb") as handle:
                stt, stt_ms = timed(
                    "POST",
                    "/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": (wav_path.name, handle, "audio/wav")},
                    data={"language": "en"},
                )
            stt_body = stt.json() if stt.status_code == 200 else {}
            transcript = stt_body.get("text", "")
            error = wer(PHRASE, transcript)
            rows.append(
                {
                    "group": "stt",
                    "owner": owner,
                    "check": f"STT from {voice}",
                    "status": stt.status_code,
                    "ok": stt.status_code == 200 and error <= 0.35,
                    "client_ms": stt_ms,
                    "server_ms": stt_body.get("processing_ms"),
                    "accuracy": None if stt.status_code != 200 else round(1.0 - error, 4),
                    "detail": {
                        "transcript": transcript,
                        "wer": None if stt.status_code != 200 else round(error, 4),
                        "rtf": stt_body.get("rtf"),
                        "duration": stt_body.get("duration"),
                        "model": stt_body.get("model"),
                    },
                }
            )

    payload = {"base": BASE, "phrase": PHRASE, "rows": rows}
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
