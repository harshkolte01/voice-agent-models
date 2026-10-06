import json
import time
from pathlib import Path
from urllib import error, request

BASE = "https://addressed-exhibit-detective-possibilities.trycloudflare.com"
key = [
    ln.split("=", 1)[1].strip()
    for ln in Path("api-keys.txt").read_text().splitlines()
    if ln.startswith("Harsh=")
][0]
AUTH = {"Authorization": f"Bearer {key}"}
outdir = Path("stress_stt_clips")
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


def transcribe(path: Path) -> dict:
    boundary = "----stressBoundary7"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"en\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = request.Request(
        BASE + "/v1/audio/transcriptions",
        data=body,
        headers={**AUTH, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
            status = resp.status
    except error.HTTPError as e:
        return {
            "ok": False,
            "status": e.code,
            "err": e.read().decode("utf-8", "replace")[:500],
        }
    except Exception as e:
        return {"ok": False, "status": 0, "err": f"{type(e).__name__}: {e}"}
    client_ms = (time.perf_counter() - t0) * 1000
    payload = json.loads(raw)
    text = payload.get("text") or ""
    lower = text.lower()
    found = [k for k in KEYWORDS if k in lower]
    processing = payload.get("processing_ms")
    audio_dur = payload.get("duration")
    return {
        "ok": status == 200,
        "client_ms": round(client_ms, 1),
        "server_processing_ms": processing,
        "audio_duration_s": audio_dur,
        "rtf": (
            round(float(processing) / 1000 / float(audio_dur), 4)
            if processing and audio_dur
            else None
        ),
        "device": payload.get("device"),
        "model": payload.get("model"),
        "language": payload.get("language"),
        "text_chars": len(text),
        "text_preview": text[:220],
        "keyword_hit_rate": round(len(found) / len(KEYWORDS), 3),
        "keywords_found": found,
        "text": text,
    }


def main() -> None:
    req = request.Request(BASE + "/v1/models", headers=AUTH)
    with request.urlopen(req, timeout=60) as resp:
        models = json.loads(resp.read())
    print("MODELS", json.dumps(models, indent=2)[:800])

    results = {
        "url": BASE,
        "key": "Harsh",
        "stt_model": None,
        "short": None,
        "long": [],
        "repeat_60s": [],
    }
    short = transcribe(outdir / "base_script_16k.wav")
    results["short"] = {k: v for k, v in short.items() if k != "text"}
    results["stt_model"] = short.get("model")
    print(
        "SHORT",
        {
            k: short[k]
            for k in (
                "ok",
                "client_ms",
                "server_processing_ms",
                "audio_duration_s",
                "rtf",
                "model",
                "keyword_hit_rate",
                "text_preview",
            )
        },
    )

    for name, target in [
        ("stress_60s_16k.wav", 60),
        ("stress_180s_16k.wav", 180),
        ("stress_300s_16k.wav", 300),
    ]:
        r = transcribe(outdir / name)
        entry = {k: v for k, v in r.items() if k != "text"}
        entry["target_s"] = target
        entry["file_duration_s"] = target
        entry["file_mb"] = round((outdir / name).stat().st_size / 1e6, 2)
        results["long"].append(entry)
        print(
            f"{target}s",
            {
                k: r[k]
                for k in (
                    "ok",
                    "client_ms",
                    "server_processing_ms",
                    "audio_duration_s",
                    "rtf",
                    "model",
                    "keyword_hit_rate",
                    "text_preview",
                )
            },
        )

    for i in (2, 3):
        r = transcribe(outdir / "stress_60s_16k.wav")
        entry = {k: v for k, v in r.items() if k != "text"}
        entry["run"] = i
        results["repeat_60s"].append(entry)
        print(
            f"repeat60 #{i}",
            {
                k: r[k]
                for k in (
                    "ok",
                    "client_ms",
                    "server_processing_ms",
                    "rtf",
                    "keyword_hit_rate",
                )
                if k in r
            },
        )

    out = outdir / "summary_whisper.json"
    out.write_text(json.dumps(results, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
