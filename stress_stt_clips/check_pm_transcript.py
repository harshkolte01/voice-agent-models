"""Local roundtrip: Kokoro speak -> Phonon-2 transcribe time phrases."""
from __future__ import annotations

import json
from pathlib import Path
from urllib import error, request

BASE = "http://127.0.0.1:8000"
key = [
    ln.split("=", 1)[1].strip()
    for ln in Path("api-keys.txt").read_text(encoding="utf-8").splitlines()
    if ln.startswith("Harsh=")
][0]
AUTH = {"Authorization": f"Bearer {key}"}
PHRASES = [
    "We have a client meeting Friday at four p.m.",
    "We have a client meeting Friday at 4 p.m.",
    "We have a client meeting Friday at 4 PM",
    "We have a client meeting Friday at four PM",
    "We have a client meeting Friday at four pm",
]


def post_json(path: str, payload: dict) -> bytes:
    req = request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={**AUTH, "Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return resp.read()


def transcribe(wav: bytes, name: str) -> dict:
    boundary = "----pmCheck"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + (
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
    try:
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except error.HTTPError as exc:
        raise RuntimeError(f"transcribe {exc.code}: {exc.read()[:800]!r}") from exc


def main() -> None:
    rows = []
    for i, phrase in enumerate(PHRASES):
        wav = post_json(
            "/v1/audio/speech",
            {"input": phrase, "speaker": "af_heart"},
        )
        result = transcribe(wav, f"pm_{i}.wav")
        text = result.get("text") or ""
        rows.append(
            {
                "spoken": phrase,
                "phonon": text,
                "repr": repr(text),
                "has_p_dot_m_dot": "p.m." in text.lower() or "p. m." in text.lower(),
                "has_pm_no_dots": "pm" in text.lower().replace("p.m.", "").replace("p. m.", ""),
                "model": result.get("model"),
                "device": result.get("device"),
            }
        )
        print(f"SPOKEN  {phrase}")
        print(f"PHONON  {text!r}")
        print()
    Path("stress_stt_clips/pm_roundtrip.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
