from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from app.auth import require_api_key
from app.config import get_settings
from app.keys import ApiKeyRecord
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    TranscriptionResponse,
)
from app.transcribe import Transcriber, public_model_id

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
READ_CHUNK_BYTES = 1024 * 1024


def _get_transcriber(app: FastAPI) -> Transcriber:
    transcriber = getattr(app.state, "transcriber", None)
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return transcriber


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.transcriber = Transcriber.from_settings(settings)
    yield
    app.state.transcriber = None


app = FastAPI(
    title="STT API",
    description="Private speech-to-text API. Send audio, receive text.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    transcriber = getattr(app.state, "transcriber", None)
    device = transcriber.device if transcriber is not None else "unknown"
    return HealthResponse(
        status="ok",
        model_loaded=bool(transcriber and transcriber.loaded),
        device=device,
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(_: ApiKeyRecord = Depends(require_api_key)) -> ModelsResponse:
    transcriber = _get_transcriber(app)
    return ModelsResponse(
        models=[
            ModelInfo(id=public_model_id(transcriber.model_name), type="stt"),
        ]
    )


def _suffix_for_upload(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="Missing audio filename")
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {allowed}",
        )
    return suffix


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    _: ApiKeyRecord = Depends(require_api_key),
) -> TranscriptionResponse:
    settings = get_settings()
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    content_length = file.size
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    suffix = _suffix_for_upload(file.filename)
    requested_language = language.strip() if language else None
    transcriber = _get_transcriber(app)

    fd, audio_path = tempfile.mkstemp(suffix=suffix)
    try:
        written = 0
        with os.fdopen(fd, "wb") as tmp:
            fd = -1
            while True:
                chunk = await file.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="File too large")
                tmp.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        started = time.perf_counter()
        result = await transcriber.transcribe(audio_path, requested_language)
        processing_ms = int((time.perf_counter() - started) * 1000)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    rtf = 0.0
    if result.duration > 0:
        rtf = round((processing_ms / 1000.0) / result.duration, 4)

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
        processing_ms=processing_ms,
        rtf=rtf,
        device=transcriber.device,
        model=public_model_id(transcriber.model_name),
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.stt_host,
        port=settings.stt_port,
    )
