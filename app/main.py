from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.access_log import AccessLogMiddleware, configure_access_log
from app.auth import require_api_key
from app.config import get_settings
from app.keys import ApiKeyRecord
from app.embed import Embedder
from app.rerank import Reranker
from app.schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SpeechRequest,
    TranscriptionResponse,
)
from app.transcribe import Transcriber, public_model_id
from app.tts import TtsEngine

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
READ_CHUNK_BYTES = 1024 * 1024


def _get_transcriber(app: FastAPI) -> Transcriber:
    transcriber = getattr(app.state, "transcriber", None)
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return transcriber


def _get_reranker(app: FastAPI) -> Reranker:
    reranker = getattr(app.state, "reranker", None)
    if reranker is None:
        raise HTTPException(status_code=503, detail="Reranker is not loaded")
    return reranker


def _get_embedder(app: FastAPI) -> Embedder:
    embedder = getattr(app.state, "embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder is not loaded")
    return embedder


def _get_tts(app: FastAPI) -> TtsEngine:
    tts = getattr(app.state, "tts", None)
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS is not enabled")
    return tts


def _stt_ids(transcriber: Transcriber) -> list[str]:
    listed = getattr(transcriber, "listed_stt_models", None)
    if callable(listed):
        ids = listed()
        if isinstance(ids, list):
            return ids
    return [public_model_id(getattr(transcriber, "model_name", "large-v3-turbo"))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_access_log()
    from app.hf_compat import disable_broken_torchvision

    disable_broken_torchvision()
    settings = get_settings()
    app.state.transcriber = Transcriber.from_settings(settings)
    app.state.reranker = Reranker.from_settings(settings)
    app.state.embedder = Embedder.from_settings(settings)
    app.state.tts = (
        TtsEngine.from_settings(settings) if settings.tts_enabled else None
    )
    yield
    app.state.transcriber = None
    app.state.reranker = None
    app.state.embedder = None
    app.state.tts = None


app = FastAPI(
    title="STT API",
    description="Private speech-to-text, TTS, rerank, and embedding API.",
    lifespan=lifespan,
)
app.add_middleware(AccessLogMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    transcriber = getattr(app.state, "transcriber", None)
    reranker = getattr(app.state, "reranker", None)
    embedder = getattr(app.state, "embedder", None)
    tts = getattr(app.state, "tts", None)
    device = transcriber.device if transcriber is not None else "unknown"
    return HealthResponse(
        status="ok",
        model_loaded=bool(transcriber and transcriber.loaded),
        reranker_loaded=bool(reranker and reranker.loaded),
        embedder_loaded=bool(embedder and embedder.loaded),
        tts_loaded=bool(tts and tts.loaded),
        device=device,
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(_: ApiKeyRecord = Depends(require_api_key)) -> ModelsResponse:
    transcriber = _get_transcriber(app)
    reranker = _get_reranker(app)
    embedder = _get_embedder(app)
    tts = getattr(app.state, "tts", None)
    models = [
        ModelInfo(id=model_id, type="stt")
        for model_id in _stt_ids(transcriber)
    ]
    models.append(ModelInfo(id=reranker.model_name, type="rerank"))
    models.append(ModelInfo(id=embedder.model_name, type="embedding"))
    if tts is not None:
        tts_ids = getattr(tts, "public_ids", None)
        if not tts_ids:
            tts_ids = [tts.public_id]
        for model_id in tts_ids:
            models.append(ModelInfo(id=model_id, type="tts"))
    return ModelsResponse(models=models)


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
    model: str | None = Form(default=None),
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
        try:
            result = await transcriber.transcribe(
                audio_path,
                requested_language,
                model=model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        model=result.model or public_model_id(transcriber.model_name),
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(
    body: RerankRequest,
    _: ApiKeyRecord = Depends(require_api_key),
) -> RerankResponse:
    settings = get_settings()
    if len(body.documents) > settings.rerank_max_docs:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents. Max: {settings.rerank_max_docs}",
        )

    reranker = _get_reranker(app)
    started = time.perf_counter()
    ranked = await reranker.rank(body.query, body.documents, body.top_k)
    processing_ms = int((time.perf_counter() - started) * 1000)

    return RerankResponse(
        results=[
            RerankResult(index=item.index, score=item.score, document=item.document)
            for item in ranked
        ],
        processing_ms=processing_ms,
        device=reranker.device,
        model=reranker.model_name,
    )


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def embed_texts(
    body: EmbedRequest,
    _: ApiKeyRecord = Depends(require_api_key),
) -> EmbedResponse:
    settings = get_settings()
    texts = body.input
    if len(texts) > settings.embed_max_texts:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts. Max: {settings.embed_max_texts}",
        )

    embedder = _get_embedder(app)
    started = time.perf_counter()
    result = await embedder.encode(texts)
    processing_ms = int((time.perf_counter() - started) * 1000)

    return EmbedResponse(
        embeddings=result.embeddings,
        dim=result.dim,
        processing_ms=processing_ms,
        device=embedder.device,
        model=embedder.model_name,
    )


@app.post("/v1/audio/speech")
async def synthesize_speech(
    body: SpeechRequest,
    _: ApiKeyRecord = Depends(require_api_key),
) -> Response:
    settings = get_settings()
    if len(body.input) > settings.tts_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"input must be at most {settings.tts_max_chars} characters",
        )

    tts = _get_tts(app)
    started = time.perf_counter()
    try:
        result = await tts.synthesize(
            text=body.input,
            speaker=body.speaker,
            pace=body.pace,
            model=body.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, AttributeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TTS failed to load. {exc}",
        ) from exc
    processing_ms = int((time.perf_counter() - started) * 1000)

    return Response(
        content=result.wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Processing-Ms": str(processing_ms),
            "X-Model": result.model,
            "X-Device": result.device,
            "X-Speaker": result.speaker,
            "X-Sample-Rate": "24000",
        },
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.stt_host,
        port=settings.stt_port,
    )
