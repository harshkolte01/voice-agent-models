from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reranker_loaded: bool
    embedder_loaded: bool
    device: str


class ModelInfo(BaseModel):
    id: str
    type: str = "stt"


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    processing_ms: int
    rtf: float
    device: str
    model: str


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped

    @field_validator("documents")
    @classmethod
    def reject_empty_documents(cls, value: list[str]) -> list[str]:
        for index, document in enumerate(value):
            if not document.strip():
                raise ValueError(f"documents[{index}] must not be empty")
        return value


class RerankResult(BaseModel):
    index: int
    score: float
    document: str


class RerankResponse(BaseModel):
    results: list[RerankResult]
    processing_ms: int
    device: str
    model: str


class EmbedRequest(BaseModel):
    input: str | list[str]

    @field_validator("input")
    @classmethod
    def normalize_input(cls, value: str | list[str]) -> list[str]:
        texts = [value] if isinstance(value, str) else value
        if not texts:
            raise ValueError("input must not be empty")
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"input[{index}] must not be empty")
        return texts


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    processing_ms: int
    device: str
    model: str
