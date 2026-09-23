from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reranker_loaded: bool
    embedder_loaded: bool
    tts_loaded: bool = False
    laya_loaded: bool = False
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


class SpeechRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: str = Field(min_length=1, validation_alias=AliasChoices("input", "text"))
    model: str | None = None
    speaker: str = "af_heart"
    voice: str | None = None
    pace: str | None = None

    @field_validator("input")
    @classmethod
    def strip_input(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("input must not be empty")
        return stripped

    @model_validator(mode="after")
    def apply_voice_alias(self):
        if self.voice:
            self.speaker = self.voice
        return self


class LayaQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "score", "noul"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @field_validator("instructions")
    @classmethod
    def strip_instructions(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("instructions must not be empty")
        return stripped

    @model_validator(mode="after")
    def require_criteria_for_typed_questions(self):
        if self.type in {"choice", "score"}:
            if self.criteria is None:
                raise ValueError(f"{self.type} questions require criteria")
            if isinstance(self.criteria, dict):
                if not self.criteria:
                    raise ValueError("criteria must not be empty")
                for key, value in self.criteria.items():
                    if not str(key).strip() or not str(value).strip():
                        raise ValueError("criteria keys and values must not be empty")
            elif isinstance(self.criteria, list):
                if not self.criteria:
                    raise ValueError("criteria must not be empty")
                for index, item in enumerate(self.criteria):
                    if not str(item).strip():
                        raise ValueError(f"criteria[{index}] must not be empty")
        return self


class DecideRequest(BaseModel):
    state: str | dict[str, Any]
    questions: dict[str, LayaQuestion] = Field(min_length=1)
    model: str | None = None

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: str | dict[str, Any]) -> str | dict[str, Any]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("state must not be empty")
            return stripped
        if not value:
            raise ValueError("state must not be empty")
        return value

    @field_validator("questions")
    @classmethod
    def reject_empty_questions(
        cls, value: dict[str, LayaQuestion]
    ) -> dict[str, LayaQuestion]:
        if not value:
            raise ValueError("questions must not be empty")
        for key in value:
            if not key.strip():
                raise ValueError("question names must not be empty")
        return value


class DecideResponse(BaseModel):
    answers: dict[str, Any]
    routing: dict[str, Any] | None = None
    checkpoint: str | None = None
    processing_ms: int
    device: str
    model: str
