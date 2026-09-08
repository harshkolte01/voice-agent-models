from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from app.config import Settings
from app.rerank import torch_cuda_available


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    dim: int


def resolve_embed_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "auto":
        return "cuda" if torch_cuda_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported EMBED_DEVICE: {device}")
    return requested


def _load_embed_model(model_name: str, device: str):
    import torch
    from app.hf_compat import disable_broken_torchvision

    disable_broken_torchvision()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


class Embedder:
    def __init__(self, model_name: str, device: str, max_length: int):
        self.model_name = model_name
        self.device = resolve_embed_device(device)
        self.max_length = max_length
        self._lock = threading.Lock()
        self.tokenizer, self.model = _load_embed_model(
            self.model_name,
            self.device,
        )
        self.loaded = True

    @classmethod
    def from_settings(cls, settings: Settings) -> Embedder:
        return cls(
            model_name=settings.embed_model,
            device=settings.embed_device,
            max_length=settings.embed_max_length,
        )

    def encode_sync(self, texts: list[str]) -> EmbeddingResult:
        import torch

        vectors: list[list[float]] = []
        batch_size = 8
        with self._lock:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    output = self.model(**encoded)
                    cls_tokens = output.last_hidden_state[:, 0].float()
                    normalized = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
                vectors.extend(normalized.detach().cpu().tolist())

        dim = len(vectors[0]) if vectors else 0
        return EmbeddingResult(embeddings=vectors, dim=dim)

    async def encode(self, texts: list[str]) -> EmbeddingResult:
        return await asyncio.to_thread(self.encode_sync, texts)
