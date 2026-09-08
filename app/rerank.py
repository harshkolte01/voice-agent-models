from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from app.config import Settings


@dataclass
class RankedDocument:
    index: int
    document: str
    score: float


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_rerank_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "auto":
        return "cuda" if torch_cuda_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported RERANK_DEVICE: {device}")
    return requested


def _load_rerank_model(model_name: str, device: str):
    import torch
    from app.hf_compat import disable_broken_torchvision

    disable_broken_torchvision()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


class Reranker:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = resolve_rerank_device(device)
        self._lock = threading.Lock()
        self.tokenizer, self.model = _load_rerank_model(
            self.model_name,
            self.device,
        )
        self.loaded = True

    @classmethod
    def from_settings(cls, settings: Settings) -> Reranker:
        return cls(
            model_name=settings.rerank_model,
            device=settings.rerank_device,
        )

    def rank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RankedDocument]:
        import torch

        pairs = [(query, doc) for doc in documents]
        with self._lock:
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded, return_dict=True).logits.view(-1).float()
                scores = torch.sigmoid(logits)
            score_list = scores.detach().cpu().tolist()

        ranked = [
            RankedDocument(
                index=index,
                document=documents[index],
                score=round(float(score_list[index]), 6),
            )
            for index in range(len(documents))
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    async def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RankedDocument]:
        return await asyncio.to_thread(self.rank_sync, query, documents, top_k)
