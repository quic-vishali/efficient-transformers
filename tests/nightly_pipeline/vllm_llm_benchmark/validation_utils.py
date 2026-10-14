"""Reusable output-comparison helpers for benchmark validation."""

from __future__ import annotations

import functools
import math
import os
import re


def cosine_similarity(left: object, right: object) -> float:
    """Return cosine similarity for two numeric vectors."""
    if not isinstance(left, list) or not isinstance(right, list) or not left or len(left) != len(right):
        raise ValueError("cosine validation requires equal-length numeric vectors")
    a = [float(x) for x in left]
    b = [float(x) for x in right]
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if denominator < 1e-12:
        raise ValueError("cosine validation cannot compare a zero vector")
    return sum(x * y for x, y in zip(a, b)) / denominator


@functools.lru_cache(maxsize=1)
def _text_similarity_model():
    """Load the fixed production text encoder once per benchmark process."""
    from transformers import AutoModel, AutoTokenizer

    model_name = os.environ.get(
        "VLLM_TEXT_VALIDATION_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu")
    model.eval()
    return tokenizer, model


def _encode_texts_for_similarity(texts: list[str]) -> list[list[float]]:
    """Encode text with a fixed sentence encoder for semantic fallback checks."""
    import torch
    import torch.nn.functional as functional

    tokenizer, model = _text_similarity_model()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embeddings = functional.normalize(pooled, p=2, dim=1)
    return embeddings.cpu().tolist()


def text_cosine_fallback(actual: list[str], reference: list[str]) -> float:
    """Return the worst pairwise text cosine score."""
    if len(actual) != len(reference) or not actual:
        return 0.0
    actual_vectors = _encode_texts_for_similarity(actual)
    reference_vectors = _encode_texts_for_similarity(reference)
    scores = [cosine_similarity(left, right) for left, right in zip(actual_vectors, reference_vectors)]
    return min(scores) if scores else 0.0


def normalize_transcription(value: object) -> str:
    """Normalize harmless ASR formatting differences before comparison."""
    text = re.sub(r"\|[^|]+\|", " ", str(value).lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))
