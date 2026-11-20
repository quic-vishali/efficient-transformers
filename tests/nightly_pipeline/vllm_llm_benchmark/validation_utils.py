"""Reusable output-comparison helpers for benchmark validation."""

from __future__ import annotations

import functools
import math
import os
import re

from QEfficient.utils.logging_utils import QEFFLogger

DEFAULT_TEXT_VALIDATION_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_TEXT_VALIDATION_SEQUENCE_LENGTH = 4000
TEXT_VALIDATION_BATCH_SIZE = 1

logger = QEFFLogger.get_logger("INFRA")


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


def _get_max_sequence_length(tokenizer, model) -> int:
    """Return a safe maximum sequence length supported by tokenizer and model."""
    limits = [MAX_TEXT_VALIDATION_SEQUENCE_LENGTH]
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(model.config, "max_position_embeddings", None)

    for limit in (tokenizer_limit, model_limit):
        if isinstance(limit, int) and 0 < limit < 1_000_000:
            limits.append(limit)

    return min(limits)


@functools.lru_cache(maxsize=1)
def _text_similarity_model():
    """Load the fixed production text encoder once per benchmark process."""
    import transformers
    from packaging.version import Version

    model_name = os.environ.get(
        "VLLM_TEXT_VALIDATION_MODEL",
        DEFAULT_TEXT_VALIDATION_MODEL,
    )
    if model_name == DEFAULT_TEXT_VALIDATION_MODEL and Version(transformers.__version__) < Version("4.51.0"):
        raise RuntimeError(f"Qwen3-Embedding requires Transformers >= 4.51.0; found {transformers.__version__}.")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        padding_side="left",
        truncation_side="right",
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=False,
    ).to("cpu")
    model.eval()
    max_sequence_length = _get_max_sequence_length(tokenizer, model)
    return tokenizer, model, max_sequence_length


def _last_token_pool(last_hidden_state, attention_mask):
    """Pool the final non-padding token from each sequence."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths,
    ]


def _encode_texts_for_similarity(texts: list[str], side: str = "text") -> list[list[float]]:
    """Encode text with Qwen3 for semantic fallback checks."""
    import torch
    import torch.nn.functional as F

    tokenizer, model, max_sequence_length = _text_similarity_model()
    vectors = []

    for start in range(0, len(texts), TEXT_VALIDATION_BATCH_SIZE):
        batch_texts = texts[start : start + TEXT_VALIDATION_BATCH_SIZE]
        untruncated = tokenizer(
            batch_texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        for offset, input_ids in enumerate(untruncated["input_ids"]):
            token_count = len(input_ids)
            if token_count > max_sequence_length:
                logger.warning(
                    "Truncating %s validation text %d from %d to %d tokens",
                    side,
                    start + offset,
                    token_count,
                    max_sequence_length,
                )

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = model(**inputs)
            pooled = _last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            embeddings = F.normalize(pooled, p=2, dim=1)
        vectors.extend(embeddings.cpu().tolist())

    return vectors


def text_cosine_fallback(actual: list[str], reference: list[str]) -> float:
    """Return the worst pairwise text cosine score."""
    if len(actual) != len(reference) or not actual:
        raise ValueError("text similarity validation requires non-empty, equal-length actual and reference lists")
    actual_vectors = _encode_texts_for_similarity(actual, "actual")
    reference_vectors = _encode_texts_for_similarity(reference, "reference")
    scores = [cosine_similarity(left, right) for left, right in zip(actual_vectors, reference_vectors)]
    return min(scores) if scores else 0.0


def normalize_transcription(value: object) -> str:
    """Normalize harmless ASR formatting differences before comparison."""
    text = re.sub(r"\|[^|]+\|", " ", str(value).lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))
