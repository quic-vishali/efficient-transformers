#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate CPU PyTorch reference embeddings for fixed JSONL text inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pooling-method", default="mean", choices=("mean", "avg", "cls", "max"))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP

    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [str(row["prompt"]) for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to("cpu")
    model.eval()
    model_dtype = next(model.parameters()).dtype

    with torch.inference_mode():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {
            key: value.to(dtype=model_dtype) if value.is_floating_point() else value for key, value in inputs.items()
        }
        outputs = model(**inputs)
        pooling_fn = POOLING_MAP.get(args.pooling_method)
        if pooling_fn is None:
            raise ValueError(f"unsupported pooling method: {args.pooling_method}")
        embeddings = pooling_fn(outputs.last_hidden_state, inputs["attention_mask"])

    vectors = embeddings.cpu().tolist()
    Path(args.output).write_text(
        json.dumps({"model": args.model, "prompts": texts, "embeddings": vectors}, indent=2) + "\n",
        encoding="utf-8",
    )
    if vectors:
        print(f"generated {len(vectors)} embedding(s), dimension={len(vectors[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
