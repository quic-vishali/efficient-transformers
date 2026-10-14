#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate a CPU PyTorch reference for fixed image-text VLM inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelForVision2Seq

    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).to("cpu")
    model.eval()
    model_dtype = next(model.parameters()).dtype

    outputs = []
    prompts = []
    for row in rows:
        prompt = str(row["prompt"])
        image_path = Path(row["image_files"][0])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[rendered], images=[image], return_tensors="pt")
        inputs = {
            key: value.to(dtype=model_dtype) if value.is_floating_point() else value for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        tokens = generated[0] if input_len == 0 else generated[0, input_len:]
        outputs.append(processor.tokenizer.decode(tokens, skip_special_tokens=True).strip())
        prompts.append(prompt)

    Path(args.output).write_text(
        json.dumps(
            {"model": args.model, "prompts": prompts, "generated_texts": outputs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
