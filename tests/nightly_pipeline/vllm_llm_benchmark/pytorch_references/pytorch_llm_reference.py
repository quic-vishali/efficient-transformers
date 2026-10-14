#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate a deterministic Hugging Face PyTorch reference for benchmark inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-chat-template", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    if "delta" in args.model.lower():
        raise RuntimeError(
            "delta checkpoints are not standalone models; merge the delta with the matching base model first"
        )

    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(rows)} prompt(s) from {dataset}", flush=True)

    print(f"Loading model configuration: {args.model}", flush=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    model_class = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
    device = "cpu"
    print(f"Loading PyTorch model on {device} using native checkpoint dtype", flush=True)
    load_started = time.monotonic()
    model = model_class.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    print(f"Model loaded in {time.monotonic() - load_started:.1f}s; dtype={model_dtype}", flush=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [str(item["prompt"]) for item in rows]
    generated_texts = []
    print(f"Starting inference for {len(prompts)} prompt(s)", flush=True)
    with torch.inference_mode():
        for index, prompt in enumerate(prompts, start=1):
            print(f"Preparing prompt {index}/{len(prompts)}", flush=True)
            formatted_prompt = prompt
            if not args.skip_chat_template and not is_encoder_decoder and getattr(tokenizer, "chat_template", None):
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            inputs = {
                key: value.to(dtype=model_dtype) if value.is_floating_point() else value
                for key, value in inputs.items()
            }
            print(
                f"Prompt {index}/{len(prompts)} has "
                f"{inputs['input_ids'].shape[1]} input token(s); "
                f"generating up to {args.max_new_tokens} token(s)",
                flush=True,
            )
            generation_started = time.monotonic()
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            print(
                f"Prompt {index}/{len(prompts)} completed in {time.monotonic() - generation_started:.1f}s",
                flush=True,
            )
            generated_texts.append(
                tokenizer.decode(
                    output[0] if is_encoder_decoder else output[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
            )

    print(f"Writing reference output to {args.output}", flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "model": args.model,
                "prompts": prompts,
                "generated_texts": generated_texts,
                "is_encoder_decoder": is_encoder_decoder,
                "device": "cpu",
                "dtype": str(model_dtype),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("PyTorch inference completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
