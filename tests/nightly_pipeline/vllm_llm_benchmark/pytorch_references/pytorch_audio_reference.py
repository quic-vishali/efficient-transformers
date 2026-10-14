#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Generate a CPU PyTorch speech-to-text reference for a fixed audio JSONL input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    dataset = Path(args.dataset)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).to("cpu")
    model.eval()
    model_dtype = next(model.parameters()).dtype

    outputs = []
    for row in rows:
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = dataset.parent.parent.parent / audio_path
            if not audio_path.exists():
                audio_path = Path.cwd() / row["audio"]
        audio, sample_rate = sf.read(audio_path)
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {
            key: value.to(dtype=model_dtype) if value.is_floating_point() else value for key, value in inputs.items()
        }
        with torch.inference_mode():
            tokens = model.generate(**inputs)
        outputs.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())

    Path(args.output).write_text(
        json.dumps({"model": args.model, "generated_texts": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
