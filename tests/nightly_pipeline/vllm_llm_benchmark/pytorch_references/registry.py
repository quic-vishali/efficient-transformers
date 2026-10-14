# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Reference-generator registry.

The benchmark orchestrator depends on this small interface instead of
knowing which script implements each model category.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceSpec:
    script: Path
    output_key: str
    requires_max_new_tokens: bool = False


_PACKAGE_DIR = Path(__file__).resolve().parent


def reference_for(endpoint: str, dataset_name: str) -> ReferenceSpec:
    endpoint = endpoint.lower()
    dataset_name = dataset_name.lower()
    if endpoint == "/v1/embeddings":
        return ReferenceSpec(_PACKAGE_DIR / "pytorch_embedding_reference.py", "embeddings")
    if endpoint == "/v1/audio/transcriptions":
        return ReferenceSpec(_PACKAGE_DIR / "pytorch_audio_reference.py", "generated_texts")
    if dataset_name == "custom_image":
        return ReferenceSpec(_PACKAGE_DIR / "pytorch_vlm_reference.py", "generated_texts", True)
    return ReferenceSpec(_PACKAGE_DIR / "pytorch_llm_reference.py", "generated_texts", True)
