# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import torch

from .check_audio_models import check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100

audio_models = {"openai/whisper-tiny": "optimum-intel-internal-testing/tiny-random-whisper"}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = set(audio_models.values())
else:
    test_models = set(audio_models.keys())


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name, manual_cleanup):
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, export_compile_only=True, manual_cleanup=manual_cleanup)


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name, manual_cleanup):
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup=manual_cleanup)


# FP16 export + FP16 Compile
@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_fp16(model_name, manual_cleanup):
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, torch_dtype=torch.float16, export_compile_only=True, manual_cleanup=manual_cleanup
    )


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models)
def test_generate_fp16(model_name, manual_cleanup):
    check_seq2seq_pytorch_vs_kv_vs_ort_vs_ai100(model_name, torch_dtype=torch.float16, manual_cleanup=manual_cleanup)
