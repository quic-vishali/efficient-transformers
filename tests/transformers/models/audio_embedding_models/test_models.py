# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import torch
import pytest

from .check_audio_embedding_models import check_ctc_pytorch_vs_kv_vs_ort_vs_ai100

audio_embedding_models = {"facebook/wav2vec2-base-960h": "hf-internal-testing/tiny-random-Wav2Vec2ForCTC"}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models = audio_embedding_models.values()
else:
    test_models = audio_embedding_models.keys()


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile(model_name, manual_cleanup):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup=manual_cleanup, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name, manual_cleanup):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup=manual_cleanup)


# FP16 export
@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_fp16(model_name, manual_cleanup):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name, torch_dtype=torch.float16, manual_cleanup=manual_cleanup, export_compile_only=True)


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models)
def test_generate_fp16(model_name, manual_cleanup):

    check_ctc_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup=manual_cleanup, torch_dtype=torch.float16)
