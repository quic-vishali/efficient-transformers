# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import torch

from QEfficient.utils.test_utils import ModelConfig

from .check_embedding_models import check_embed_pytorch_vs_ort_vs_ai100

embedding_models = {
    # BertModel
    "BAAI/bge-base-en-v1.5": "hf-internal-testing/tiny-bert",
    "BAAI/bge-large-en-v1.5": "hf-internal-testing/tiny-bert",
    "BAAI/bge-small-en-v1.5": "hf-internal-testing/tiny-bert",
    "intfloat/e5-large-v2": "hf-internal-testing/tiny-bert",

    # MPNetForMaskedLM
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": "hf-internal-testing/tiny-random-MPNetForMaskedLM",

    # NomicBertModel
    "Nomic-ai/Nomic-embed-text-v1.5": "Nomic-ai/Nomic-embed-text-v1.5",

    # RobertaModel
    "ibm-granite/granite-embedding-30m-english": "hf-internal-testing/tiny-random-RobertaModel",
    "ibm-granite/granite-embedding-125m-english": "hf-internal-testing/tiny-random-RobertaModel",

    # XLMRobertaForSequenceClassification
    "BAAI/bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",

    # XLMRobertaModel
    "ibm-granite/granite-embedding-107m-multilingual": "optimum-intel-internal-testing/tiny-random-xlm-roberta",
    "ibm-granite/granite-embedding-278m-multilingual": "optimum-intel-internal-testing/tiny-random-xlm-roberta",
    "intfloat/multilingual-e5-large": "optimum-intel-internal-testing/tiny-random-xlm-roberta",

    # JinaBertForMaskedLM
    "jinaai/jina-embeddings-v2-base-code": "jinaai/jina-embeddings-v2-small-en",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    embed_test_models = set(embedding_models.values())
else:
    embed_test_models = set(embedding_models.keys())

poolings = ["mean", "max", "cls", "avg", None] 


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
        export_compile_only=True,
        manual_cleanup=manual_cleanup,
        # use_onnx_subfunctions=True,
    )

@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
        manual_cleanup=manual_cleanup,
        # use_onnx_subfunctions=True,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_multiseqlen(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=[32, 20],
        pooling=pooling,
        num_devices=1,
        export_compile_only=True,
        manual_cleanup=manual_cleanup,
    )

@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_multiseqlen(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=[32, 20],
        pooling=pooling,
        num_devices=1,
        manual_cleanup=manual_cleanup,
    )


# Fp16 export + FP16 compile, pooling, devices=1
@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_fp16(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
        torch_dtype=torch.float16,
        export_compile_only=True,
        num_devices=1,
        manual_cleanup=manual_cleanup,
    )

@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_fp16(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=32,
        pooling=pooling,
        torch_dtype=torch.float16,
        num_devices=1,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_fp16_multiseqlen(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=[32, 20],
        pooling=pooling,
        torch_dtype=torch.float16,
        export_compile_only=True,
        num_devices=1,
        manual_cleanup=manual_cleanup,
    )

@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", embed_test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_generate_fp16_multiseqlen(model_name, pooling, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    check_embed_pytorch_vs_ort_vs_ai100(
        model_name=model_name,
        seq_len=[32, 20],
        pooling=pooling,
        torch_dtype=torch.float16,
        num_devices=1,
        manual_cleanup=manual_cleanup,
    )
