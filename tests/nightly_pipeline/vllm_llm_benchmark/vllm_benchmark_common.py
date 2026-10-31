#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Shared CSV-driven orchestration for vLLM QAIC serving benchmarks.

This module holds every piece of logic that does not vary by model domain
(LLM, embedding, audio): CSV parsing, command builders, server/client
subprocess management, benchmark-result parsing, and output-row writing.

Each domain-specific runner script (vllm_llm_benchmark.py,
vllm_embedding_benchmark.py, vllm_audio_benchmark.py) imports
`build_arg_parser` and `run_benchmarks` from here and supplies only its own
`LATEST_MODELS` curated set and CLI description.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .pytorch_references import reference_for
    from .qaic_device_cleanup import (
        DeviceCleanupError,
        extract_device_groups,
        release_owned_busy_processes,
    )
    from .validation_utils import cosine_similarity, normalize_transcription, text_cosine_fallback
else:
    from pytorch_references import reference_for
    from qaic_device_cleanup import (
        DeviceCleanupError,
        extract_device_groups,
        release_owned_busy_processes,
    )
    from validation_utils import cosine_similarity, normalize_transcription, text_cosine_fallback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


API_SERVER_READY_MARKERS = (
    "Application startup complete",
    "Uvicorn running on",
)
DISAGG_READY_MARKERS = ("Press Ctl-C once to shutdown all services.",)

SKIPPED_MODELS = {
    "zai-org/GLM-4.5",
    "hpcai-tech/grok-1",
}
CUSTOM_DATASET_NAMES = frozenset({"custom", "custom_image", "custom_audio"})
LARGE_HF_MODEL_CACHE_LIMIT_BYTES = 50_000_000_000

BENCH_BLOCK_RE = re.compile(r"=+\s*Serving Benchmark Result\s*=+(.*?)=+", re.DOTALL)
BENCH_KV_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 /().\-]*?):\s+([-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)

BENCH_LABEL_TO_KEY = {
    "Successful requests": "successful_requests",
    "Failed requests": "failed_requests",
    "Maximum request concurrency": "max_request_concurrency",
    "Benchmark duration (s)": "benchmark_duration_s",
    "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Request throughput (req/s)": "request_throughput_req_s",
    "Output token throughput (tok/s)": "output_token_throughput_tok_s",
    "Peak output token throughput (tok/s)": "peak_output_token_throughput_tok_s",
    "Peak concurrent requests": "peak_concurrent_requests",
    "Total token throughput (tok/s)": "total_token_throughput_tok_s",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
}

INT_RESULT_KEYS = {
    "successful_requests",
    "failed_requests",
    "total_input_tokens",
    "total_generated_tokens",
}

OUTPUT_FIELDS = [
    "run_timestamp_utc",
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
    "config_name",
    "status",
    "error",
    "model",
    "server_type",
    "config_summary",
    "mode_type",
    "device_group",
    "encode_device_group",
    "prefill_device_group",
    "decode_device_group",
    "BS",
    "VBS",
    "PBS",
    "DBS",
    "PL",
    "GL",
    "CL",
    "num_prompts",
    "max_concurrency",
    "failed_requests",
    "benchmark_duration_s",
    "export_compile_time_s",
    "prefill_mdp_export_compile_time_s",
    "prefill_export_compile_time_s",
    "decode_export_compile_time_s",
    "encode_export_compile_time_s",
    "qpc_count",
    "qpc_size_mb",
    "qpc_sizes_mb",
    "request_throughput_req_s",
    "output_token_throughput_tok_s",
    "total_token_throughput_tok_s",
    "mean_TTFT_ms",
    "P99_TTFT_ms",
    "mean_TPOT_ms",
    "P99_TPOT_ms",
    "mean_ITL_ms",
    "P99_ITL_ms",
    "decode_TPS",
    "vllm_exec_time_s",
    "server_command",
    "client_command",
    "server_log",
    "client_log",
    "response_file",
    "response_text_file",
    "reference_file",
    "reference_text_file",
    "validation_actual_text",
    "validation_reference_text",
    "golden_output_file",
    "golden_status",
    "validation_method",
    "validation_status",
    "validation_score",
    "validation_error",
    "pooling_method",
]

PUBLISHED_FIELDS = [
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
    "model",
    "model_category",
    "config_name",
    "config_summary",
    "status",
    "error",
    "validation_method",
    "validation_status",
    "validation_score",
    "validation_error",
    "validation_actual_text",
    "validation_reference_text",
    "golden_status",
    "golden_output_file",
    "qpc_count",
    "previous_qpc_size_mb",
    "qpc_size_mb",
    "qpc_sizes_mb",
    "previous_export_compile_time_s",
    "export_compile_time_s",
    "previous_prefill_mdp_export_compile_time_s",
    "prefill_mdp_export_compile_time_s",
    "previous_prefill_export_compile_time_s",
    "prefill_export_compile_time_s",
    "previous_decode_export_compile_time_s",
    "decode_export_compile_time_s",
    "previous_encode_export_compile_time_s",
    "encode_export_compile_time_s",
    "mean_ttft_s",
    "mean_tpot_s",
    "mean_itl_s",
    "comparison_build_number",
    "decode_TPS",
    "request_throughput_req_s",
    "vllm_exec_time_s",
    "server_command",
    "client_command",
]


class BenchmarkError(RuntimeError):
    """Raised when a benchmark row cannot be completed."""


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_path: Path
    ready_event: threading.Event
    stream_thread: threading.Thread
    started_at: float
    ready_at: float | None = None

    @property
    def ready_time_s(self) -> float | None:
        if self.ready_at is None:
            return None
        return self.ready_at - self.started_at

    @property
    def returncode(self) -> int | None:
        return self.process.poll()


def value(row: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        raw = row.get(key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return default


def parse_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {raw!r}")


def optional_int(row: dict, *keys: str, default: int | None = None) -> int | None:
    raw = value(row, *keys)
    if raw == "":
        return default
    return int(raw)


def add_value_arg(cmd: list[str], flag: str, row: dict, *keys: str) -> None:
    raw = value(row, *keys)
    if raw != "":
        cmd.extend([flag, raw])


def add_bool_arg(cmd: list[str], flag: str, row: dict, key: str) -> None:
    if parse_bool(row.get(key), default=False):
        cmd.append(flag)


def parse_json_cell(raw: str, field_name: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name}: invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{field_name}: expected a JSON object")
    return parsed


def compact_json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


def parse_device_group(raw: str) -> list[int]:
    values = []
    for item in re.split(r"[, ]+", raw.strip()):
        if item:
            values.append(int(item))
    return values


def device_count(raw: str) -> int:
    raw = (raw or "").strip()
    if not raw:
        return 0
    total = 0
    for token in raw.split():
        if ":" in token:
            parts = [p for p in token.split(":") if p]
            if len(parts) == 2:
                total += max(0, int(parts[1]) - int(parts[0]))
            elif len(parts) == 3:
                total += len(range(int(parts[0]), int(parts[1]), int(parts[2])))
            continue
        total += len([p for p in token.split(",") if p.strip()])
    return total


def split_space_tokens(raw: str) -> list[str]:
    return [token for token in (raw or "").split() if token]


def iter_ports(raw: str) -> Iterable[int]:
    for token in split_space_tokens(raw):
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pieces = [int(p) for p in part.split(":") if p]
                if len(pieces) == 2:
                    yield from range(pieces[0], pieces[1] + 1)
                elif len(pieces) == 3:
                    yield from range(pieces[0], pieces[1] + 1, pieces[2])
                continue
            yield int(part)


def sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return cleaned.strip("_").lower() or "unknown"


def command_to_shell_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _normalize_csv_row(row: dict, fields: list[str]) -> dict:
    normalized = dict(row)
    for field in fields:
        raw = normalized.get(field)
        if raw is None or str(raw).strip() == "":
            normalized[field] = "N/A"
    return normalized


def build_api_additional_config(row: dict) -> str:
    raw = value(row, "additional_config", "additional_config_json")
    if raw:
        # Validate early, but preserve a compact form in the command.
        return compact_json(parse_json_cell(raw, "additional_config"))

    additional_config: dict = {}
    device_group = value(row, "device_group")
    if device_group:
        additional_config["device_group"] = parse_device_group(device_group)

    override_config = parse_json_cell(
        value(row, "override_qaic_config"),
        "override_qaic_config",
    )

    for csv_key, json_key in [
        ("use_onnx_subfunctions", "use_onnx_subfunctions"),
        ("aic_enable_depth_first", "aic_enable_depth_first"),
        ("user_tiled", "user_tiled"),
        ("allow_mxint8_mdp_io", "allow_mxint8_mdp_io"),
        ("mxfp6_matmul", "mxfp6_matmul"),
        ("mxint8_kv_cache", "mxint8_kv_cache"),
    ]:
        raw_value = value(row, csv_key)
        if raw_value:
            override_config[json_key] = parse_bool(raw_value)

    qaic_config = parse_json_cell(value(row, "qaic_config"), "qaic_config")
    if value(row, "enable_blocking"):
        qaic_config["enable_blocking"] = parse_bool(row.get("enable_blocking"))
    if value(row, "blocking_mode"):
        qaic_config["blocking_mode"] = value(row, "blocking_mode")
    if value(row, "skip_kv"):
        qaic_config["skip_kv"] = parse_bool(row.get("skip_kv"))
    if qaic_config:
        override_config["qaic_config"] = qaic_config

    if override_config:
        additional_config["override_qaic_config"] = override_config

    if value(row, "ccl_enabled"):
        additional_config["ccl_enabled"] = parse_bool(row.get("ccl_enabled"))
    add_if_present(additional_config, "comp_ctx_lengths_prefill", row)
    add_if_present(additional_config, "comp_ctx_lengths_decode", row)

    return compact_json(additional_config) if additional_config else ""


def add_if_present(
    target: dict,
    key: str,
    row: dict,
    source_key: str | None = None,
) -> None:
    raw = value(row, source_key or key)
    if raw:
        target[key] = raw


def build_api_server_command(row: dict, args) -> list[str]:
    python_bin = value(row, "python_bin", default=args.python_bin)
    cmd = [python_bin, "-m", "vllm.entrypoints.openai.api_server"]

    add_bool_arg(cmd, "--enable-chunked-prefill", row, "enable_chunked_prefill")
    add_bool_arg(cmd, "--no-enable-chunked-prefill", row, "no_enable_chunked_prefill")

    add_value_arg(cmd, "--port", row, "port", "server_port")
    add_value_arg(cmd, "--max-model-len", row, "max_model_len", "CL")
    add_value_arg(cmd, "--kv-cache-dtype", row, "kv_cache_dtype")
    add_value_arg(cmd, "--host", row, "host")
    add_value_arg(
        cmd,
        "--long-prefill-token-threshold",
        row,
        "long_prefill_token_threshold",
    )
    add_value_arg(cmd, "--quantization", row, "quantization")
    add_value_arg(cmd, "--max-num-seqs", row, "max_num_seqs", "BS")
    add_value_arg(cmd, "--dtype", row, "dtype")
    add_value_arg(cmd, "--generation-config", row, "generation_config")
    add_value_arg(cmd, "--runner", row, "runner")
    add_value_arg(cmd, "--seed", row, "seed")
    add_value_arg(cmd, "--model", row, "model")
    add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")

    additional_config = build_api_additional_config(row)
    if additional_config:
        cmd.extend(["--additional-config", additional_config])

    pooler_config = value(row, "pooler_config")
    if pooler_config:
        parse_json_cell(pooler_config, "pooler_config")
        cmd.extend(["--pooler-config", compact_json(json.loads(pooler_config))])

    extra_args = value(row, "server_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def add_many_token_arg(cmd: list[str], flag: str, row: dict, key: str) -> None:
    raw = value(row, key)
    if raw:
        cmd.append(flag)
        cmd.extend(split_space_tokens(raw))


def build_disagg_server_command(row: dict, args) -> list[str]:
    python_bin = value(row, "python_bin", default=args.python_bin)
    cmd = [python_bin, "-m", "qaic_disagg"]

    add_value_arg(cmd, "--host", row, "host")
    add_value_arg(cmd, "--port", row, "port", "server_port")
    add_many_token_arg(cmd, "--encode-port", row, "encode_port")
    add_many_token_arg(cmd, "--encode-device-group", row, "encode_device_group")
    add_many_token_arg(cmd, "--decode-port", row, "decode_port")
    add_many_token_arg(cmd, "--decode-device-group", row, "decode_device_group")
    add_many_token_arg(cmd, "--prefill-port", row, "prefill_port")
    add_many_token_arg(cmd, "--prefill-device-group", row, "prefill_device_group")
    add_value_arg(cmd, "--model", row, "model")
    add_value_arg(cmd, "--encode-max-num-seqs", row, "encode_max_num_seqs", "VBS")
    add_value_arg(cmd, "--prefill-max-num-seqs", row, "prefill_max_num_seqs", "PBS")
    add_value_arg(cmd, "--decode-max-num-seqs", row, "decode_max_num_seqs", "DBS")
    add_value_arg(
        cmd,
        "--decode-long-prefill-token-threshold",
        row,
        "decode_long_prefill_token_threshold",
    )
    add_value_arg(
        cmd,
        "--prefill-long-prefill-token-threshold",
        row,
        "prefill_long_prefill_token_threshold",
    )
    add_value_arg(
        cmd,
        "--prefill-max-num-batched-tokens",
        row,
        "prefill_max_num_batched_tokens",
    )
    add_value_arg(cmd, "--max-model-len", row, "max_model_len", "CL")

    enable_blocking = parse_bool(value(row, "enable_blocking", default="true"))
    enable_ccl = parse_bool(value(row, "enable_ccl", default="true"))

    encode_override = value(row, "encode_override_qaic_config")
    if encode_override:
        parse_json_cell(encode_override, "encode_override_qaic_config")
        config = json.loads(encode_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--encode-override-qaic-config",
                compact_json(config),
            ]
        )

    prefill_override = value(row, "prefill_override_qaic_config")
    if prefill_override:
        parse_json_cell(prefill_override, "prefill_override_qaic_config")
        config = json.loads(prefill_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--prefill-override-qaic-config",
                compact_json(config),
            ]
        )

    decode_override = value(row, "decode_override_qaic_config")
    if decode_override:
        parse_json_cell(decode_override, "decode_override_qaic_config")
        config = json.loads(decode_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--decode-override-qaic-config",
                compact_json(config),
            ]
        )

    verbosity = optional_int(row, "verbosity", default=0) or 0
    if verbosity > 0:
        cmd.append("-" + ("v" * verbosity))

    add_value_arg(cmd, "--dtype", row, "dtype")
    add_value_arg(cmd, "--kv-cache-dtype", row, "kv_cache_dtype")
    add_value_arg(cmd, "--quantization", row, "quantization")
    add_value_arg(cmd, "--tool-call-parser", row, "tool_call_parser")
    add_bool_arg(cmd, "--enable-auto-tool-choice", row, "enable_auto_tool_choice")
    add_bool_arg(cmd, "--enable-log-outputs", row, "enable_log_outputs")
    add_value_arg(cmd, "--reasoning-parser", row, "reasoning_parser")
    # qaic_disagg accepts the vLLM generation config spelling, but current
    # production commands use --generation_config. Preserve that convention.
    add_value_arg(cmd, "--generation_config", row, "generation_config")
    # VLM production commands use the hyphenated vLLM spelling instead; kept as a
    # distinct column so it never collides with the underscore LLM-PD convention above.
    add_value_arg(cmd, "--generation-config", row, "vlm_generation_config")
    add_bool_arg(cmd, "--enable-log-requests", row, "enable_log_requests")
    add_value_arg(
        cmd,
        "--chat-template-content-format",
        row,
        "chat_template_content_format",
    )
    add_value_arg(cmd, "--chat-template", row, "chat_template")
    add_value_arg(cmd, "--kv-store-size", row, "kv_store_size")
    add_value_arg(cmd, "--kv-handOff-port", row, "kv_handoff_port")
    add_value_arg(cmd, "--scheduling-policy", row, "scheduling_policy")
    add_bool_arg(cmd, "--no-enable-prefix-caching", row, "no_enable_prefix_caching")
    add_value_arg(
        cmd,
        "--prefill-pipeline-parallel-size",
        row,
        "prefill_pipeline_parallel_size",
    )
    add_value_arg(cmd, "--proxy-worker", row, "proxy_worker")

    limit_mm_per_prompt = value(row, "limit_mm_per_prompt")
    if limit_mm_per_prompt:
        parse_json_cell(limit_mm_per_prompt, "limit_mm_per_prompt")
        cmd.extend(
            [
                "--limit-mm-per-prompt",
                compact_json(json.loads(limit_mm_per_prompt)),
            ]
        )

    extra_args = value(row, "server_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def resolve_client_type(row: dict, args) -> str:
    requested = value(row, "client_type", default="auto").lower()
    if requested == "auto":
        has_qserve_script = resolve_qserve_script(row, args, require_exists=False)
        return "qserve" if has_qserve_script else "vllm_bench"
    # If qserve is explicitly requested but not available, fallback to vllm_bench
    if requested == "qserve":
        has_qserve_script = resolve_qserve_script(row, args, require_exists=False)
        if not has_qserve_script:
            return "vllm_bench"
    return requested


def resolve_qserve_script(row: dict, args, require_exists: bool = True) -> str:
    candidates = [
        value(row, "client_script"),
        args.qserve_benchmark_script,
        os.environ.get("QSERVE_BENCHMARK_SCRIPT", ""),
    ]
    # Check dedicated QSERVE_DIR first (Jenkins clones qserve to a separate directory)
    qserve_dir = os.environ.get("QSERVE_DIR", "")
    if qserve_dir:
        candidates.append(str(Path(qserve_dir) / "qserve/benchmarks/benchmark_serving.py"))
    # Also check inside vllm-qaic (legacy fallback)
    vllm_dir = Path(args.vllm_qaic_dir) if args.vllm_qaic_dir else None
    if vllm_dir:
        candidates.append(str(vllm_dir / "qserve/qserve/benchmarks/benchmark_serving.py"))
    candidates.append("/home/qraniumtest/qserve/qserve/benchmarks/benchmark_serving.py")

    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).is_file():
                return candidate
        except OSError:
            continue

    explicit = value(row, "client_script") or args.qserve_benchmark_script
    if explicit and not require_exists:
        return explicit
    if require_exists:
        raise BenchmarkError(
            "qserve benchmark_serving.py was requested but not found. "
            "Set --qserve-benchmark-script or use client_type=vllm_bench."
        )
    return ""


def resolve_dataset_path(raw_path: str, base_dir: str = ".") -> str:
    """Resolve a config dataset path before passing it to an external client."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        Path(base_dir).expanduser() / path,
        Path.cwd() / path,
        Path(__file__).resolve().parents[3] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # Keep a deterministic absolute path in the error reported by vLLM.
    return str(candidates[0].resolve())


def add_hf_random_dataset_args(cmd: list[str], row: dict, base_dir: str = ".") -> None:
    tokenizer = value(row, "tokenizer")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    random_range_ratio = value(row, "random_range_ratio")
    if random_range_ratio:
        cmd.extend(["--random-range-ratio", random_range_ratio])
    dataset_path = value(row, "dataset_path")
    if dataset_path:
        cmd.extend(["--dataset-path", resolve_dataset_path(dataset_path, base_dir)])
    hf_subset = value(row, "hf_subset")
    if hf_subset:
        cmd.extend(["--hf-subset", hf_subset])
    hf_split = value(row, "hf_split")
    if hf_split:
        cmd.extend(["--hf-split", hf_split])
    hf_output_len = value(row, "hf_output_len")
    if hf_output_len:
        cmd.extend(["--hf-output-len", hf_output_len])


def add_custom_dataset_path_arg(cmd: list[str], row: dict, base_dir: str = ".") -> None:
    dataset_path = value(row, "dataset_path")
    if dataset_path:
        cmd.extend(["--dataset-path", resolve_dataset_path(dataset_path, base_dir)])


def add_random_mm_args(cmd: list[str], row: dict) -> None:
    base_items = value(row, "random_mm_base_items_per_request")
    if base_items:
        cmd.extend(["--random-mm-base-items-per-request", base_items])
    range_ratio = value(row, "random_mm_num_mm_items_range_ratio")
    if range_ratio:
        cmd.extend(["--random-mm-num-mm-items-range-ratio", range_ratio])
    limit_mm_per_prompt = value(row, "random_mm_limit_mm_per_prompt")
    if limit_mm_per_prompt:
        parse_json_cell(limit_mm_per_prompt, "random_mm_limit_mm_per_prompt")
        cmd.extend(["--random-mm-limit-mm-per-prompt", compact_json(json.loads(limit_mm_per_prompt))])
    # --random-mm-bucket-config uses a Python-tuple-keyed literal (e.g.
    # '{(720,1080,1):1.0}'), which is not valid JSON - pass it through verbatim.
    bucket_config = value(row, "random_mm_bucket_config")
    if bucket_config:
        cmd.extend(["--random-mm-bucket-config", bucket_config])


def build_client_command(row: dict, args, result_dir: Path | None = None) -> list[str]:
    client_type = resolve_client_type(row, args)
    host = value(row, "host", default="127.0.0.1")
    port = value(row, "port", "server_port", default="8080")
    endpoint = value(row, "endpoint")
    backend = value(row, "backend", default="vllm")
    dataset_name = value(row, "dataset_name", "dataset-name", default="random")
    num_prompts = value(row, "num_prompts", "num-prompts", default="2")
    max_concurrency = value(row, "max_concurrency", "max-concurrency")
    seed = value(row, "seed")
    request_rate = value(row, "request_rate", "request-rate")
    input_len = value(row, "input_len", "random_input_len", "PL")
    output_len = value(row, "output_len", "random_output_len", "GL")
    is_custom_image = dataset_name == "custom_image"
    is_custom_dataset = dataset_name in CUSTOM_DATASET_NAMES

    if client_type == "qserve":
        script_path = resolve_qserve_script(row, args)
        python_bin = value(
            row,
            "client_python_bin",
            "python_bin",
            default=args.python_bin,
        )
        cmd = [
            python_bin,
            script_path,
            "--backend",
            backend,
            "--dataset-name",
            dataset_name,
            "--model",
            value(row, "model"),
            "--num-prompts",
            num_prompts,
            "--host",
            host,
            "--port",
            port,
        ]
        if max_concurrency:
            cmd.extend(["--max-concurrency", max_concurrency])
        if input_len:
            cmd.extend(["--random-input-len", input_len])
        if output_len:
            cmd.extend(["--random-output-len", output_len])
        if endpoint:
            cmd.extend(["--endpoint", endpoint])
        if seed:
            cmd.extend(["--seed", seed])
        if is_custom_image:
            add_custom_dataset_path_arg(cmd, row, args.base_dir)
        else:
            add_hf_random_dataset_args(cmd, row, args.base_dir)
        add_bool_arg(cmd, "--ignore-eos", row, "ignore_eos")
        add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")
        add_bool_arg(cmd, "--save-result", row, "save_result")
    elif client_type == "vllm_bench":
        base_url = value(row, "base_url") or f"http://{host}:{port}"
        # ``vllm-embed`` is the legacy backend name used by the QServe
        # benchmark script.  ``vllm bench serve`` uses the upstream backend
        # name ``openai-embeddings`` for the same /v1/embeddings API.
        vllm_bench_backend = "openai-embeddings" if backend == "vllm-embed" else backend
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            vllm_bench_backend,
        ]
        if endpoint:
            cmd.extend(["--endpoint", endpoint])
        cmd.extend(
            [
                "--base-url",
                base_url,
                "--model",
                value(row, "model"),
                "--dataset-name",
                dataset_name,
                "--num-prompts",
                num_prompts,
            ]
        )
        if request_rate:
            cmd.extend(["--request-rate", request_rate])
        if max_concurrency:
            cmd.extend(["--max-concurrency", max_concurrency])

        if not is_custom_dataset:
            length_style = value(row, "length_arg_style", default="random").lower()
            if length_style == "general":
                if input_len:
                    cmd.extend(["--input-len", input_len])
                if output_len:
                    cmd.extend(["--output-len", output_len])
            else:
                if input_len:
                    cmd.extend(["--random-input-len", input_len])
                if output_len:
                    cmd.extend(["--random-output-len", output_len])

        if is_custom_dataset:
            custom_output_len = value(row, "custom_output_len", "output_len", "GL")
            if custom_output_len:
                cmd.extend(["--custom-output-len", custom_output_len])

        if seed:
            cmd.extend(["--seed", seed])
        if is_custom_image:
            add_custom_dataset_path_arg(cmd, row, args.base_dir)
        else:
            add_hf_random_dataset_args(cmd, row, args.base_dir)
            add_random_mm_args(cmd, row)
        if is_custom_image:
            # Encode local image files in the client request. The benchmark
            # client does not expose the server-only allowed-local-media-path
            # option, and remote/local file URLs would otherwise be rejected.
            cmd.append("--custom-ensure-client-side-data")
        # Custom embedding inputs are plain text, not chat messages. vLLM's
        # CustomDataset otherwise calls tokenizer.apply_chat_template(), which
        # fails for embedding tokenizers such as BGE that have no template.
        if endpoint == "/v1/embeddings":
            cmd.append("--skip-chat-template")
        else:
            add_bool_arg(cmd, "--skip-chat-template", row, "skip_chat_template")
        add_bool_arg(cmd, "--ignore-eos", row, "ignore_eos")
        add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")
        temperature = value(row, "temperature")
        if temperature:
            cmd.extend(["--temperature", temperature])
        elif _validation_method(row).lower() == "pytorch" and endpoint not in {
            "/v1/embeddings",
            "/v1/audio/transcriptions",
        }:
            # Match pytorch_llm_reference.py, which uses deterministic greedy
            # decoding (do_sample=False) for parity validation.
            cmd.extend(["--temperature", "0.0"])
        if _capture_enabled(row):
            cmd.extend(["--save-result", "--save-detailed"])
        else:
            add_bool_arg(cmd, "--save-result", row, "save_result")
            add_bool_arg(cmd, "--save-detailed", row, "save_detailed")
        if _capture_enabled(row):
            if result_dir is None:
                raise BenchmarkError("result_dir is required when save_detailed=true")
            cmd.extend(["--result-dir", str(result_dir), "--result-filename", "benchmark_result.json"])
    else:
        raise BenchmarkError(f"unsupported client_type: {client_type}")

    extra_args = value(row, "client_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def build_server_command(row: dict, args) -> list[str]:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type == "api_server":
        return build_api_server_command(row, args)
    if server_type in {"qaic_disagg", "disagg"}:
        return build_disagg_server_command(row, args)
    raise BenchmarkError(f"unsupported server_type: {server_type}")


def server_ready_markers(row: dict) -> tuple[str, ...]:
    explicit = value(row, "server_ready_markers")
    if explicit:
        return tuple(marker.strip() for marker in explicit.split("|") if marker.strip())
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type == "api_server":
        return API_SERVER_READY_MARKERS
    return DISAGG_READY_MARKERS


def server_ports(row: dict) -> list[int]:
    ports = []
    for key in ("port", "server_port", "prefill_port", "decode_port", "encode_port"):
        raw = value(row, key)
        if not raw:
            continue
        try:
            ports.extend(iter_ports(raw))
        except ValueError:
            continue
    return sorted(set(ports))


def kill_ports(ports: Iterable[int]) -> None:
    for port in ports:
        pids = []

        # Method 1: Try lsof (preferred, fast)
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via lsof), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 2: Try ss (common in containers)
        try:
            result = subprocess.run(
                ["ss", "-tlnp"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line:
                    match = re.search(r"pid=(\d+)", line)
                    if match:
                        pids.append(match.group(1))
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via ss), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 3: Try fuser
        try:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [p.strip() for p in result.stdout.split() if p.strip()]
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via fuser), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 4: Try /proc/net/tcp (works in Linux containers without tools)
        try:
            with open("/proc/net/tcp", "r") as f:
                lines = f.readlines()
            port_hex = f"{port:X}"
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 10:
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port_hex}"):
                        inode = parts[9]
                        for proc_dir in Path("/proc").glob("*/fd/*"):
                            try:
                                if proc_dir.is_symlink():
                                    target = str(proc_dir.resolve())
                                    if f"socket:[{inode}]" in target:
                                        pid = proc_dir.parent.parent.name
                                        if pid.isdigit() and pid not in pids:
                                            pids.append(pid)
                            except (OSError, ValueError):
                                pass
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via /proc), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, OSError, ValueError):
            pass

        # Method 5: Try Python socket (universal, always available)
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                print(f"  Port {port}: in use but cannot determine PID (socket check)")
            else:
                print(f"  Port {port}: free (no processes)")
            continue
        except (OSError, ImportError):
            pass

        print(f"  Port {port}: unable to check (no tools available)")


def write_log_header(log_path: Path, command: list[str], title: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"{title}\n")
        log_file.write("=" * len(title) + "\n")
        timestamp = datetime.now(timezone.utc).isoformat()
        log_file.write(f"timestamp_utc: {timestamp}\n")
        log_file.write(f"command: {command_to_shell_string(command)}\n\n")


def launch_server(
    command: list[str],
    log_path: Path,
    markers: tuple[str, ...],
    cwd: str,
    timeout_s: int,
) -> ServerProcess:
    ports = server_ports_from_command(command)
    if ports:
        print(f"  Final port cleanup check before server launch for ports: {ports}")
        kill_ports(ports)
        time.sleep(0.5)  # Brief wait to ensure ports are released

    write_log_header(log_path, command, "Server Command")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,  # noqa: PLW1509
    )

    ready_event = threading.Event()
    started_at = time.monotonic()
    server = ServerProcess(
        process=process,
        log_path=log_path,
        ready_event=ready_event,
        stream_thread=threading.Thread(),
        started_at=started_at,
    )

    def stream_output() -> None:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                if not ready_event.is_set() and any(marker in line for marker in markers):
                    server.ready_at = time.monotonic()
                    ready_event.set()

    stream_thread = threading.Thread(target=stream_output, daemon=True)
    server.stream_thread = stream_thread
    stream_thread.start()

    while True:
        if ready_event.is_set():
            return server
        if process.poll() is not None:
            stream_thread.join(timeout=5)
            raise BenchmarkError(f"server exited before ready marker; returncode={process.returncode}; log={log_path}")
        if timeout_s > 0 and time.monotonic() - started_at > timeout_s:
            raise BenchmarkError(f"server did not become ready within {timeout_s}s; log={log_path}")
        time.sleep(1)


def server_ports_from_command(command: list[str]) -> list[int]:
    ports = []
    for index, token in enumerate(command):
        if token in {
            "--port",
            "--prefill-port",
            "--decode-port",
            "--encode-port",
            "--kv-handOff-port",
        }:
            cursor = index + 1
            while cursor < len(command) and not command[cursor].startswith("--"):
                ports.extend(iter_ports(command[cursor]))
                cursor += 1
    return sorted(set(ports))


def cleanup_busy_qaic_processes(server_command: list[str]) -> None:
    """Release matching stale QAIC servers before starting a benchmark row."""
    try:
        device_groups = extract_device_groups(server_command, os.environ)
        if not device_groups:
            return

        force_kill = parse_bool(os.environ.get("QAIC_FORCE_KILL"), default=False)
        print(f"  QAIC force kill enabled: {force_kill}")
        cleanup = release_owned_busy_processes(
            device_groups,
            terminate_timeout_s=30.0,
            force_kill=force_kill,
        )
    except (DeviceCleanupError, OSError, ValueError) as exc:
        print(f"  ERROR: QAIC device cleanup failed: {exc}; continuing inference")
        return

    print(
        "  QAIC device cleanup: "
        f"groups={device_groups}, busy_qids={cleanup.busy_qids}, "
        f"killed_process_groups={cleanup.killed_process_groups}"
    )
    print(
        "  QAIC cleanup details: "
        f"force_kill={force_kill}, ownership_scope={cleanup.ownership_scope}, "
        f"matched_processes={len(cleanup.matched_processes)}, "
        f"other_matching_processes={len(cleanup.other_matching_processes)}"
    )
    for process in cleanup.matched_processes:
        print(
            "  QAIC matched process: "
            f"pid={process.pid}, pgid={process.pgid}, "
            f"process_groups={process.matched_groups}, command={shlex.join(process.command)}"
        )
    for process in cleanup.other_matching_processes:
        print(
            "  QAIC ownership-filtered process: "
            f"pid={process.pid}, pgid={process.pgid}, "
            f"process_groups={process.matched_groups}, command={shlex.join(process.command)}"
        )
    if cleanup.busy_qids and not cleanup.killed_process_groups:
        if cleanup.other_matching_processes:
            reason = "matching process found but excluded by Jenkins/UID ownership checks"
        else:
            reason = (
                "no visible process exposed any requested busy QID in its command "
                "line or QAIC device-group environment variables"
            )
        print(
            "  ERROR: requested QAIC devices are busy and no matching server process was killed: "
            + ",".join(map(str, cleanup.busy_qids))
            + f" ({reason}); continuing inference"
        )


def terminate_server(server: ServerProcess, timeout_s: int) -> None:
    if server.process.poll() is not None:
        return
    print(f"  Terminating server pid {server.process.pid}")
    try:
        os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
        server.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  Server did not exit in {timeout_s}s; sending SIGKILL")
        os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
        server.process.wait(timeout=30)
    finally:
        server.stream_thread.join(timeout=5)


def run_client(command: list[str], log_path: Path, cwd: str) -> int:
    write_log_header(log_path, command, "Client Command")
    env = os.environ.copy()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        process.wait()
        log_file.write(f"\nreturncode: {process.returncode}\n")
    return process.returncode


def extract_server_start_time(log_path: Path) -> datetime | None:
    """Extract server start timestamp from log header."""
    if not log_path.exists():
        return None
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        for line in content.split("\n"):
            if line.startswith("timestamp_utc:"):
                timestamp_str = line.split(":", 1)[1].strip()
                return datetime.fromisoformat(timestamp_str)
    except (ValueError, IndexError):
        pass
    return None


def extract_compilation_times(log_path: Path, server_type: str) -> dict[str, float]:
    """Extract export+compile times for each stage from server log."""
    times = {}
    if not log_path.exists():
        return times

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")

        # Find server start time
        start_time = extract_server_start_time(log_path)
        if not start_time:
            return times

        # Infer timezone offset from first log entry
        tz_offset_seconds = 0
        for line in lines:
            time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", line)
            if time_match:
                month, day, hour, minute, second = map(int, time_match.groups())
                try:
                    # Parse the log time as if it were UTC
                    log_time_utc = start_time.replace(month=month, day=day, hour=hour, minute=minute, second=second)
                    # Calculate the offset
                    tz_offset_seconds = int((log_time_utc - start_time).total_seconds())
                    # Only accept reasonable offsets (-12 to +14 hours)
                    if -43200 <= tz_offset_seconds <= 50400:
                        break
                except ValueError:
                    pass

        # For disagg modes, find stage-specific timestamps
        if server_type in {"qaic_disagg", "disagg"}:
            # Find timestamps for each stage start
            stage_times = {}

            for i, line in enumerate(lines):
                # Find encode start
                if "Started compilation for encode qpc" in line and "encode_start" not in stage_times:
                    # Look for first timestamp after this line
                    for j in range(i, min(i + 50, len(lines))):
                        time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                        if time_match:
                            month, day, hour, minute, second = map(int, time_match.groups())
                            try:
                                log_time_utc = start_time.replace(
                                    month=month, day=day, hour=hour, minute=minute, second=second
                                )
                                stage_times["encode_start"] = log_time_utc - __import__("datetime").timedelta(
                                    seconds=tz_offset_seconds
                                )
                                break
                            except ValueError:
                                pass

                # Find prefill start
                if (
                    "Started compilation for PP" in line
                    and "TS1 prefill qpc" in line
                    and "prefill_start" not in stage_times
                ):
                    for j in range(i, min(i + 50, len(lines))):
                        time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                        if time_match:
                            month, day, hour, minute, second = map(int, time_match.groups())
                            try:
                                log_time_utc = start_time.replace(
                                    month=month, day=day, hour=hour, minute=minute, second=second
                                )
                                stage_times["prefill_start"] = log_time_utc - __import__("datetime").timedelta(
                                    seconds=tz_offset_seconds
                                )
                                break
                            except ValueError:
                                pass

                # Find decode start
                if "Started compilation for decode qpc" in line and "decode_start" not in stage_times:
                    for j in range(i, min(i + 50, len(lines))):
                        time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                        if time_match:
                            month, day, hour, minute, second = map(int, time_match.groups())
                            try:
                                log_time_utc = start_time.replace(
                                    month=month, day=day, hour=hour, minute=minute, second=second
                                )
                                stage_times["decode_start"] = log_time_utc - __import__("datetime").timedelta(
                                    seconds=tz_offset_seconds
                                )
                                break
                            except ValueError:
                                pass

                # Find ready time
                if "Press Ctl-C once to shutdown all services" in line and "ready" not in stage_times:
                    # Search backwards for last timestamp
                    for j in range(i, max(0, i - 100), -1):
                        time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                        if time_match:
                            month, day, hour, minute, second = map(int, time_match.groups())
                            try:
                                log_time_utc = start_time.replace(
                                    month=month, day=day, hour=hour, minute=minute, second=second
                                )
                                stage_times["ready"] = log_time_utc - __import__("datetime").timedelta(
                                    seconds=tz_offset_seconds
                                )
                                break
                            except ValueError:
                                pass

            # Calculate durations for each stage
            if "ready" in stage_times:
                if "encode_start" in stage_times and "prefill_start" in stage_times:
                    encode_duration = (stage_times["prefill_start"] - stage_times["encode_start"]).total_seconds()
                    if 0 < encode_duration < 7200:
                        times["encode_export_compile_time_s"] = encode_duration

                if "prefill_start" in stage_times and "decode_start" in stage_times:
                    prefill_duration = (stage_times["decode_start"] - stage_times["prefill_start"]).total_seconds()
                    if 0 < prefill_duration < 7200:
                        times["prefill_export_compile_time_s"] = prefill_duration

                if "decode_start" in stage_times:
                    decode_duration = (stage_times["ready"] - stage_times["decode_start"]).total_seconds()
                    if 0 < decode_duration < 7200:
                        times["decode_export_compile_time_s"] = decode_duration

                # For LLM disagg_pd, also calculate prefill_mdp time
                if "Started compilation for prefill qpc" in content and "prefill_mdp" not in times:
                    # Find prefill MDP start time
                    for i, line in enumerate(lines):
                        if "Started compilation for prefill qpc" in line and "PP" not in line:
                            for j in range(i, min(i + 50, len(lines))):
                                time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                                if time_match:
                                    month, day, hour, minute, second = map(int, time_match.groups())
                                    try:
                                        log_time_utc = start_time.replace(
                                            month=month, day=day, hour=hour, minute=minute, second=second
                                        )
                                        prefill_mdp_start = log_time_utc - __import__("datetime").timedelta(
                                            seconds=tz_offset_seconds
                                        )
                                        if "prefill_start" in stage_times:
                                            mdp_duration = (
                                                stage_times["prefill_start"] - prefill_mdp_start
                                            ).total_seconds()
                                            if 0 < mdp_duration < 7200:
                                                times["prefill_mdp_export_compile_time_s"] = mdp_duration
                                        break
                                    except ValueError:
                                        pass
                            break

                return times

        # For non-disagg modes, find single export+compile time
        for i in range(len(lines) - 1, -1, -1):
            is_api_ready = "Application startup complete" in lines[i]

            if is_api_ready:
                # Search backwards from here for a timestamp
                for j in range(i, max(0, i - 100), -1):
                    time_match = re.search(r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", lines[j])
                    if time_match:
                        month, day, hour, minute, second = map(int, time_match.groups())
                        try:
                            log_time_utc = start_time.replace(
                                month=month, day=day, hour=hour, minute=minute, second=second
                            )
                            ready_dt = log_time_utc - __import__("datetime").timedelta(seconds=tz_offset_seconds)
                            total_time = (ready_dt - start_time).total_seconds()

                            if 0 < total_time < 7200:
                                times["export_compile_time_s"] = total_time
                                return times
                        except ValueError:
                            pass
    except OSError:
        pass

    return times


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SERVER_EXCEPTION_RE = re.compile(r"\b(?P<exception>[A-Za-z_][\w.]*(?:Error|Exception)):\s+(?P<message>.+)$")
_SERVER_ERROR_RE = re.compile(
    r"\bERROR\b(?:\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})?\s+(?P<message>.+)$",
    re.IGNORECASE,
)
_SERVER_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)?(?:\([^)]*\)\s*)?"
    r"(?:(?:INFO|WARNING|ERROR|DEBUG)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+)?",
    re.IGNORECASE,
)
_QPC_PATH_MARKER_RE = re.compile(r"(?:\bUsing\s+)?\bqpc\s*:-\s*", re.IGNORECASE)
_QPC_PATH_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./+:-]+(?:\s*\n\s*[A-Za-z0-9_./+:-]+)*?/\s*q\s*p\s*c)"
    r"(?![A-Za-z0-9_./+:-])"
)
_LOG_TIME_PREFIX_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+")

NO_ERROR_MESSAGE_AVAILABLE = "No error message available; may be due to space issue. Check the logs."

_GENERIC_SERVER_ERRORS = (
    "calledprocesserror",
    "engine core initialization failed",
    "enginecore encountered an issue",
    "engine core",
    "output_handler failed",
    "failed to start",
    "fastapi.exceptions.httpexception",
    "request failed with",
    "internal server error",
    "error occurred in disagg proxy server",
    "occurred in disagg proxy server",
    "no error message available",
)
DEFAULT_COSINE_THRESHOLD = 0.98


def is_generic_server_error(error: object) -> bool:
    """Return whether *error* is an outer server/compiler wrapper.

    These messages are useful when no deeper diagnostic is available, but
    should be replaced when the server log contains the actual compiler or
    runtime cause.
    """
    normalized = str(error or "").strip().lower()
    return not normalized or any(marker in normalized for marker in _GENERIC_SERVER_ERRORS)


def extract_server_error(log_path: Path) -> str:
    """Extract the most useful failure message from a server log.

    Wrapper errors such as ``Engine core initialization failed`` are ignored
    when a more specific exception or compiler error is available earlier in
    the log. Informational messages containing words such as ``runtime error``
    are not treated as failures.
    """
    if not log_path.exists():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    # vLLM prefixes every line and frequently wraps one error over several
    # physical lines. Remove those prefixes first so messages such as
    # ``Error message: Unable to ensure`` + ``partial update ...`` can be
    # extracted as one logical error.
    cleaned_lines = [
        _SERVER_LOG_PREFIX_RE.sub("", _ANSI_ESCAPE_RE.sub("", raw_line)).strip() for raw_line in content.splitlines()
    ]
    cleaned_content = "\n".join(cleaned_lines)

    candidates: list[tuple[int, int, str]] = []

    def add_candidate(priority: int, index: int, message: str) -> None:
        message = " ".join(message.split()).strip(" -:")
        if not message or len(message) > 500:
            return
        message_lower = message.lower()
        if "resource_tracker" in message_lower:
            return
        if "enginedeaderror" in message_lower:
            nearby_lines = cleaned_lines[max(0, index - 20) : index + 1]
            if any("[shutdown]" in line.lower() or "shutdown" in line.lower() for line in nearby_lines):
                return
        if priority < 130 and any(error in message_lower for error in _GENERIC_SERVER_ERRORS):
            priority = min(priority, 20)
        if message_lower.startswith("runtimeerror: engine core"):
            priority = min(priority, 5)
        candidates.append((priority, index, message))

    # These patterns intentionally operate across newlines because compiler
    # diagnostics and Python exception messages are commonly wrapped.
    root_cause_patterns = (
        (
            120,
            re.compile(
                r"Error message:\s*(?P<message>.+?)(?=\n(?:Unable to Compile|"
                r"ERROR\b|Traceback\b|[A-Za-z_][\w.]*Error:)|\Z)",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            115,
            re.compile(
                r"QAIC_ERROR:\s*(?P<message>Network compilation failed:.*?)(?=\n"
                r"(?:Unable to Compile|ERROR\b|Traceback\b)|\Z)",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            110,
            re.compile(
                r"(?P<exception>(?:Attribute|Assertion|FileNotFound|Import|"
                r"Index|Key|ModuleNotFound|NotImplemented|OSError|Type|Value)Error):\s*"
                r"(?P<message>.+?)(?=\n(?:Traceback\b|INFO\b|WARNING\b|ERROR\b|"
                r"[A-Za-z_][\w.]*Error:|The above exception|Log file:|"
                r"Transforming\b|QEFF\b|Compiler\b|Gracefully\b|JobManager\b)|\Z)",
                re.DOTALL,
            ),
        ),
        (
            105,
            re.compile(
                r"(?P<message>Invalid option\s+[^\n]+)",
                re.IGNORECASE,
            ),
        ),
        (
            70,
            re.compile(
                r"(?P<exception>ValueError):\s*(?P<message>Compilation failed for\s+"
                r"[^\n]+)",
                re.IGNORECASE,
            ),
        ),
    )
    for priority, pattern in root_cause_patterns:
        for match in pattern.finditer(cleaned_content):
            if match.groupdict().get("exception"):
                add_candidate(
                    priority,
                    match.start(),
                    f"{match.group('exception')}: {match.group('message')}",
                )
            else:
                add_candidate(priority, match.start(), match.group("message"))

    for index, normalized in enumerate(cleaned_lines):
        if not normalized:
            continue

        exception_match = _SERVER_EXCEPTION_RE.search(normalized)
        if exception_match:
            message = f"{exception_match.group('exception')}: {exception_match.group('message')}"
            message_lower = message.lower()
            generic = any(error in message_lower for error in _GENERIC_SERVER_ERRORS)
            add_candidate(20 if generic else 80, index, message)
            continue

        error_match = _SERVER_ERROR_RE.search(normalized)
        if error_match:
            message = error_match.group("message").strip()
            message_lower = message.lower()
            if message and not message_lower.endswith("traceback (most recent call last):"):
                generic = any(error in message_lower for error in _GENERIC_SERVER_ERRORS)
                add_candidate(10 if generic else 60, index, f"Server error: {message}")

    if not candidates:
        return ""
    # For equally ranked errors, keep the first root cause rather than the
    # later wrapper exception raised while shutting the server down.
    priority, _, message = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
    return message


def _qpc_directory_size(path: Path) -> int:
    """Return the size of a QPC file or directory, ignoring unreadable entries."""
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0
    except OSError:
        return 0

    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def extract_qpc_sizes(log_path: Path, base_dir: str = ".") -> dict[str, object]:
    """Extract unique paths from ``Using qpc:-`` log entries and size them.

    Long paths can be wrapped across log lines. Timestamp-only prefixes added by
    some launchers are removed before joining the wrapped path components. Other
    log references such as ``qpc_path=`` and ``-aic-binary-dir=`` are ignored
    because they duplicate the ``Using qpc:-`` entry.
    """
    result: dict[str, object] = {
        # No QPC entry means this metric is not applicable to the row.  The
        # CSV normalizer converts this empty value to ``N/A``.
        "qpc_count": "",
        "qpc_size_mb": "",
        "qpc_sizes_mb": "",
    }
    if not log_path.exists():
        return result

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    content = _ANSI_ESCAPE_RE.sub("", content)
    # Some captured logs prefix every wrapped line with only a wall-clock time.
    # Removing that prefix lets the path regex join a wrapped path correctly.
    normalized_lines = [_LOG_TIME_PREFIX_RE.sub("", line) for line in content.splitlines()]
    content = "\n".join(normalized_lines)

    discovered: dict[str, Path] = {}
    for marker in _QPC_PATH_MARKER_RE.finditer(content):
        path_match = _QPC_PATH_RE.match(content, marker.end())
        if path_match is None:
            continue

        raw_path = re.sub(r"\s+", "", path_match.group("path"))
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(base_dir).expanduser() / path
        try:
            path = path.resolve(strict=False)
        except OSError:
            path = path.absolute()
        discovered[str(path)] = path

    if not discovered:
        return result

    sizes_bytes = [_qpc_directory_size(path) for path in discovered.values()]
    total_bytes = sum(sizes_bytes)
    result["qpc_count"] = len(discovered)
    result["qpc_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    result["qpc_sizes_mb"] = ";".join(str(round(size / (1024 * 1024), 2)) for size in sizes_bytes)
    return result


def _clear_directory_contents(path: Path, label: str) -> None:
    """Remove children of a runtime artifact directory without deleting the directory itself."""
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        resolved = path.expanduser().absolute()

    # Guard against a malformed env var causing a broad delete.
    if str(resolved) in {"/", "/root", "/home", "/tmp", "/efficient-transformers"}:
        print(f"  Skipping {label} cleanup for unsafe path: {resolved}")
        return
    if label == "TMPDIR" and not str(resolved).endswith("/vllm_llm_workspace/tmp"):
        print(f"  Skipping {label} cleanup for non-pipeline path: {resolved}")
        return
    if not resolved.exists():
        return
    if not resolved.is_dir():
        print(f"  Skipping {label} cleanup because path is not a directory: {resolved}")
        return

    removed = 0
    for child in resolved.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        except OSError as exc:
            print(f"  Warning: failed to remove {child}: {exc}")
    print(f"  Cleaned {label}: removed {removed} entries from {resolved}")


def _huggingface_hub_cache_dir() -> Path:
    """Resolve the Hugging Face Hub cache using standard environment overrides."""
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        configured_path = os.environ.get(variable, "").strip()
        if configured_path:
            expanded_path = os.path.expandvars(os.path.expanduser(configured_path))
            return Path(expanded_path)

    hf_home = os.environ.get("HF_HOME", "").strip()
    if not hf_home:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        cache_home = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
        hf_home = str(cache_home / "huggingface")

    expanded_path = os.path.expandvars(os.path.expanduser(str(Path(hf_home) / "hub")))
    return Path(expanded_path)


def _directory_file_size_bytes(directory: Path) -> int | None:
    """Count unique regular-file bytes without following Hub snapshot symlinks."""
    total_bytes = 0
    seen_files = set()
    pending_directories = [directory]
    try:
        while pending_directories:
            current = pending_directories.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending_directories.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        file_stat = entry.stat(follow_symlinks=False)
                        file_identity = (file_stat.st_dev, file_stat.st_ino)
                        if file_identity not in seen_files:
                            total_bytes += file_stat.st_size
                            seen_files.add(file_identity)
    except OSError as exc:
        print(f"  Warning: cannot measure Hugging Face model cache {directory}: {exc}; preserving it")
        return None
    return total_bytes


def cleanup_large_hf_model_caches() -> None:
    """Remove individual Hugging Face model caches larger than 50 GB."""
    cache_root = _huggingface_hub_cache_dir()
    try:
        resolved_root = cache_root.resolve(strict=False)
        if resolved_root in {Path("/"), Path("/home"), Path("/root"), Path("/tmp")}:
            print(f"  Skipping Hugging Face cache cleanup for unsafe path: {resolved_root}")
            return
        if not resolved_root.is_dir():
            return
        entries = list(os.scandir(resolved_root))
    except OSError as exc:
        print(f"  Warning: cannot inspect Hugging Face cache {cache_root}: {exc}; preserving it")
        return

    for entry in entries:
        if not entry.name.startswith("models--") or entry.is_symlink():
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            model_cache = Path(entry.path)
            resolved_model_cache = model_cache.resolve(strict=False)
            if resolved_model_cache.parent != resolved_root:
                print(f"  Skipping Hugging Face cache cleanup for unsafe path: {resolved_model_cache}")
                continue

            size_bytes = _directory_file_size_bytes(resolved_model_cache)
            if size_bytes is None or size_bytes <= LARGE_HF_MODEL_CACHE_LIMIT_BYTES:
                continue

            # Recheck the direct-child invariant immediately before removal.
            if model_cache.is_symlink() or model_cache.resolve(strict=False).parent != resolved_root:
                print(f"  Skipping Hugging Face cache cleanup for changed path: {model_cache}")
                continue
            size_gb = size_bytes / 1_000_000_000
            print(f"  Removing Hugging Face model cache ({size_gb:.2f} GB): {resolved_model_cache}")
            shutil.rmtree(resolved_model_cache)
        except OSError as exc:
            print(f"  Warning: failed to clean Hugging Face model cache {entry.path}: {exc}")


def cleanup_runtime_artifacts() -> None:
    """Free per-row compiled artifacts so large model rows do not accumulate on disk."""
    qeff_home = os.environ.get("QEFF_HOME", "").strip()
    if qeff_home:
        _clear_directory_contents(Path(qeff_home), "QEFF_HOME")
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir:
        _clear_directory_contents(Path(tmpdir), "TMPDIR")


def parse_benchmark_output(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    content = log_path.read_text(encoding="utf-8", errors="replace")
    runs = []
    for run_number, block_match in enumerate(BENCH_BLOCK_RE.finditer(content), 1):
        run = {"run_number": run_number}
        for kv_match in BENCH_KV_RE.finditer(block_match.group(1)):
            label = kv_match.group(1).strip()
            key = BENCH_LABEL_TO_KEY.get(label)
            if key is None:
                continue
            raw = kv_match.group(2)
            run[key] = int(float(raw)) if key in INT_RESULT_KEYS else float(raw)
        for ms_key, sec_key in [
            ("mean_ttft_ms", "ttft_s"),
            ("median_ttft_ms", "median_ttft_s"),
            ("p99_ttft_ms", "p99_ttft_s"),
            ("mean_tpot_ms", "tpot_s"),
            ("median_tpot_ms", "median_tpot_s"),
            ("p99_tpot_ms", "p99_tpot_s"),
            ("mean_itl_ms", "itl_s"),
            ("median_itl_ms", "median_itl_s"),
            ("p99_itl_ms", "p99_itl_s"),
        ]:
            run[sec_key] = run.get(ms_key) / 1000.0 if run.get(ms_key) is not None else None
        runs.append(run)
    return runs


def rounded(value_: object) -> object:
    if isinstance(value_, (float, int)) and not isinstance(value_, bool):
        return round(value_, 2)
    return value_ if value_ is not None else ""


def build_config_summary(row: dict) -> str:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type in {"qaic_disagg", "disagg"}:
        vbs = value(row, "encode_max_num_seqs", "VBS")
        vbs_part = f"VBS:{vbs} / " if vbs else ""
        return (
            f"{vbs_part}"
            f"PBS:{value(row, 'prefill_max_num_seqs', 'PBS')} / "
            f"DBS:{value(row, 'decode_max_num_seqs', 'DBS')} / "
            f"PL:{value(row, 'input_len', 'random_input_len', 'PL')} / "
            f"GL:{value(row, 'output_len', 'random_output_len', 'GL')} / "
            f"CL:{value(row, 'max_model_len', 'CL')}"
        )
    return (
        f"BS:{value(row, 'max_num_seqs', 'BS')} / "
        f"PL:{value(row, 'input_len', 'random_input_len', 'PL')} / "
        f"GL:{value(row, 'output_len', 'random_output_len', 'GL')} / "
        f"CL:{value(row, 'max_model_len', 'CL')}"
    )


def build_mode_type(row: dict) -> str:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type in {"qaic_disagg", "disagg"}:
        es = device_count(value(row, "encode_device_group"))
        pp = value(row, "prefill_pipeline_parallel_size") or device_count(value(row, "prefill_device_group"))
        ts = device_count(value(row, "decode_device_group"))
        parts = []
        if es:
            parts.append(f"ES{es}")
        if pp:
            parts.append(f"PP{pp}")
        if ts:
            parts.append(f"TS{ts}")
        return "+".join(parts)
    ts = device_count(value(row, "device_group"))
    return f"TS{ts}" if ts else ""


def result_status(client_returncode: int | None, parsed_runs: list[dict]) -> str:
    if client_returncode is None:
        return "server_failed"
    if client_returncode != 0:
        return "client_failed_with_results" if parsed_runs else "client_failed"
    successful_requests = sum(int(run.get("successful_requests", 0) or 0) for run in parsed_runs)
    if successful_requests == 0:
        return "no_results"
    return "success"


def get_qaic_sdk_version() -> str:
    try:
        result = subprocess.run(
            ["/opt/qti-aic/tools/qaic-version-util"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _validation_bool(row: dict, key: str) -> bool:
    return parse_bool(row.get(key), default=False)


def _validation_method(row: dict) -> str:
    configured = value(row, "validation_method")
    if configured:
        return configured
    endpoint = value(row, "endpoint").lower()
    dataset_name = value(row, "dataset_name").lower()
    if dataset_name in CUSTOM_DATASET_NAMES and endpoint in {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/audio/transcriptions",
    }:
        return "pytorch"
    return "none"


def _capture_enabled(row: dict) -> bool:
    return _validation_bool(row, "save_detailed") or (value(row, "dataset_name").lower() in CUSTOM_DATASET_NAMES)


def _golden_path(row: dict, base_dir: str) -> Path:
    configured = value(row, "golden_output_file")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(base_dir) / path
    return Path(base_dir) / "tests" / "nightly_pipeline" / "vllm_llm_benchmark" / "golden_outputs" / "goldens.json"


def _reference_max_new_tokens(row: dict) -> int:
    return optional_int(row, "custom_output_len", "GL", default=1) or 1


def _reference_generation_parameters(row: dict) -> dict:
    """Return only options that are actually passed to the PyTorch reference."""
    reference = reference_for(value(row, "endpoint"), value(row, "dataset_name"))
    parameters = {"trust_remote_code": _validation_bool(row, "trust_remote_code")}
    if reference.requires_max_new_tokens:
        parameters["max_new_tokens"] = _reference_max_new_tokens(row)
    if reference.script.name == "pytorch_llm_reference.py":
        parameters["skip_chat_template"] = _validation_bool(row, "skip_chat_template")
    if reference.output_key == "embeddings":
        parameters["pooling_method"] = value(row, "pooling_method", default="mean")
    return parameters


def _reference_asset_identity(raw_path: object, dataset: Path, base_dir: str, *, audio: bool = False) -> dict | None:
    if raw_path in (None, ""):
        return None
    asset = Path(str(raw_path))
    if asset.is_absolute():
        candidates = [asset]
    elif audio:
        candidates = [dataset.parent.parent.parent / asset, Path(base_dir) / asset]
    else:
        candidates = [Path(base_dir) / asset]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return {
                    "path": str(raw_path),
                    "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                }
        except OSError:
            continue
    return {"path": str(raw_path)}


def _reference_dataset_sha256(
    rows: list[dict], dataset: Path, base_dir: str, endpoint: str, dataset_name: str, fallback: bytes
) -> tuple[list[str], str]:
    """Hash the inputs consumed by the PyTorch script, not unused JSONL fields."""
    reference = reference_for(endpoint, dataset_name)
    if reference.script.name == "pytorch_audio_reference.py":
        inputs = [{"audio": _reference_asset_identity(row.get("audio"), dataset, base_dir, audio=True)} for row in rows]
        prompts: list[str] = []
    elif reference.script.name == "pytorch_vlm_reference.py":
        inputs = []
        prompts = []
        for row in rows:
            prompt = str(row.get("prompt", ""))
            image_files = row.get("image_files") or []
            image_path = image_files[0] if isinstance(image_files, list) and image_files else None
            inputs.append(
                {
                    "prompt": prompt,
                    "image": _reference_asset_identity(image_path, dataset, base_dir),
                }
            )
            prompts.append(prompt)
    else:
        prompts = [str(row.get("prompt", "")) for row in rows]
        inputs = [{"prompt": prompt} for prompt in prompts]

    try:
        encoded_inputs = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        encoded_inputs = fallback
    return prompts, hashlib.sha256(encoded_inputs).hexdigest()


def _normalize_reference_generation_parameters(identity: dict) -> dict:
    """Normalize current and legacy golden identities to reference inputs."""
    raw = identity.get("generation_parameters") or {}
    endpoint = str(identity.get("endpoint", ""))
    dataset_name = str(identity.get("dataset_name", ""))
    reference = reference_for(endpoint, dataset_name)
    parameters = {"trust_remote_code": parse_bool(raw.get("trust_remote_code"), default=False)}

    if reference.requires_max_new_tokens:
        if raw.get("max_new_tokens") not in (None, ""):
            max_new_tokens = int(raw["max_new_tokens"])
        else:
            legacy_row = {
                "custom_output_len": raw.get("custom_output_len"),
                "GL": raw.get("GL"),
            }
            max_new_tokens = optional_int(legacy_row, "custom_output_len", "GL", default=1) or 1
        parameters["max_new_tokens"] = max_new_tokens

    if reference.script.name == "pytorch_llm_reference.py":
        parameters["skip_chat_template"] = parse_bool(raw.get("skip_chat_template"), default=False)
    if reference.output_key == "embeddings":
        parameters["pooling_method"] = str(raw.get("pooling_method") or "mean")
    return parameters


def _golden_identity(row: dict, base_dir: str) -> dict:
    dataset = Path(value(row, "dataset_path"))
    if not dataset.is_absolute():
        dataset = Path(base_dir) / dataset
    try:
        dataset_bytes = dataset.read_bytes()
    except OSError:
        dataset_bytes = str(dataset).encode("utf-8")
    dataset_rows = []
    try:
        for line in dataset_bytes.decode("utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError("JSONL records must be objects")
                dataset_rows.append(item)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        dataset_rows = []
        prompts = []
        reference_dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    else:
        prompts, reference_dataset_sha256 = _reference_dataset_sha256(
            dataset_rows,
            dataset,
            base_dir,
            value(row, "endpoint"),
            value(row, "dataset_name"),
            dataset_bytes,
        )
    return {
        "model": value(row, "model"),
        "endpoint": value(row, "endpoint"),
        "dataset_name": value(row, "dataset_name"),
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "reference_inputs_sha256": reference_dataset_sha256,
        "prompts": prompts,
        "generation_parameters": _reference_generation_parameters(row),
    }


def _golden_key(row: dict, base_dir: str) -> tuple[str, str, str, str, dict]:
    identity = _golden_identity(row, base_dir)
    key_identity = dict(identity)
    # Keep the raw file digest for diagnostics and legacy lookup, but key on
    # the canonical PyTorch inputs so unused JSONL fields do not fork goldens.
    key_identity.pop("dataset_sha256", None)
    encoded = json.dumps(key_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    endpoint = value(row, "endpoint").lower()
    family = "image_text_to_text" if "image" in value(row, "dataset_name").lower() else "causal_lm"
    if endpoint == "/v1/embeddings":
        family = "embedding"
    elif endpoint == "/v1/audio/transcriptions":
        family = "audio"
    model = value(row, "model")
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    config_fp = hashlib.sha256(
        json.dumps(
            {"model": model, "generation_parameters": identity["generation_parameters"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:8]
    generation_len = identity["generation_parameters"].get("max_new_tokens", 0)
    variant = f"{model_slug}_gl{generation_len}_{config_fp}_{digest}"
    return family, model, variant, digest, identity


def _write_golden_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(store, output, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _golden_identity_matches(candidate: object, expected: dict) -> bool:
    """Match a stored golden, including entries written before key migration."""
    if not isinstance(candidate, dict):
        return False
    candidate_identity = candidate.get("identity")
    if not isinstance(candidate_identity, dict):
        return False

    candidate_reference_digest = candidate_identity.get("reference_inputs_sha256")
    expected_reference_digest = expected.get("reference_inputs_sha256")
    if candidate_reference_digest is not None and expected_reference_digest is not None:
        if candidate_reference_digest != expected_reference_digest:
            return False
    else:
        if candidate_reference_digest is None:
            reference = reference_for(str(expected.get("endpoint", "")), str(expected.get("dataset_name", "")))
            if reference.script.name in {"pytorch_vlm_reference.py", "pytorch_audio_reference.py"}:
                # Legacy multimodal entries fingerprinted only the JSONL, not
                # the referenced media. Their asset contents cannot be
                # verified, so regenerate instead of reusing a stale golden.
                return False
        if candidate_identity.get("dataset_sha256") != expected.get("dataset_sha256"):
            # Older text stores have only the raw JSONL digest. Reuse them
            # when the source file is unchanged; new entries use the
            # reference-input digest.
            return False

    def normalized(identity: dict) -> dict:
        result = dict(identity)
        result.pop("config", None)
        result.pop("dataset_sha256", None)
        result.pop("reference_inputs_sha256", None)
        result["generation_parameters"] = _normalize_reference_generation_parameters(identity)
        return result

    return normalized(candidate_identity) == normalized(expected)


def _find_golden(store: object, family: str, model: str, variant: str, identity: dict) -> dict | None:
    if not isinstance(store, dict):
        return None
    family_models = store.get(family, {})
    if isinstance(family_models, dict):
        model_entries = family_models.get(model, {})
        if isinstance(model_entries, dict):
            exact = model_entries.get(variant)
            if isinstance(exact, dict) and _reference_values(exact):
                return exact
            for candidate in model_entries.values():
                if _golden_identity_matches(candidate, identity) and _reference_values(candidate):
                    return candidate

    # Support the previous entries-based format during migration.
    entries = store.get("entries", {})
    if isinstance(entries, dict):
        exact = entries.get(variant)
        if isinstance(exact, dict) and _reference_values(exact):
            return exact
        for candidate in entries.values():
            if _golden_identity_matches(candidate, identity) and _reference_values(candidate):
                return candidate

    if _reference_values(store):
        return store
    return None


def _reference_values(record: object) -> object | None:
    if not isinstance(record, dict):
        return None
    for key in ("generated_texts", "embeddings"):
        values = record.get(key)
        if values:
            return values
    return None


def _response_values(response: object, kind: str) -> object | None:
    """Extract saved vLLM values across text, embedding, and audio formats."""
    if not isinstance(response, dict):
        return None
    keys = ("embeddings", "data") if kind == "embeddings" else ("generated_texts", "transcriptions", "texts")
    for key in keys:
        values = response.get(key)
        if values:
            if kind == "embeddings" and key == "data" and isinstance(values, list):
                return [item.get("embedding") if isinstance(item, dict) else item for item in values]
            return values
    outputs = response.get("outputs")
    if isinstance(outputs, list) and outputs:
        if kind == "embeddings":
            return [item.get("embedding", item) if isinstance(item, dict) else item for item in outputs]
        return [
            item.get("text", item.get("generated_text", item)) if isinstance(item, dict) else item for item in outputs
        ]
    return None


def _capture_embedding_response(row: dict, log_dir: Path, base_dir: str) -> dict:
    """Capture embedding vectors because vLLM bench saves metrics only."""
    dataset = Path(value(row, "dataset_path"))
    if not dataset.is_absolute():
        dataset = Path(base_dir) / dataset
    prompts = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(str(json.loads(line)["prompt"]))
    limit = optional_int(row, "num_prompts", default=len(prompts)) or len(prompts)
    payload = {
        "model": value(row, "model"),
        "input": prompts[:limit],
    }
    host = value(row, "host", default="127.0.0.1")
    port = value(row, "port", "server_port", default="8080")
    endpoint = value(row, "endpoint", default="/v1/embeddings")
    url = f"http://{host}:{port}{endpoint}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        captured = json.loads(response.read().decode("utf-8"))
    capture_file = log_dir / "embedding_response.json"
    capture_file.write_text(json.dumps(captured, indent=2) + "\n", encoding="utf-8")
    return captured


def _expected_value(row: dict, base_dir: str) -> object:
    raw = value(row, "expected_output")
    expected_file = value(row, "expected_output_file")
    if expected_file:
        path = Path(expected_file)
        if not path.is_absolute():
            path = Path(base_dir) / path
        raw = path.read_text(encoding="utf-8")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _run_reference(row: dict, log_dir: Path, base_dir: str) -> tuple[Path, object, str]:
    golden_file = _golden_path(row, base_dir)
    family, model, variant, _digest, identity = _golden_key(row, base_dir)
    reference = reference_for(value(row, "endpoint"), value(row, "dataset_name"))
    reference_kind = reference.output_key
    if golden_file.exists():
        store = json.loads(golden_file.read_text(encoding="utf-8"))
        golden = _find_golden(store, family, model, variant, identity)
        if golden is not None:
            values = _reference_values(golden)
            if reference_kind == "generated_texts":
                values = [str(item) for item in values]
            return golden_file, values, "golden"

    dataset = Path(value(row, "dataset_path"))
    if not dataset.is_absolute():
        dataset = Path(base_dir) / dataset
    max_new_tokens = _reference_max_new_tokens(row)
    # Keep the temporary HF/PyTorch result outside the run logs. Only the
    # golden reference is persisted for later comparisons.
    with tempfile.TemporaryDirectory(prefix="pytorch_llm_reference_") as temp_dir:
        reference_file = Path(temp_dir) / "reference.json"
        command = [
            sys.executable,
            str(reference.script),
            "--model",
            value(row, "model"),
            "--dataset",
            str(dataset),
            "--output",
            str(reference_file),
        ]
        if reference.requires_max_new_tokens:
            command.extend(["--max-new-tokens", str(max_new_tokens)])
        if reference.output_key == "embeddings":
            command.extend(["--pooling-method", value(row, "pooling_method", default="mean")])
        if _validation_bool(row, "trust_remote_code"):
            command.append("--trust-remote-code")
        if reference.script.name == "pytorch_llm_reference.py" and _validation_bool(row, "skip_chat_template"):
            command.append("--skip-chat-template")
        completed = subprocess.run(
            command,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=int(value(row, "pytorch_timeout_s", default="1800")),
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ValueError(f"PyTorch reference failed ({completed.returncode}): {detail[-2000:]}")
        reference = json.loads(reference_file.read_text(encoding="utf-8"))
        generated = reference.get(reference_kind, [])
        if not generated:
            raise ValueError(f"PyTorch reference did not produce {reference_kind}")
        store: dict = {}
        if golden_file.exists():
            existing = json.loads(golden_file.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                store = existing
        store.setdefault(family, {})
        store[family].setdefault(model, {})
        store[family][model][variant] = {
            "identity": identity,
            "model": reference.get("model", value(row, "model")),
            "prompts": reference.get("prompts", []),
            reference_kind: generated,
            "is_encoder_decoder": reference.get("is_encoder_decoder", False),
            "device": "cpu",
            "dtype": reference.get("dtype", "torch.float32"),
            "config_fp": variant.split("_")[-2],
            "gen_len": max_new_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _write_golden_store(golden_file, store)
    if reference_kind == "generated_texts":
        generated = [str(item) for item in generated]
    return golden_file, generated, "created"


def _remove_reference_log_artifacts(log_dir: Path) -> None:
    """Remove temporary PyTorch text artifacts, preserving vLLM JSON output."""
    for artifact in (
        log_dir / "response.txt",
        log_dir / "pytorch_response.txt",
        log_dir / "pytorch_llm_reference.json",
    ):
        try:
            artifact.unlink()
        except FileNotFoundError:
            pass


def _format_validation_text(value: object) -> str:
    """Serialize validation text without truncation for complete CSV records."""
    return json.dumps(value, ensure_ascii=False)


def validate_saved_benchmark_output(row: dict, log_dir: Path, base_dir: str) -> dict:
    """Validate the exact responses saved by vllm bench --save-detailed."""
    result = {
        "response_file": "",
        "response_text_file": "",
        "reference_file": "",
        "reference_text_file": "",
        "validation_method": _validation_method(row),
        "validation_status": "not_run",
        "validation_score": "",
        "validation_error": "",
        "validation_actual_text": "",
        "validation_reference_text": "",
        "golden_output_file": "",
        "golden_status": "",
    }
    if not _capture_enabled(row):
        return result

    response_file = log_dir / "benchmark_result.json"
    result["response_file"] = str(response_file)
    try:
        response = json.loads(response_file.read_text(encoding="utf-8"))
        method = result["validation_method"].lower()
        endpoint = value(row, "endpoint").lower()
        response_kind = "embeddings" if endpoint == "/v1/embeddings" else "generated_texts"
        generated = _response_values(response, response_kind)
        if response_kind == "embeddings" and not generated:
            # vLLM bench records embedding metrics but omits vector payloads
            # from benchmark_result.json. Query the still-running server once
            # to capture the exact vectors used for golden validation.
            response = _capture_embedding_response(row, log_dir, base_dir)
            generated = _response_values(response, response_kind)
        if not generated:
            raise ValueError(f"benchmark_result.json does not contain {response_kind}; use --save-detailed")
        result["validation_actual_text"] = _format_validation_text(generated)
        expected = _expected_value(row, base_dir)
        if method in {"", "none", "off"}:
            result["validation_status"] = "not_run"
        elif method == "pytorch":
            reference_file, reference_generated, golden_status = _run_reference(row, log_dir, base_dir)
            result["reference_file"] = str(reference_file)
            result["golden_output_file"] = str(reference_file)
            result["golden_status"] = golden_status
            result["validation_reference_text"] = _format_validation_text(reference_generated)
            actual = generated if response_kind == "embeddings" else [str(item) for item in generated]
            # Whether the golden was reused or created above, the QAIC result
            # must be checked against the PyTorch reference on this run.
            if response_kind == "embeddings":
                if (
                    not isinstance(actual, list)
                    or not isinstance(reference_generated, list)
                    or len(actual) != len(reference_generated)
                ):
                    matched = False
                    score = 0.0
                else:
                    scores = [cosine_similarity(left, right) for left, right in zip(actual, reference_generated)]
                    score = min(scores) if scores else 0.0
                    matched = score >= float(value(row, "cosine_threshold", default=str(DEFAULT_COSINE_THRESHOLD)))
                result["validation_score"] = str(round(score, 6))
                if not matched:
                    result["validation_error"] = "cosine similarity failed"
            else:
                # Text validation is exact first. If formatting or harmless
                # generation differences remain, use a fixed sentence encoder
                # as a semantic fallback rather than comparing model hidden
                # states, which are not a stable sentence-embedding space.
                if endpoint == "/v1/audio/transcriptions":
                    actual_for_match = [normalize_transcription(item) for item in actual]
                    reference_for_match = [normalize_transcription(item) for item in reference_generated]
                else:
                    actual_for_match = actual
                    reference_for_match = reference_generated

                matched = actual_for_match == reference_for_match
                if matched:
                    result["validation_score"] = "1.0"
                else:
                    score = text_cosine_fallback(actual, reference_generated)
                    threshold = float(value(row, "text_cosine_threshold", "cosine_threshold", default="0.98"))
                    matched = score >= threshold
                    result["validation_score"] = str(round(score, 6))
                    result["validation_error"] = (
                        "string match failed; cosine similarity passed"
                        if matched
                        else (
                            "string match failed; cosine similarity failed"
                            f"; vLLM={_format_validation_text(actual)}"
                            f"; PyTorch={_format_validation_text(reference_generated)}"
                        )
                    )
            result["validation_status"] = "passed" if matched else "failed"
        elif method in {"string", "contains", "exact", "regex"}:
            if expected is None:
                raise ValueError("expected_output or expected_output_file is required for validation")
            actual = "\n".join(str(item) for item in generated)
            expected_text = str(expected)
            result["validation_reference_text"] = _format_validation_text(expected_text)
            match_mode = value(row, "string_match_mode", default=method).lower()
            if match_mode == "regex":
                matched = re.search(expected_text, actual, flags=re.DOTALL) is not None
            elif match_mode == "exact":
                matched = actual.strip() == expected_text.strip()
            else:
                matched = expected_text in actual
            result["validation_status"] = "passed" if matched else "failed"
            result["validation_score"] = "1.0" if matched else "0.0"
        elif method == "cosine":
            if expected is None:
                raise ValueError("expected_output or expected_output_file is required for validation")
            result["validation_reference_text"] = _format_validation_text(expected)
            score = cosine_similarity(generated[0], expected)
            threshold = float(value(row, "cosine_threshold", default=str(DEFAULT_COSINE_THRESHOLD)))
            result["validation_score"] = str(round(score, 6))
            result["validation_status"] = "passed" if score >= threshold else "failed"
        else:
            raise ValueError(f"unsupported validation_method: {method}")
        _remove_reference_log_artifacts(log_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        result["validation_status"] = "error"
        result["validation_error"] = str(exc)
        # Preserve benchmark_result.json, including generated_texts, so the
        # complete vLLM response remains available for inspection.
        try:
            _remove_reference_log_artifacts(log_dir)
        except OSError:
            pass
    return result


def make_output_row(
    row: dict,
    config_name: str,
    status: str,
    error: str,
    run: dict,
    server_log: Path,
    client_log: Path,
    base_dir: str = ".",
    vllm_exec_time_s: float | str = "",
    server_cmd: list[str] | None = None,
    client_cmd: list[str] | None = None,
    validation: dict | None = None,
) -> dict:
    model = value(row, "model")
    max_concurrency = value(row, "max_concurrency", "max-concurrency")
    decode_bs = optional_int(row, "decode_max_num_seqs", "DBS", default=None)
    if decode_bs is None:
        decode_bs = optional_int(row, "max_num_seqs", "BS", default=None)
    mean_tpot_ms = run.get("mean_tpot_ms")
    decode_tps = ""
    if mean_tpot_ms and decode_bs:
        decode_tps = round((1000.0 / mean_tpot_ms) * decode_bs, 2)

    # Extract compilation times from server log
    server_type = value(row, "server_type", default="api_server").lower()
    timing_data = extract_compilation_times(server_log, server_type)
    qpc_data = extract_qpc_sizes(server_log, base_dir)

    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_name,
        "status": status,
        "error": error,
        "model": model,
        "server_type": server_type,
        "config_summary": build_config_summary(row),
        "mode_type": build_mode_type(row),
        "device_group": value(row, "device_group"),
        "encode_device_group": value(row, "encode_device_group"),
        "prefill_device_group": value(row, "prefill_device_group"),
        "decode_device_group": value(row, "decode_device_group"),
        "BS": value(row, "max_num_seqs", "BS"),
        "VBS": value(row, "encode_max_num_seqs", "VBS"),
        "PBS": value(row, "prefill_max_num_seqs", "PBS"),
        "DBS": value(row, "decode_max_num_seqs", "DBS"),
        "PL": value(row, "input_len", "random_input_len", "PL"),
        "GL": value(row, "output_len", "random_output_len", "GL"),
        "CL": value(row, "max_model_len", "CL"),
        "num_prompts": value(row, "num_prompts", "num-prompts"),
        "max_concurrency": max_concurrency,
        "failed_requests": run.get("failed_requests", ""),
        "benchmark_duration_s": rounded(run.get("benchmark_duration_s", "")),
        "export_compile_time_s": rounded(timing_data.get("export_compile_time_s", "")),
        "prefill_mdp_export_compile_time_s": rounded(timing_data.get("prefill_mdp_export_compile_time_s", "")),
        "prefill_export_compile_time_s": rounded(timing_data.get("prefill_export_compile_time_s", "")),
        "decode_export_compile_time_s": rounded(timing_data.get("decode_export_compile_time_s", "")),
        "encode_export_compile_time_s": rounded(timing_data.get("encode_export_compile_time_s", "")),
        "qpc_count": qpc_data["qpc_count"],
        "qpc_size_mb": qpc_data["qpc_size_mb"],
        "qpc_sizes_mb": qpc_data["qpc_sizes_mb"],
        "request_throughput_req_s": rounded(run.get("request_throughput_req_s", "")),
        "output_token_throughput_tok_s": rounded(run.get("output_token_throughput_tok_s", "")),
        "total_token_throughput_tok_s": rounded(run.get("total_token_throughput_tok_s", "")),
        "mean_TTFT_ms": rounded(run.get("mean_ttft_ms", "")),
        "P99_TTFT_ms": rounded(run.get("p99_ttft_ms", "")),
        "mean_TPOT_ms": rounded(run.get("mean_tpot_ms", "")),
        "P99_TPOT_ms": rounded(run.get("p99_tpot_ms", "")),
        "mean_ITL_ms": rounded(run.get("mean_itl_ms", "")),
        "P99_ITL_ms": rounded(run.get("p99_itl_ms", "")),
        "decode_TPS": decode_tps,
        "vllm_exec_time_s": rounded(vllm_exec_time_s),
        "vllm_qaic_branch": os.environ.get("VLLM_QAIC_BRANCH", ""),
        "qaic_disagg_branch": os.environ.get("QAIC_DISAGG_BRANCH", ""),
        "qserve_branch": os.environ.get("QSERVE_BRANCH", ""),
        "qeff_branch": os.environ.get("QEFF_BRANCH", ""),
        "qaic_sdk_version": get_qaic_sdk_version(),
        "server_command": command_to_shell_string(server_cmd) if server_cmd else "",
        "client_command": command_to_shell_string(client_cmd) if client_cmd else "",
        "server_log": str(server_log),
        "client_log": str(client_log),
        "response_file": (validation or {}).get("response_file", ""),
        "response_text_file": (validation or {}).get("response_text_file", ""),
        "reference_file": (validation or {}).get("reference_file", ""),
        "reference_text_file": (validation or {}).get("reference_text_file", ""),
        "validation_actual_text": (validation or {}).get("validation_actual_text", ""),
        "validation_reference_text": (validation or {}).get("validation_reference_text", ""),
        "golden_output_file": (validation or {}).get("golden_output_file", ""),
        "golden_status": (validation or {}).get("golden_status", ""),
        "validation_method": (validation or {}).get("validation_method", "none"),
        "validation_status": (validation or {}).get("validation_status", "not_run"),
        "validation_score": (validation or {}).get("validation_score", ""),
        "validation_error": (validation or {}).get("validation_error", ""),
        "pooling_method": value(row, "pooling_method"),
    }


def append_output_rows(output_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    rewrite = False
    if output_csv.exists():
        try:
            with output_csv.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                rewrite = reader.fieldnames != OUTPUT_FIELDS
        except (OSError, csv.Error):
            rewrite = True

    output_exists = output_csv.exists()
    existing_rows = [_normalize_csv_row(row, OUTPUT_FIELDS) for row in existing_rows]
    rows = [_normalize_csv_row(row, OUTPUT_FIELDS) for row in rows]
    with output_csv.open("w" if rewrite else "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if rewrite or not output_exists:
            writer.writeheader()
        if rewrite:
            writer.writerows(existing_rows)
        writer.writerows(rows)


def generate_published_csv(input_csv: Path, output_csv: Path) -> None:
    """Generate a simplified published CSV with only key fields for team distribution."""
    if not input_csv.exists():
        return
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    # Convert milliseconds to seconds for latency metrics and add model_category
    for row in rows:
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = str(round(float(row["mean_TTFT_ms"]) / 1000, 4))
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = str(round(float(row["mean_TPOT_ms"]) / 1000, 4))
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = str(round(float(row["mean_ITL_ms"]) / 1000, 4))

        # Determine model category based on config_name
        config_name = row.get("config_name", "").lower()
        if "embedding" in config_name:
            row["model_category"] = "Embedding"
        elif "audio" in config_name:
            row["model_category"] = "Audio"
        elif "vlm" in config_name:
            row["model_category"] = "VLM"
        else:
            row["model_category"] = "LLM"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_csv_row(row, PUBLISHED_FIELDS) for row in rows)


def merge_published_csvs(results_dir: Path, output_csv: Path) -> None:
    """Merge all published CSVs in results_dir into a single consolidated published CSV."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return

    all_rows = []
    published_csvs = sorted(results_dir.glob("*_results_published.csv"))

    for pub_csv in published_csvs:
        if not pub_csv.exists():
            continue
        with pub_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            all_rows.extend(rows)

    if not all_rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_csv_row(row, PUBLISHED_FIELDS) for row in all_rows)


def read_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for index, row in enumerate(csv.DictReader(f), 1):
            normalized = {str(k).strip(): (v or "").strip() for k, v in row.items()}
            normalized["_data_row"] = str(index)
            if parse_bool(normalized.get("enabled"), default=True):
                rows.append(normalized)
        return rows


def parse_rows_spec(row_spec: str, total_rows: int) -> list[int]:
    row_spec = (row_spec or "ALL").strip()
    if row_spec.upper() == "ALL":
        return list(range(1, total_rows + 1))
    selected = []
    for chunk in row_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start, end = [int(part.strip()) for part in chunk.split(":", 1)]
            if start > end:
                raise ValueError(f"invalid row range {chunk!r}: start > end")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(chunk))
    invalid = [row for row in selected if row < 1 or row > total_rows]
    if invalid:
        raise ValueError(f"row(s) out of range for {total_rows} data rows: {invalid}")
    return selected


def run_one(row: dict, args, config_name: str, output_csv: Path) -> bool:
    row_number = int(row["_data_row"])
    model = value(row, "model")

    # Determine effective_config_name based on config_name
    if config_name == "audio":
        # Audio: always 'audio'
        effective_config_name = "audio"
    elif config_name == "embedding":
        # Embedding: use pooling_method
        effective_config_name = value(row, "pooling_method", default="unknown")
    elif config_name == "vlm" or config_name.startswith("vlm_"):
        # VLM: keep the run variant in the effective name.  Without this,
        # vlm_weight_free rows are indistinguishable from regular VLM rows
        # after consolidation because both used only the mode tuple.
        variant = config_name.removeprefix("vlm_")
        specs = value(row, "specialization_mode", default="single")
        blocking = "blocking" if parse_bool(row.get("enable_blocking"), default=False) else "non_blocking"
        ccl = "ccl" if parse_bool(row.get("enable_ccl"), default=False) else "non_ccl"
        disagg_mode = value(row, "disagg_mode", default="single")
        mode_name = f"{specs}/{blocking}/{ccl}/{disagg_mode}"
        effective_config_name = f"{variant}/{mode_name}" if variant else mode_name
    else:
        # LLM: use config_name as-is (default, ccl, blocking, disagg_pd)
        effective_config_name = config_name

    run_id = f"{row_number:03d}_{sanitize_name(model)}_{sanitize_name(effective_config_name)}"
    log_dir = Path(args.results_dir) / "logs" / sanitize_name(config_name) / run_id
    server_log = log_dir / "server.log"
    client_log = log_dir / "client.log"

    server_cmd = build_server_command(row, args)
    client_cmd = build_client_command(row, args, result_dir=log_dir)

    # Clean up configured ports before starting the server. Port cleanup uses
    # the port-based termination behavior; QAIC device cleanup happens
    # inside the row-level try block below.
    ports_to_use = server_ports_from_command(server_cmd)
    if ports_to_use:
        print()
        print("=" * 80)
        print(f"Pre-flight port check for ports: {ports_to_use}")
        print("=" * 80)
        kill_ports(ports_to_use)
        print("  Ports cleaned and ready for use")
        time.sleep(1)  # Wait for ports to fully release

    print()
    print("=" * 80)
    print(f"Config: {effective_config_name}  Row: {row_number}  Model: {model}")
    print(f"Server: {command_to_shell_string(server_cmd)}")
    print(f"Client: {command_to_shell_string(client_cmd)}")
    print("=" * 80)

    if args.dry_run:
        write_log_header(server_log, server_cmd, "Server Command Dry Run")
        write_log_header(client_log, client_cmd, "Client Command Dry Run")
        append_output_rows(
            output_csv,
            [
                make_output_row(
                    row=row,
                    config_name=effective_config_name,
                    status="dry_run",
                    error="",
                    run={},
                    server_log=server_log,
                    client_log=client_log,
                    base_dir=args.base_dir,
                    vllm_exec_time_s="",
                    server_cmd=server_cmd,
                    client_cmd=client_cmd,
                    validation={"validation_method": _validation_method(row)},
                )
            ],
        )
        return True

    try:
        cleanup_large_hf_model_caches()
    except Exception as exc:
        print(f"  Warning: pre-run Hugging Face cache cleanup failed: {exc}")

    server = None
    client_returncode = None
    parsed_runs: list[dict] = []
    error = ""
    # Initialize before launch_server: Ctrl-C or an early startup failure can
    # enter finally before the normal result_status() assignment.
    status = "error"
    validation = {"validation_method": _validation_method(row), "validation_status": "not_run"}
    vllm_exec_time_s: float | str = ""
    execution_start_time: float | None = None

    # Keep an explicit artifact when server startup fails before the client is
    # launched.  run_client() overwrites this header on normal runs.
    write_log_header(client_log, client_cmd, "Client Command (not started)")

    try:
        cleanup_busy_qaic_processes(server_cmd)
        execution_start_time = time.monotonic()
        server = launch_server(
            server_cmd,
            server_log,
            server_ready_markers(row),
            cwd=args.base_dir,
            timeout_s=args.server_ready_timeout_s,
        )
        print(f"  Server ready after {server.ready_time_s:.2f}s")
        client_returncode = run_client(client_cmd, client_log, cwd=args.base_dir)
        execution_end_time = time.monotonic()
        vllm_exec_time_s = execution_end_time - execution_start_time
        parsed_runs = parse_benchmark_output(client_log)
        status = result_status(client_returncode, parsed_runs)
    except (BenchmarkError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        status = "error"
        error = str(exc)
        print(f"  ERROR: {error}")
    finally:
        try:
            if execution_start_time is not None and vllm_exec_time_s == "":
                vllm_exec_time_s = time.monotonic() - execution_start_time
            benchmark_result = log_dir / "benchmark_result.json"
            if _capture_enabled(row) and (status == "success" or benchmark_result.exists()):
                validation = validate_saved_benchmark_output(row, log_dir, args.base_dir)
                if validation["validation_status"] in {"failed", "error"} and status == "success":
                    status = "validation_failed"
                    error = validation["validation_error"] or "response validation failed"
            # Preserve the actionable validation mismatch. Server logs can contain
            # unrelated compiler warnings/errors from startup, which otherwise
            # overwrite the actual validation failure in the CSV.
            if status not in {"success", "dry_run", "validation_failed"}:
                server_error = extract_server_error(server_log)
                if server_error:
                    error = server_error
                elif not error or is_generic_server_error(error):
                    error = NO_ERROR_MESSAGE_AVAILABLE
            if server is not None:
                terminate_server(server, timeout_s=args.server_stop_timeout_s)

            output_runs = parsed_runs or [{}]
            append_output_rows(
                output_csv,
                [
                    make_output_row(
                        row=row,
                        config_name=effective_config_name,
                        status=status,
                        error=error,
                        run=run,
                        server_log=server_log,
                        client_log=client_log,
                        base_dir=args.base_dir,
                        vllm_exec_time_s=vllm_exec_time_s,
                        server_cmd=server_cmd,
                        client_cmd=client_cmd,
                        validation=validation,
                    )
                    for run in output_runs
                ],
            )
        finally:
            try:
                cleanup_runtime_artifacts()
            except Exception as exc:
                print(f"  Warning: runtime artifact cleanup failed: {exc}")
            try:
                cleanup_large_hf_model_caches()
            except Exception as exc:
                print(f"  Warning: post-run Hugging Face cache cleanup failed: {exc}")
    return status == "success"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-name",
        required=True,
        help="Logical config name for output rows.",
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--rows",
        default="ALL",
        help="Data row selector: ALL, 1, 1:4, or 1,3.",
    )
    parser.add_argument(
        "--latest-models-only",
        default="true",
        help="If true (default), restrict to the curated LATEST_MODELS list. SKIPPED_MODELS is always excluded.",
    )
    parser.add_argument(
        "--results-dir",
        default="vllm_llm_results",
        help="Log/result root.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[3]),
        help="Subprocess working directory.",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python binary for server/client modules.",
    )
    parser.add_argument(
        "--qserve-benchmark-script",
        default="",
        help="Path to qserve/qserve/benchmarks/benchmark_serving.py.",
    )
    parser.add_argument(
        "--vllm-qaic-dir",
        default=os.environ.get("VLLM_QAIC_DIR", ""),
        help="vllm-qaic checkout path used to auto-resolve qserve benchmark_serving.py.",
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=int,
        default=0,
        help="Seconds to wait for server readiness; 0 waits forever.",
    )
    parser.add_argument(
        "--server-stop-timeout-s",
        type=int,
        default=120,
        help="Seconds to wait for graceful server shutdown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands/log stubs only.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed row.",
    )
    parser.add_argument(
        "--disagg-mode",
        default="ALL",
        help="VLM only: filter rows by disagg_mode column (ED, PD, EPD, or ALL).",
    )
    parser.add_argument(
        "--specialization-mode",
        default="ALL",
        help="VLM only: filter rows by specialization_mode column (single, multi, or ALL).",
    )
    parser.add_argument(
        "--blocking-mode",
        default="ALL",
        help="VLM only: filter rows by blocking_mode column (blocking, non_blocking, or ALL).",
    )
    parser.add_argument(
        "--device-group-override",
        default="",
        help="Override device_group column value (e.g., '0,1,2,3'). Empty = use CSV values.",
    )
    parser.add_argument(
        "--device-group-encode-override",
        default="",
        help="VLM only: override encode_device_group column value. Empty = use CSV values.",
    )
    parser.add_argument(
        "--device-group-prefill-override",
        default="",
        help="VLM/disagg only: override prefill_device_group column value. Empty = use CSV values.",
    )
    parser.add_argument(
        "--device-group-decode-override",
        default="",
        help="VLM/disagg only: override decode_device_group column value. Empty = use CSV values.",
    )
    return parser


def row_matches_filter(row: dict, column: str, filter_value: str) -> bool:
    filter_value = (filter_value or "ALL").strip()
    if filter_value.upper() == "ALL":
        return True
    row_value = value(row, column)
    if not row_value:
        return True
    return row_value.lower() == filter_value.lower()


def update_device_group_in_additional_config(row: dict, device_group_override: str) -> None:
    """Update device_group in additional_config JSON if override is provided."""
    if not device_group_override:
        return
    additional_config_str = value(row, "additional_config")
    if not additional_config_str:
        return
    try:
        additional_config = parse_json_cell(additional_config_str, "additional_config")
        if isinstance(additional_config, dict):
            additional_config["device_group"] = parse_device_group(device_group_override)
            row["additional_config"] = compact_json(additional_config)
    except Exception:
        pass


def run_benchmarks(args, latest_models: set[str]) -> int:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    rows = read_csv_rows(input_csv)
    selected = set(parse_rows_spec(args.rows, len(rows)))
    latest_only = parse_bool(args.latest_models_only, default=True)
    disagg_mode = getattr(args, "disagg_mode", "ALL")
    specialization_mode = getattr(args, "specialization_mode", "ALL")
    blocking_mode = getattr(args, "blocking_mode", "ALL")
    device_group_override = getattr(args, "device_group_override", "")
    device_group_encode_override = getattr(args, "device_group_encode_override", "")
    device_group_prefill_override = getattr(args, "device_group_prefill_override", "")
    device_group_decode_override = getattr(args, "device_group_decode_override", "")

    if output_csv.exists():
        output_csv.unlink()

    cleanup_runtime_artifacts()

    failures = 0
    for row in rows:
        if int(row["_data_row"]) not in selected:
            continue
        model = value(row, "model")
        if model in SKIPPED_MODELS:
            print(f"Skipping row {row['_data_row']} ({model}): in SKIPPED_MODELS (>70B).")
            continue
        if latest_only and model not in latest_models:
            print(f"Skipping row {row['_data_row']} ({model}): not in LATEST_MODELS (latest-only mode).")
            continue
        if not row_matches_filter(row, "disagg_mode", disagg_mode):
            print(f"Skipping row {row['_data_row']} ({model}): disagg_mode does not match filter {disagg_mode!r}.")
            continue
        if not row_matches_filter(row, "specialization_mode", specialization_mode):
            print(
                f"Skipping row {row['_data_row']} ({model}): specialization_mode does not match "
                f"filter {specialization_mode!r}."
            )
            continue
        if not row_matches_filter(row, "blocking_mode", blocking_mode):
            print(f"Skipping row {row['_data_row']} ({model}): blocking_mode does not match filter {blocking_mode!r}.")
            continue

        # Apply device group overrides if provided
        if device_group_override:
            # For configs with device_group in additional_config JSON (embedding, audio)
            update_device_group_in_additional_config(row, device_group_override)
            # Also update direct device_group column if it exists (LLM configs)
            if "device_group" in row and value(row, "device_group") is not None:
                row["device_group"] = device_group_override
        if device_group_encode_override:
            row["encode_device_group"] = device_group_encode_override
        if device_group_prefill_override:
            row["prefill_device_group"] = device_group_prefill_override
        if device_group_decode_override:
            row["decode_device_group"] = device_group_decode_override

        ok = run_one(row, args, args.config_name, output_csv)
        if not ok:
            failures += 1
            if args.fail_fast:
                break

    print()
    print(f"Output CSV: {output_csv}")
    print("Note: Use merge_published_results.py to generate consolidated published CSV")

    if failures:
        print(f"Completed with {failures} failed row(s).")
        return 1
    print("Completed successfully.")
    return 0
