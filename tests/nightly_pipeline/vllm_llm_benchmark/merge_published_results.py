#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Generate a consolidated published CSV from all benchmark result CSVs.

This script scans a results directory for all *_results.csv files
(from different benchmark categories: default, ccl, blocking, disagg_pd, embedding, audio, vlm)
and generates a single consolidated published CSV with key fields for team distribution.

Usage:
    python3 merge_published_results.py --results-dir resultdir/ --output consolidated_published_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .vllm_benchmark_common import (
        NO_ERROR_MESSAGE_AVAILABLE,
        extract_server_error,
        is_generic_server_error,
    )
except ImportError:
    from vllm_benchmark_common import NO_ERROR_MESSAGE_AVAILABLE, extract_server_error, is_generic_server_error

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


def _normalize_published_row(row: dict) -> dict:
    normalized = dict(row)
    for field in PUBLISHED_FIELDS:
        if normalized.get(field) is None or str(normalized.get(field)).strip() == "":
            normalized[field] = "N/A"
    return normalized


def _safe_str(raw: object) -> str:
    return "" if raw is None else str(raw)


def _format_validation_text(value: object) -> str:
    """Serialize the complete benchmark output for published CSV artifacts."""
    return json.dumps(value, ensure_ascii=False)


def _milliseconds_to_seconds(value: object) -> str:
    try:
        return str(round(float(value) / 1000, 4))
    except (TypeError, ValueError):
        return "N/A"


def _resolve_server_log(raw_path: str, results_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    marker = "/logs/"
    if marker in raw_path:
        candidate = results_dir / "logs" / raw_path.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    candidate = results_dir / raw_path
    return candidate if candidate.exists() else None


def _resolve_result_artifact(raw_path: str, results_dir: Path) -> Path | None:
    if not raw_path or raw_path in {"N/A", "-"}:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    marker = "/logs/"
    if marker in raw_path:
        candidate = results_dir / "logs" / raw_path.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    candidate = results_dir / raw_path
    return candidate if candidate.exists() else None


def _backfill_validation_text(row: dict, results_dir: Path) -> None:
    if _safe_str(row.get("validation_actual_text")).strip() not in {"", "N/A", "-"}:
        return
    response_file = _resolve_result_artifact(_safe_str(row.get("response_file")), results_dir)
    if response_file is None:
        return
    try:
        response = json.loads(response_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for key in ("generated_texts", "transcriptions", "embeddings"):
        values = response.get(key)
        if values:
            row["validation_actual_text"] = _format_validation_text(values)
            return


def _backfill_server_error(row: dict, results_dir: Path) -> None:
    status = (row.get("status") or "").strip().lower()
    if status in {"dry_run", "pass"}:
        return
    existing_error = str(row.get("error") or "").strip()
    validation_status = (row.get("validation_status") or "").strip().lower()
    validation_error = str(row.get("validation_error") or "").strip()
    validation_error_available = validation_status in {"failed", "error"} and validation_error not in {"", "N/A", "-"}
    error_is_missing = existing_error in {"", "N/A", "-"} or is_generic_server_error(existing_error)

    if validation_error_available and status in {"success", "validation_failed"} and error_is_missing:
        row["error"] = validation_error
        return
    if existing_error not in {"", "N/A", "-"} and not is_generic_server_error(existing_error):
        return
    server_log = _resolve_server_log(row.get("server_log", ""), results_dir)
    if server_log is None:
        if validation_error_available and error_is_missing:
            row["error"] = validation_error
        elif error_is_missing:
            row["error"] = NO_ERROR_MESSAGE_AVAILABLE
        return
    server_error = extract_server_error(server_log)
    if server_error:
        row["error"] = server_error
    elif validation_error_available and error_is_missing:
        row["error"] = validation_error
    elif error_is_missing:
        row["error"] = NO_ERROR_MESSAGE_AVAILABLE


def _model_category_from_filename(filename: str) -> str | None:
    """Infer model category from a *_results.csv filename, e.g. 'vlm_results.csv' -> 'VLM'."""
    name = filename.lower()
    if "embedding" in name:
        return "Embedding"
    if "audio" in name:
        return "Audio"
    if "vlm" in name:
        return "VLM"
    return None


def _assign_model_category(row: dict, source_filename: str) -> None:
    pooling_method = _safe_str(row.get("pooling_method")).lower()
    if pooling_method in ("mean", "avg", "cls", "max"):
        category = "Embedding"
    else:
        category = _model_category_from_filename(source_filename) or "LLM"

    row["model_category"] = category

    source_is_weight_free = "weight_free" in source_filename.lower()
    if not source_is_weight_free:
        return

    config_name = _safe_str(row.get("config_name")).strip()
    if category == "VLM":
        if not config_name.lower().startswith("weight_free/"):
            row["config_name"] = f"weight_free/{config_name}" if config_name else "weight_free"
    elif category == "LLM" and "weight_free" not in config_name.lower():
        row["config_name"] = "weight_free"


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
        _backfill_server_error(row, input_csv.parent)
        _backfill_validation_text(row, input_csv.parent)
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = _milliseconds_to_seconds(row["mean_TTFT_ms"])
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = _milliseconds_to_seconds(row["mean_TPOT_ms"])
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = _milliseconds_to_seconds(row["mean_ITL_ms"])

        # Concatenate config_summary and mode_type
        config_summary = _safe_str(row.get("config_summary")).strip()
        mode_type = _safe_str(row.get("mode_type")).strip()
        if mode_type:
            row["config_summary"] = f"{config_summary} | {mode_type}" if config_summary else mode_type

        # Determine model category from the source filename (config_name alone doesn't
        # identify VLM/audio rows, since their effective config_name never contains those
        # words) and pooling_method
        _assign_model_category(row, input_csv.name)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_published_row(row) for row in rows)


def merge_results_to_published_csv(
    results_dir: Path,
    output_csv: Path,
    input_files: list[str] | None = None,
) -> int:
    """Merge selected result CSVs in results_dir into a consolidated CSV."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: results directory does not exist: {results_dir}")
        return 1

    all_rows = []
    if input_files:
        result_csvs = [results_dir / name for name in input_files]
    else:
        result_csvs = sorted(results_dir.glob("*_results.csv"))
    # Exclude published CSVs if they exist
    result_csvs = [f for f in result_csvs if "_published" not in f.name]

    if not result_csvs:
        print(f"Warning: no result CSV files found in {results_dir}")
        return 1

    for result_csv in result_csvs:
        if not result_csv.exists():
            continue
        print(f"  Reading: {result_csv.name}")
        with result_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in rows:
                row["_source_filename"] = result_csv.name
            all_rows.extend(rows)
            print(f"    -> {len(rows)} rows")

    if not all_rows:
        print("Error: no data found in result CSV files")
        return 1

    # Convert milliseconds to seconds for latency metrics and add model_category
    for row in all_rows:
        _backfill_server_error(row, results_dir)
        _backfill_validation_text(row, results_dir)
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = _milliseconds_to_seconds(row["mean_TTFT_ms"])
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = _milliseconds_to_seconds(row["mean_TPOT_ms"])
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = _milliseconds_to_seconds(row["mean_ITL_ms"])

        # Concatenate config_summary and mode_type
        config_summary = _safe_str(row.get("config_summary")).strip()
        mode_type = _safe_str(row.get("mode_type")).strip()
        if mode_type:
            row["config_summary"] = f"{config_summary} | {mode_type}" if config_summary else mode_type

        # Determine model category from the source filename (config_name alone doesn't
        # identify VLM/audio rows, since their effective config_name never contains those
        # words) and pooling_method
        _assign_model_category(row, row.pop("_source_filename"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_normalize_published_row(row) for row in all_rows)

    print()
    print(f"✓ Merged {len(result_csvs)} result CSV files")
    print(f"✓ Total rows: {len(all_rows)}")
    print(f"✓ Output: {output_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a consolidated published CSV from all benchmark result CSVs."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing *_results.csv files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output consolidated published CSV path",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help="Specific result CSV basenames to merge; default is all *_results.csv files.",
    )
    args = parser.parse_args()

    return merge_results_to_published_csv(Path(args.results_dir), Path(args.output), args.input_files)


if __name__ == "__main__":
    raise SystemExit(main())
