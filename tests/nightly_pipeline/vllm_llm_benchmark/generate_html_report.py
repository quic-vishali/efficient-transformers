#!/usr/bin/env python3
"""
Generate an HTML report from the consolidated published CSV for email distribution.

This script reads the consolidated published CSV and generates a formatted HTML report
with environment info (branch details, SDK version) and test results table.

Usage:
    python3 generate_html_report.py --csv consolidated_published_results.csv --output report.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from .vllm_benchmark_common import extract_server_error
except ImportError:
    from vllm_benchmark_common import extract_server_error


COMPARISON_THRESHOLD_PERCENT = 5.0
ROW_KEY_FIELDS = ("model", "model_category", "config_name", "config_summary")
COMPARISON_FIELDS = (
    ("qpc_size_mb", "QPC total size (MB)"),
    ("export_compile_time_s", "Export/compile time (s)"),
    ("prefill_mdp_export_compile_time_s", "Prefill MDP export/compile time (s)"),
    ("prefill_export_compile_time_s", "Prefill export/compile time (s)"),
    ("decode_export_compile_time_s", "Decode export/compile time (s)"),
    ("encode_export_compile_time_s", "Encode export/compile time (s)"),
)
PREVIOUS_VALUE_FIELDS = tuple(f"previous_{field}" for field, _ in COMPARISON_FIELDS)
HTML_ENVIRONMENT_FIELDS = {
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
}
REPORT_INTERNAL_FIELDS = {"error"}
COMPARISON_CURRENT_FIELDS = {field for field, _ in COMPARISON_FIELDS}
VLLM_EXEC_TIME_FIELD = "vllm_exec_time_s"
CURRENT_METRIC_FIELDS = {
    "qpc_count",
    "qpc_sizes_mb",
    "mean_ttft_s",
    "mean_tpot_s",
    "mean_itl_s",
    "decode_TPS",
    "request_throughput_req_s",
    VLLM_EXEC_TIME_FIELD,
}
REPORT_METRIC_FIELDS = COMPARISON_CURRENT_FIELDS | CURRENT_METRIC_FIELDS
COMPARISON_REASON_PREFIXES = tuple(f"{label}:" for _, label in COMPARISON_FIELDS)
SERVER_LOG_PATH_RE = re.compile(r"(?P<path>/[^\s,]+/server\.log)")
ENVIRONMENT_METADATA_SUFFIX = ".environment.json"


def _is_missing(raw: object) -> bool:
    return raw is None or str(raw).strip() in {"", "N/A", "-"}


def _is_empty_or_na(raw: object) -> bool:
    return raw is None or str(raw).strip() in {"", "N/A"}


def _display_value(row: dict, field: str, default: str = "N/A") -> str:
    raw = row.get(field)
    return default if _is_missing(raw) else str(raw)


def _is_comparison_reason(raw: object) -> bool:
    reason = str(raw or "").strip()
    return any(part.strip().startswith(prefix) for part in reason.split(";") for prefix in COMPARISON_REASON_PREFIXES)


def _read_csv(csv_path: Path) -> tuple[list[dict], list[str]]:
    if not csv_path.exists():
        return [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def _read_rows(csv_path: Path) -> list[dict]:
    return _read_csv(csv_path)[0]


def _environment_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ENVIRONMENT_METADATA_SUFFIX)


def _read_environment_metadata(csv_path: Path) -> dict[str, str]:
    metadata_path = _environment_metadata_path(csv_path)
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    return {field: str(metadata.get(field, "")) for field in HTML_ENVIRONMENT_FIELDS if metadata.get(field) is not None}


def _write_environment_metadata(csv_path: Path, row: dict) -> None:
    metadata = {field: str(row.get(field, "") or "") for field in HTML_ENVIRONMENT_FIELDS}
    if not any(metadata.values()):
        return
    metadata_path = _environment_metadata_path(csv_path)
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError:
        # The CSV and HTML report remain usable if the optional metadata file
        # cannot be written.
        pass


def _server_log_for_row(row: dict, csv_path: Path) -> Path | None:
    raw_values = [
        row.get("server_log", ""),
        row.get("error", ""),
        row.get("reason", ""),
    ]
    raw_text = " ".join(str(value or "") for value in raw_values)
    match = SERVER_LOG_PATH_RE.search(raw_text)
    if match:
        raw_path = Path(match.group("path"))
        if raw_path.exists():
            return raw_path

        marker = "/logs/"
        if marker in str(raw_path):
            relative_log = str(raw_path).split(marker, 1)[1]
            for base in (csv_path.parent, *csv_path.parents):
                candidate = base / "logs" / relative_log
                if candidate.exists():
                    return candidate

    # Older consolidated CSVs did not retain server_log or error. Infer the
    # matching archived log from its model/category/config directory name.
    model_key = _log_name_key(row.get("model", ""))
    config_name = row.get("config_name", "")
    category = (row.get("model_category") or "").strip().lower()
    log_category = config_name.lower() if category == "llm" else category
    config_key = _log_name_key(config_name)
    if not model_key or not log_category:
        return None

    for base in (csv_path.parent, *csv_path.parents):
        logs_root = base / "logs"
        if not logs_root.is_dir():
            continue
        candidates = []
        for candidate in logs_root.rglob("server.log"):
            directory_key = _log_name_key(candidate.parent.name)
            category_key = _log_name_key(candidate.parent.parent.name)
            if model_key in directory_key and config_key in directory_key and log_category == category_key:
                candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
    return None


def _log_name_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _backfill_server_errors(rows: list[dict], csv_path: Path) -> None:
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if status in {"dry_run", "pass"}:
            continue
        server_log = _server_log_for_row(row, csv_path)
        if server_log is None:
            continue
        error = extract_server_error(server_log)
        if error:
            row["error"] = error


def _row_key(row: dict) -> tuple[str, ...]:
    return tuple((row.get(field) or "").strip() for field in ROW_KEY_FIELDS)


def _numeric_value(raw: str) -> float | None:
    raw = (raw or "").strip()
    if not raw or raw in {"N/A", "-"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _change_percent(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0 if current == 0 else float("inf")
    return abs(current - previous) / abs(previous) * 100.0


def _format_value(value: float) -> str:
    return f"{value:.2f}"


def _comparison_reasons(row: dict, previous_row: dict) -> list[str]:
    reasons = []
    for field, label in COMPARISON_FIELDS:
        current = _numeric_value(row.get(field, ""))
        previous = _numeric_value(previous_row.get(field, ""))
        if current is None or previous is None:
            continue

        change = _change_percent(current, previous)
        if change > COMPARISON_THRESHOLD_PERCENT:
            unit = " MB" if field == "qpc_size_mb" else " s"
            reasons.append(f"{label}: {_format_value(previous)} -> {_format_value(current)}{unit} ({change:.1f}%)")
    return reasons


def _comparison_results(rows: list[dict], previous_rows: list[dict]) -> dict[int, tuple[bool, str, dict]]:
    """Return match status, failure reason, and previous values for each row."""
    previous_by_key: dict[tuple[str, ...], list[dict]] = {}
    for previous_row in previous_rows:
        previous_by_key.setdefault(_row_key(previous_row), []).append(previous_row)

    results: dict[int, tuple[bool, str, dict]] = {}
    for index, row in enumerate(rows):
        candidates = previous_by_key.get(_row_key(row), [])
        if not candidates:
            results[index] = (False, "", {})
            continue
        previous_row = candidates.pop(0)
        reasons = _comparison_reasons(row, previous_row)
        results[index] = (True, "; ".join(reasons), previous_row)
    return results


def _comparison_headers_html(fields=COMPARISON_FIELDS) -> str:
    headers = []
    for _, label in fields:
        headers.extend(
            [
                f'<th style="text-align: right;">Previous {label}</th>',
                f'<th style="text-align: right;">Current {label}</th>',
            ]
        )
    return "\n                            ".join(headers)


def _comparison_cells_html(row: dict, previous_row: dict, fields=COMPARISON_FIELDS) -> str:
    cells = []
    for field, _ in fields:
        previous_value = _display_value(previous_row, field) if previous_row else "-"
        current_value = _display_value(row, field)
        cells.extend(
            [
                f'<td class="metric comparison-previous">{html.escape(previous_value)}</td>',
                f'<td class="metric comparison-current">{html.escape(current_value)}</td>',
            ]
        )
    return "\n                            ".join(cells)


def _write_comparison_csv(
    csv_path: Path,
    rows: list[dict],
    fieldnames: list[str],
    comparison_results: dict[int, tuple[bool, str, dict]],
    comparison_build_number: str,
) -> None:
    # These values are rendered once in the HTML Environment Information section
    # instead of being repeated for every row in the final consolidated CSV.
    output_fields = [
        field
        for field in fieldnames
        if (
            field not in HTML_ENVIRONMENT_FIELDS
            and field not in REPORT_INTERNAL_FIELDS
            and field not in COMPARISON_CURRENT_FIELDS
            and not field.startswith("previous_")
            and field != VLLM_EXEC_TIME_FIELD
            and field not in {"comparison_build_number", "reason"}
        )
    ]
    status_index = output_fields.index("status") + 1 if "status" in output_fields else 0
    comparison_columns = []
    for field, _ in COMPARISON_FIELDS:
        comparison_columns.extend([f"previous_{field}", field])
    output_fields[status_index:status_index] = comparison_columns
    output_fields.extend(
        [
            "comparison_build_number",
            VLLM_EXEC_TIME_FIELD,
            "reason",
        ]
    )

    for index, row in enumerate(rows):
        matched, reason, previous_row = comparison_results.get(index, (False, "", {}))
        model_status = (row.get("status") or "").strip().lower()
        model_error = (row.get("error") or "").strip()
        if _is_missing(model_error):
            model_error = ""
        existing_reason = (row.get("reason") or "").strip()
        model_failed = model_status not in {"success", "dry_run", "pass"}
        # A prior report may already have merged comparison failures into
        # status=FAIL. Preserve their yellow comparison classification when
        # that CSV is regenerated.
        if model_status == "fail" and not model_error and _is_comparison_reason(existing_reason):
            model_failed = False
        if model_failed:
            row["status"] = "FAIL"
            row["reason"] = model_error or existing_reason or model_status or "Model benchmark failed"
        elif reason:
            row["status"] = "FAIL"
            row["reason"] = reason
        elif model_status == "fail" and _is_comparison_reason(existing_reason):
            row["status"] = "FAIL"
            row["reason"] = existing_reason
        else:
            row["status"] = "PASS"
            row["reason"] = "-"
        if model_failed:
            row["_failure_type"] = "model"
        elif reason:
            row["_failure_type"] = "comparison"
        else:
            row["_failure_type"] = ""
        row["comparison_build_number"] = comparison_build_number
        for field in PREVIOUS_VALUE_FIELDS:
            source_field = field.removeprefix("previous_")
            row[field] = previous_row.get(source_field, "") if matched else "-"

        # A failed inference can leave expected benchmark metrics empty. Keep
        # genuinely non-applicable fields as N/A, but make unavailable metrics
        # explicit in both the CSV and HTML reports.
        if model_failed:
            for field in REPORT_METRIC_FIELDS:
                if _is_missing(row.get(field)):
                    row[field] = "-"

        for field in output_fields:
            if field == "reason":
                row[field] = row.get(field) or "-"
            elif _is_empty_or_na(row.get(field)):
                row[field] = "N/A"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_html_report(
    csv_path: Path,
    output_path: Path,
    build_url: str = "N/A",
    previous_csv_path: Path | None = None,
    comparison_build_number: str = "",
) -> int:
    """Generate an HTML report from the consolidated published CSV."""
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        return 1

    rows, fieldnames = _read_csv(csv_path)

    if not rows:
        print("Error: no data found in CSV file")
        return 1

    environment_metadata = _read_environment_metadata(csv_path)
    if not environment_metadata:
        _write_environment_metadata(csv_path, rows[0])
        environment_metadata = {field: str(rows[0].get(field, "") or "") for field in HTML_ENVIRONMENT_FIELDS}

    _backfill_server_errors(rows, csv_path)
    previous_rows = _read_rows(previous_csv_path) if previous_csv_path else []
    comparison_results = _comparison_results(rows, previous_rows) if previous_rows else {}
    _write_comparison_csv(
        csv_path,
        rows,
        fieldnames,
        comparison_results,
        comparison_build_number,
    )
    inference_failure_count = sum(1 for row in rows if row.get("_failure_type") == "model")
    comparison_failure_count = sum(1 for row in rows if row.get("_failure_type") == "comparison")
    inference_failure_label = (
        f"{inference_failure_count} model" if inference_failure_count == 1 else f"{inference_failure_count} models"
    )
    comparison_failure_label = (
        f"{comparison_failure_count} model" if comparison_failure_count == 1 else f"{comparison_failure_count} models"
    )
    if previous_rows:
        baseline = f" build #{comparison_build_number}" if comparison_build_number else ""
        comparison_message = (
            f"Red rows indicate inference failures ({inference_failure_label}). "
            f"Amber rows indicate performance comparison failures ({comparison_failure_label}), "
            f"where an absolute "
            f"change greater than {COMPARISON_THRESHOLD_PERCENT:.0f}% in QPC total size "
            f"or export/compile timing versus the comparison{baseline}."
        )
    else:
        comparison_message = "No comparison build was available."

    # Extract environment info from first row
    env_info = {
        "vllm_qaic_branch": _display_value(environment_metadata, "vllm_qaic_branch"),
        "qaic_disagg_branch": _display_value(environment_metadata, "qaic_disagg_branch"),
        "qserve_branch": _display_value(environment_metadata, "qserve_branch"),
        "qeff_branch": _display_value(environment_metadata, "qeff_branch"),
        "qaic_sdk_version": _display_value(environment_metadata, "qaic_sdk_version"),
        "build_url": build_url,
    }

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM QAIC Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .env-grid {{
            width: 100%;
            margin-bottom: 20px;
        }}
        .env-row {{
            display: block;
            margin-bottom: 15px;
        }}
        .env-card {{
            background-color: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: inline-block;
            width: 48%;
            margin-right: 2%;
            vertical-align: top;
        }}
        .env-card:nth-child(odd) {{
            margin-right: 2%;
        }}
        .env-card:nth-child(even) {{
            margin-right: 0;
        }}
        .env-card-label {{
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        .env-card-value {{
            font-size: 13px;
            font-weight: 500;
            color: #333;
            word-break: break-all;
            font-family: 'Courier New', monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table-wrapper {{
            width: 100%;
            overflow-x: auto;
            overflow-y: visible;
            -webkit-overflow-scrolling: touch;
        }}
        .results-table-wrapper > table {{
            width: max-content;
            min-width: 100%;
        }}
        thead {{
            background-color: #f0f0f0;
        }}
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border: 1px solid #ddd;
            font-size: 12px;
            white-space: nowrap;
        }}
        td {{
            padding: 12px;
            border: 1px solid #eee;
            font-size: 12px;
            word-wrap: break-word;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: 600;
        }}
        .status-failed {{
            color: #e74c3c;
            font-weight: 600;
        }}
        .model-name {{
            font-family: 'Courier New', monospace;
            font-size: 11px;
            word-break: break-word;
        }}
        .metric {{
            text-align: right;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }}
        .footer {{
            background-color: #f5f5f5;
            padding: 20px 30px;
            text-align: center;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #eee;
        }}
        .summary-table {{
            width: 100%;
            margin-bottom: 20px;
        }}
        .summary-cell {{
            width: 33.33%;
            padding: 15px;
            background-color: #f0f7ff;
            border: 1px solid #b3d9ff;
            text-align: center;
            vertical-align: top;
        }}
        .summary-card-value {{
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }}
        .summary-card-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .comparison-note {{
            margin: 0 0 12px 0;
            padding: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffecb5;
            color: #664d03;
            font-size: 12px;
        }}
        tr.model-failed {{
            background-color: #f8d7da !important;
        }}
        tr.comparison-failed {{
            background-color: #ffedd5 !important;
        }}
        .model-failure-reason {{
            color: #b02a37;
            font-weight: 700;
        }}
        .comparison-failure-reason {{
            color: #9a3412;
            font-weight: 700;
        }}
        .failure-legend {{
            margin: 0 0 12px 0;
            font-size: 12px;
        }}
        .failure-legend span {{
            display: inline-block;
            margin-right: 16px;
            padding: 4px 8px;
            border: 1px solid #ddd;
        }}
        .model-failure-legend {{
            background-color: #f8d7da;
        }}
        .comparison-failure-legend {{
            background-color: #ffedd5;
        }}
        .comparison-reason-fail {{
            color: #b02a37;
            font-weight: 700;
        }}
        .comparison-reason {{
            min-width: 260px;
            font-size: 11px;
        }}
        .comparison-previous {{
            min-width: 300px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>vLLM QAIC Benchmark Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        </div>

        <div class="content">
            <!-- Environment Info Section -->
            <div class="section">
                <div class="section-title">Environment Information</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">vLLM QAIC Branch / Commit</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["vllm_qaic_branch"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QAIC Disagg Branch / Commit</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qaic_disagg_branch"]}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QServe Branch / Commit</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qserve_branch"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QEff Branch / Commit</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qeff_branch"]}</div>
                        </td>
                    </tr>
                    <tr>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">QAIC SDK Version</div>
                            <div style="font-size: 13px; font-weight: 500; color: #333; font-family: 'Courier New', monospace; word-break: break-all;">{env_info["qaic_sdk_version"]}</div>
                        </td>
                        <td style="width: 50%; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #667eea; border-bottom: 1px solid #eee;">
                            <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; font-weight: 600;">Build URL</div>
                            <div style="font-size: 13px; font-weight: 500; color: #0066cc; font-family: 'Courier New', monospace; word-break: break-all;">
                                <a href="{env_info["build_url"]}" style="color: #0066cc; text-decoration: none;">{env_info["build_url"]}</a>
                            </div>
                        </td>
                    </tr>
                </table>
            </div>

            <!-- Test Results Summary -->
            <div class="section">
                <div class="section-title">Test Results Summary</div>
                <table class="summary-table">
                    <tr>
                        <td class="summary-cell">
                            <div class="summary-card-value">{len(rows)}</div>
                            <div class="summary-card-label">Total Tests</div>
                        </td>
                        <td class="summary-cell">
                            <div class="summary-card-value">{sum(1 for r in rows if (r.get("status") or "").upper() == "PASS")}</div>
                            <div class="summary-card-label">Passed</div>
                        </td>
                        <td class="summary-cell">
                            <div class="summary-card-value">{sum(1 for r in rows if (r.get("status") or "").upper() != "PASS")}</div>
                            <div class="summary-card-label">Failed</div>
                        </td>
                    </tr>
                </table>
            </div>

            <!-- Test Results Table -->
            <div class="section">
                <div class="section-title">Detailed Test Results</div>
                <div class="comparison-note">{html.escape(comparison_message)}</div>
                <div class="failure-legend">
                    <span class="model-failure-legend">Inference failure ({inference_failure_label})</span>
                    <span class="comparison-failure-legend">Perf comparison failure ({comparison_failure_label})</span>
                </div>
                <div class="results-table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th style="text-align: left;">Model</th>
                            <th style="text-align: left;">Category</th>
                            <th style="text-align: left;">Config</th>
                            <th style="text-align: left;">Summary</th>
                            <th style="text-align: left;">Status</th>
                            {_comparison_headers_html(COMPARISON_FIELDS)}
                            <th style="text-align: right;">TTFT (s)</th>
                            <th style="text-align: right;">TPOT (s)</th>
                            <th style="text-align: right;">ITL (s)</th>
                            <th style="text-align: right;">Decode TPS</th>
                            <th style="text-align: right;">Throughput (req/s)</th>
                            <th style="text-align: right;">vLLM execution time (s)</th>
                            <th style="text-align: left;">Failure Reason</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for index, row in enumerate(rows):
        status = (row.get("status") or "FAIL").upper()
        status_class = "status-success" if status == "PASS" else "status-failed"
        status_text = "✓ PASS" if status == "PASS" else "✗ FAIL"
        _, _, previous_row = comparison_results.get(index, (False, "", {}))
        reason = row.get("reason", "-")
        failure_type = row.get("_failure_type", "")
        if failure_type == "model":
            reason_class = "model-failure-reason"
            row_class = ' class="model-failed"'
        elif failure_type == "comparison":
            reason_class = "comparison-failure-reason"
            row_class = ' class="comparison-failed"'
        else:
            reason_class = "comparison-reason"
            row_class = ""

        html_content += f"""                        <tr{row_class}>
                            <td class="model-name" style="text-align: left;">{_display_value(row, "model")}</td>
                            <td style="text-align: left;">{_display_value(row, "model_category")}</td>
                            <td style="text-align: left;">{_display_value(row, "config_name")}</td>
                            <td style="text-align: left;">{_display_value(row, "config_summary")}</td>
                            <td class="{status_class}" style="text-align: left;">{status_text}</td>
                            {_comparison_cells_html(row, previous_row, COMPARISON_FIELDS)}
                            <td class="metric">{_display_value(row, "mean_ttft_s")}</td>
                            <td class="metric">{_display_value(row, "mean_tpot_s")}</td>
                            <td class="metric">{_display_value(row, "mean_itl_s")}</td>
                            <td class="metric">{_display_value(row, "decode_TPS")}</td>
                            <td class="metric">{_display_value(row, "request_throughput_req_s")}</td>
                            <td class="metric">{_display_value(row, "vllm_exec_time_s")}</td>
                            <td class="{reason_class}">{html.escape(reason) or "-"}</td>
                        </tr>
"""

    html_content += """                    </tbody>
                </table>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>This report was automatically generated from vLLM QAIC benchmark results.</p>
            <p>For questions or issues, please contact the QEfficient team.</p>
        </div>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an HTML report from the consolidated published CSV.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to consolidated published CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML report path",
    )
    parser.add_argument(
        "--build-url",
        default="N/A",
        help="Jenkins build URL (optional)",
    )
    parser.add_argument(
        "--previous-csv",
        default="",
        help="Previous consolidated CSV for >5%% comparison highlighting (optional)",
    )
    parser.add_argument(
        "--comparison-build-number",
        default="",
        help="Build number represented by --previous-csv (optional)",
    )
    args = parser.parse_args()

    previous_csv = Path(args.previous_csv) if args.previous_csv else None
    return generate_html_report(
        Path(args.csv),
        Path(args.output),
        args.build_url,
        previous_csv,
        args.comparison_build_number,
    )


if __name__ == "__main__":
    raise SystemExit(main())
