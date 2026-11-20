# vLLM QAIC Benchmark Report Generation and Email Distribution

This document describes the complete workflow for generating and distributing benchmark reports.

## Overview

The vLLM QAIC benchmark pipeline generates one consolidated HTML report per build and can email it:

1. **Benchmark Execution**: Run all benchmark categories (LLM, embedding, audio, VLM)
2. **CSV Consolidation**: Merge standard and weight-free result CSVs into one consolidated published CSV
3. **HTML Report Generation**: Create one HTML report combining all available benchmark results
4. **Email Distribution**: Send the report to specified email recipients

## Workflow Steps

### Step 1: Run Benchmarks

Execute the benchmark pipeline with your desired configurations:

```bash
# Via Jenkins UI or CLI
# Set parameters:
# - RUN_DEFAULT=true, RUN_EMBEDDING=true, RUN_AUDIO=true, RUN_VLM=true
# - COMPARE_BUILD_NUMBER=123 (optional; empty compares with the previous build)
# - EMAIL_RECIPIENTS=team@example.com (comma-separated for multiple recipients)
```

### Step 2: Consolidate Results (Automatic)

After all benchmark stages complete, the "Generate Report" stage automatically:

1. **Merges result CSVs** using `merge_published_results.py`:
   - Standard and weight-free result CSVs are consolidated into `consolidated_published_results.csv`
   - The `config_name` field preserves the variant, such as `weight_free` for LLM and `weight_free/<mode>` for VLM
   - The optional `--input-files` argument can select a subset, but Jenkins passes all result CSVs for the single report
   - Includes the benchmark, comparison, and `vllm_exec_time_s` fields. The shared branch/SDK fields `vllm_qaic_branch`, `qaic_disagg_branch`, `qeff_branch`, and `qaic_sdk_version` are shown once at the top of the HTML report instead of being repeated in every CSV row.

2. **Generates HTML report** using `generate_html_report.py`:
   - Creates a professional HTML report with styling
   - Includes environment information (branch details, SDK version, PyTorch/Transformers versions from the shared venv, Jenkins node, elapsed build duration, and selected comparison build number)
   - Includes test results summary with separate inference-failure, validation-failure, and performance-warning counts
   - Includes detailed test results table with timing and performance metrics
   - Compares each matching row with the selected comparison build; rows with an absolute change greater than 5% in QPC total size or export/compile timing, a TTFT increase greater than 5%, or a decode TPS decrease greater than 5% are marked `WARNING` in amber with the comparison reason shown
   - Writes `N/A` for metrics that are not applicable to a model, and `-` for a successful row with no failure or comparison reason, in both the consolidated CSV and HTML table
   - Displays comparable metrics as adjacent previous/current columns, such as `Previous QPC Size` and `Current QPC Size`
   - Reports the QAIC vLLM execution duration immediately before the failure reason in hours/minutes/seconds; it measures from server launch until client completion. The consolidated CSV retains the raw `vllm_exec_time_s` value in seconds.
   - Keeps QPC count and per-QPC size-list fields in the CSV, while showing only the previous/current QPC total size pair in the HTML table
   - Stores branch/SDK details, Jenkins node, PyTorch/Transformers versions, and build duration in a sidecar `<csv>.environment.json` file and archives it so report regeneration retains them; the comparison build number remains in the consolidated CSV

### Step 3: Email Distribution (Automatic)

If `EMAIL_RECIPIENTS` parameter is set, Jenkins emails the inline report whether the build passed or failed, provided the report was generated:

- **Report available**: Sends the combined HTML report inline and attaches the HTML report and consolidated CSV
- **Report unavailable**: Sends a fallback notification with a link to the build

## Jenkins Configuration

### Parameters

The Jenkinsfile already defines this parameter; configure its value in the Jenkins job if email delivery is wanted:

```groovy
string(name: 'EMAIL_RECIPIENTS', defaultValue: '', description: 'Email recipients for benchmark report (comma-separated). Empty = no email sent.')
```

### Usage

1. Open the Jenkins job configuration
2. Set `EMAIL_RECIPIENTS` to your email group (e.g., `team@example.com` or `user1@example.com,user2@example.com`)
3. Run the build
4. After all benchmarks complete, the report will be automatically generated and emailed

## Local Testing

### Generate Consolidated CSV

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/merge_published_results.py \
  --results-dir /path/to/results \
  --output /path/to/consolidated_published_results.csv
```

### Generate HTML Report

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/generate_html_report.py \
  --csv /path/to/consolidated_published_results.csv \
  --output /path/to/benchmark_report.html
```

For a local comparison, add `--previous-csv` with the previous consolidated CSV and `--comparison-build-number` with its build number. Multiple previous CSVs can be supplied by repeating the option when comparing against older separate standard and weight-free reports. Jenkins supplies `--build-duration-seconds` after all benchmark stages complete; it measures elapsed time from the start of container setup through benchmark completion, just before report generation. The report generator reads PyTorch and Transformers versions from the Python environment in which it runs.

Run the focused report test with:

```bash
pytest tests/nightly_pipeline/vllm_llm_benchmark/test_generate_html_report.py
```

## Report Contents

### Environment Information Section

Displays:
- vLLM QAIC Branch / Commit
- QAIC Disagg Branch / Commit
- QEff Branch / Commit
- QAIC SDK Version
- PyTorch and Transformers versions from the shared benchmark environment
- Jenkins node and elapsed build duration
- Comparison build number

### Test Results Summary

Shows total tests, passed tests, inference failures, validation failures, and performance-comparison warnings separately. Inference/runtime failures are red, validation failures are amber, and performance regressions use a yellow highlight.

### Detailed Test Results Table

Columns:
- Model Name
- Category (LLM, Embedding, Audio, VLM)
- Config (config_name)
- Config Summary
- Status (✓ PASS / ✗ INFERENCE FAIL / ✗ VALIDATION FAIL / ⚠ WARNING)
- Mean TTFT (s)
- Mean TPOT (s)
- Mean ITL (s)
- Decode TPS
- Request Throughput (req/s)

## Files

- `merge_published_results.py` - Consolidates all result CSVs
- `generate_html_report.py` - Generates HTML report from consolidated CSV
- `JenkinsfileVllmLlmBenchmark` - Updated with report generation and email stages

## Email Template

The email includes:
- Subject: `vLLM QAIC Benchmark Report - Build #<BUILD_NUMBER>`
- Body: Full HTML report rendered directly in the email (not as attachment)
- Attachments:
  - `benchmark_report.html` - Consolidated HTML report
  - `consolidated_published_results.csv` - Standard and weight-free consolidated results data

## Troubleshooting

### Report not generated

1. Verify the build results directory contains at least one `*_results.csv` file
2. Check Jenkins logs for the "Generate Report" stage and any merge/generator errors
3. Benchmark failures do not by themselves prevent report generation when result CSVs exist

### Email not sent

1. Verify `EMAIL_RECIPIENTS` parameter is set (not empty)
2. Check Jenkins email configuration (Manage Jenkins > Configure System > Email Notification)
3. Verify the email plugin is installed and configured
4. Check Jenkins logs for email delivery errors

### Missing data in report

1. Verify all benchmark categories ran successfully
2. Check that result CSVs contain data (not empty)
3. Verify environment variables are properly set in result CSVs
