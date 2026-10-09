# vLLM QAIC Benchmark Report Generation and Email Distribution

This document describes the complete workflow for generating and distributing benchmark reports.

## Overview

The vLLM QAIC benchmark pipeline now includes automated report generation and email distribution:

1. **Benchmark Execution**: Run all benchmark categories (LLM, embedding, audio, VLM)
2. **CSV Consolidation**: Merge all result CSVs into a single consolidated published CSV
3. **HTML Report Generation**: Create a professional HTML report from the consolidated CSV
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

1. **Merges all result CSVs** using `merge_published_results.py`:
   - Reads all `*_results.csv` files from the results directory
   - Consolidates into a single `consolidated_published_results.csv`
   - Includes the benchmark, comparison, and `vllm_exec_time_s` fields. The common environment fields `vllm_qaic_branch`, `qaic_disagg_branch`, `qserve_branch`, `qeff_branch`, and `qaic_sdk_version` are shown once at the top of the HTML report instead of being repeated in every CSV row.

2. **Generates HTML report** using `generate_html_report.py`:
   - Creates a professional HTML report with styling
   - Includes environment information section (branch details, SDK version)
   - Includes test results summary (total, passed, failed counts)
   - Includes detailed test results table with timing and performance metrics
   - Compares each matching row with the selected comparison build; rows with an absolute change greater than 5% in QPC total size or export/compile timing metrics are marked `FAIL` in red with the failure reason shown
   - Writes `N/A` for metrics that are not applicable to a model, and `-` for a successful row with no failure or comparison reason, in both the consolidated CSV and HTML table
   - Displays comparable metrics as adjacent previous/current columns, such as `Previous QPC Size` and `Current QPC Size`
   - Reports `vllm_exec_time_s` immediately before the failure reason; it measures from server launch until client completion
   - Keeps QPC count and per-QPC size-list fields in the CSV, while showing only the previous/current QPC total size pair in the HTML table
   - Stores the five common environment values in a sidecar `<csv>.environment.json` file so rerunning report generation does not lose the values after they are removed from the CSV rows

### Step 3: Email Distribution (Automatic)

If `EMAIL_RECIPIENTS` parameter is set:

- **On Success**: Sends HTML report with both HTML and CSV attachments
- **On Failure**: Sends failure notification with build details link

## Jenkins Configuration

### Parameters

Add the following parameter to your Jenkins job:

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

For a local comparison, add `--previous-csv` with the previous consolidated CSV.

## Report Contents

### Environment Information Section

Displays:
- vLLM QAIC Branch / Commit
- QAIC Disagg Branch / Commit
- QServe Branch / Commit
- QEff Branch / Commit
- QAIC SDK Version

### Test Results Summary

Shows:
- Total number of tests
- Number of passed tests
- Number of failed tests

### Detailed Test Results Table

Columns:
- Model Name
- Category (LLM, Embedding, Audio, VLM)
- Config (config_name)
- Config Summary
- Status (✓ PASS / ✗ FAIL)
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
  - `consolidated_published_results.csv` - Consolidated results data for reference

## Troubleshooting

### Report not generated

1. Check that all benchmark stages completed successfully
2. Verify the results directory contains `*_results.csv` files
3. Check Jenkins logs for the "Generate Report" stage

### Email not sent

1. Verify `EMAIL_RECIPIENTS` parameter is set (not empty)
2. Check Jenkins email configuration (Manage Jenkins > Configure System > Email Notification)
3. Verify the email plugin is installed and configured
4. Check Jenkins logs for email delivery errors

### Missing data in report

1. Verify all benchmark categories ran successfully
2. Check that result CSVs contain data (not empty)
3. Verify environment variables are properly set in result CSVs
