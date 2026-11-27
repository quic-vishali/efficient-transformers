# vLLM LLM Benchmark Pipeline

This directory contains the vLLM QAIC benchmark runners used by
`scripts/JenkinsfileVllmLlmBenchmark`. There are four runner scripts, each
covering one model domain, sharing all CSV parsing, command building, and
subprocess orchestration from `vllm_benchmark_common.py`:

- `vllm_llm_benchmark.py` — LLM serving configs.
- `vllm_embedding_benchmark.py` — embedding-model serving configs.
- `vllm_audio_benchmark.py` — audio (whisper) serving configs.
- `vllm_vlm_benchmark.py` — vision-language model serving configs.

Each script defines its own `LATEST_MODELS` curated set; `SKIPPED_MODELS`
(the >70B hardware-capability exclusion) is shared and applied to all four.
The VLM runner also has a VLM-only skip list for models currently failing to download in the nightly.

The runners are CSV-driven. Each input CSV owns one serving mode and produces
one output CSV:

| Mode | Input CSV | Output CSV |
| --- | --- | --- |
| LLM EPD (subfunction, weight-free, blocking) | `configs/llm_configs.csv` | `llm_results.csv` |
| embedding | `configs/embedding_configs.csv` | `embedding_results.csv` |
| audio | `configs/audio_configs.csv` | `audio_results.csv` |
| VLM | `configs/vlm_configs.csv` | `vlm_results.csv` |

## Local Dry Run

Dry run validates CSV parsing, command generation, log paths, and output CSV
writing without starting vLLM or using QAIC devices:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name llm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_configs.csv \
  --output-csv /tmp/vllm_llm_smoke/llm_results.csv \
  --results-dir /tmp/vllm_llm_smoke \
  --rows 1 \
  --dry-run
```

Row 1 in the unified LLM CSV is `openai/gpt-oss-20b`, intended for a quick
smoke check before running all rows.

The embedding and audio runners work the same way:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_embedding_benchmark.py \
  --config-name embedding \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/embedding_configs.csv \
  --output-csv /tmp/vllm_embedding_smoke/embedding_results.csv \
  --results-dir /tmp/vllm_embedding_smoke \
  --rows 1 \
  --dry-run

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_audio_benchmark.py \
  --config-name audio \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/audio_configs.csv \
  --output-csv tmp/vllm_audio_smoke/audio_results.csv \
  --results-dir tmp/vllm_audio_smoke \
  --rows 1 \
  --dry-run

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_vlm_benchmark.py \
  --config-name vlm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/vlm_configs.csv \
  --output-csv vllm_vlm_smoke/vlm_results.csv \
  --results-dir vllm_vlm_smoke \
  --rows 3 \
  --dry-run
```

## Consolidated Published CSV

After all benchmark categories complete, run `merge_published_results.py` to generate
a consolidated published CSV with only the key fields for team distribution. Jenkins
merges all selected benchmark results into one consolidated CSV. The generic
command below merges the files it finds in a local results directory:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/merge_published_results.py \
  --results-dir vllm_llm_results/ \
  --output vllm_llm_results/consolidated_published_results.csv
```

This merges the selected `*_results.csv` files (from LLM, embedding, audio, and VLM categories)
into one file with row-level fields including model, model_category, config_name,
config_summary, status, previous/current QPC total size, previous/current
export/compile timings, qpc_count, qpc_sizes_mb, mean_ttft_s, mean_tpot_s,
mean_itl_s, decode_TPS, request_throughput_req_s, server_command,
client_command, comparison_build_number, vllm_exec_time_s, and reason. The four shared
branch/SDK fields (`vllm_qaic_branch`, `qaic_disagg_branch`, `qeff_branch`, and
`qaic_sdk_version`) are removed from the final CSV and shown once at the top of the
HTML report. Additional report-only metadata is stored in an environment sidecar.
Model categories
are auto-detected based on config_name. The five `*_export_compile_time_s` columns
are populated only for the fields relevant to that row's server type — e.g. a
non-disagg (`api_server`) row only fills `export_compile_time_s`, while a disagg
row fills the per-stage `prefill_mdp_export_compile_time_s`/
`prefill_export_compile_time_s`/`decode_export_compile_time_s`/
`encode_export_compile_time_s` fields instead.
Fields that do not apply to a row are written as `N/A` in the consolidated CSV
and displayed as `N/A` in the HTML table. A successful row with no failure or
comparison reason uses `-` in the `reason` column.

## Response Capture and Validation

Response capture is opt-in per CSV row. Add the following fields to a row:

```text
save_result=true
save_detailed=true
validation_method=string
expected_output=OK
```

The exact benchmark result is saved under the case log directory as
`benchmark_result.json`; generated text is in its `generated_texts` field and
is also written to `response.txt`.

For PyTorch-backed validation, embedding responses are compared as native
vectors using cosine similarity. Text responses are compared exactly first;
when they differ, both sides are encoded with
`Qwen/Qwen3-Embedding-0.6B` using Hugging Face Transformers (no
`sentence-transformers` dependency), left padding, and final-token pooling.
Embeddings are L2-normalized before cosine scoring. Inputs are capped at 8192
tokens, bounded by the model and tokenizer limits; longer inputs are truncated
from the right (preserving the start of the text) with a warning. Transformers
4.51.0 or newer is required. Text cosine uses `text_cosine_threshold` (or
`cosine_threshold`), defaulting to `0.98`. A semantic fallback pass is
reported as `passed_cosine`. Set `VLLM_TEXT_VALIDATION_MODEL` to override the
encoder; overrides must be compatible with final-token pooling.

For standalone validation, use `validation_method=string` for substring
matching or `validation_method=cosine` for numeric vectors, with
`expected_output` or `expected_output_file` containing a JSON vector and an
optional `cosine_threshold` (default `0.98`). Embedding rows should provide a
request for `/v1/embeddings`, such as:

```text
validation_method=cosine
validation_request={"model":"my-embedding-model","input":"hello"}
expected_output_file=validation/hello_embedding.json
cosine_threshold=0.95
```

The result CSV records `response_file`, `validation_status`,
`validation_score`, and `validation_error`. Failed or errored validation is
reported as `validation_failed`. Response JSON files and server/client logs are
written under the build's results directory; Jenkins archives the consolidated
CSV, HTML report, and environment metadata sidecar, but not those per-request files.
Validation failures remain visible in the CSV and HTML report, but by themselves
do not fail the Jenkins benchmark stage or overall build. Server, client, and
other benchmark execution failures still fail the stage.

## HTML Report Generation

After generating the consolidated CSV, create an HTML report for email distribution:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/generate_html_report.py \
  --csv vllm_llm_results/consolidated_published_results.csv \
  --output vllm_llm_results/benchmark_report.html
```

For a local comparison against a previous consolidated CSV, add
`--previous-csv /path/to/previous/consolidated_published_results.csv` and
`--comparison-build-number 123`. Jenkins defaults to the preceding build unless
`COMPARE_BUILD_NUMBER` selects another one. Jenkins also passes
`--build-duration-seconds` when it generates the report. Rows with
an absolute change greater than 5% in QPC size or export/compile timing, a TTFT
increase greater than 5%, or a decode TPS decrease greater than 5% are marked
`WARNING` in the merged status column and include the previous value, current
value, percentage change, and comparison reason.

The HTML report includes:
- **Environment Information**: Branch and QAIC SDK details, PyTorch and Transformers
  versions from the shared venv, Jenkins node, elapsed build duration, and the
  selected comparison build number
- **Test Results Summary**: Total tests, passed tests, inference failures, validation failures, and performance-comparison warnings counted separately
- **Detailed Test Results**: Table with model name, category, config, status, per-stage
  export/compile times (export/compile, prefill MDP, prefill, decode, encode), previous
  timing values, and performance metrics
(TTFT/TPOT/ITL/decode TPS/throughput) with a combined failure-reason column. Inference failures are shown in red, validation failures in amber, and performance-comparison warnings with a yellow highlight.
CSV comparison metrics are displayed as adjacent previous/current pairs, for
example `Previous QPC Size` followed by `Current QPC Size`. TTFT and decode TPS
are compared directionally: a TTFT increase or decode TPS decrease greater
than 5% produces a warning; improvements do not.
`vllm_exec_time_s` measures elapsed time from server process launch until the
client process exits, including server startup/load and benchmark execution.
The HTML report formats this duration as hours/minutes/seconds; the consolidated
CSV keeps the raw value in seconds.
QPC count and per-QPC size-list fields remain available in the CSV but are not
previous-value comparison fields and are omitted from the HTML table. The HTML
table includes only the previous/current QPC total size pair.
The generator also writes `<csv>.environment.json` beside the consolidated CSV;
this preserves branch/SDK details, Jenkins node, PyTorch and Transformers versions,
and build duration when the report is generated again. The comparison build number
remains in the consolidated CSV.

## Jenkins Flow

The Jenkins pipeline uses one shared Python environment at
`vllm_llm_workspace/venv` for all benchmarks. It does not create a separate
weight-free environment or dependency-preparation stage. With
`SKIP_INSTALL=true`, Jenkins reuses the existing shared venv.

The Jenkins pipeline:

1. Creates or reuses the shared Python venv.
2. Clones `https://github.com/qualcomm/vllm-qaic.git`.
3. Runs `scripts/install.sh aot`, which installs QEfficient internally.
4. Clones and installs `qaic-disagg`.
5. Runs the selected LLM/embedding/audio/VLM config CSVs using the shared venv.
6. Runs `merge_published_results.py` once for all selected benchmark CSVs.
7. Runs `generate_html_report.py` to render one `benchmark_report.html` containing all rows.
   The displayed build duration is measured from the start of container setup through
   completion of benchmark stages, immediately before report generation.
8. Archives the build's CSV files, HTML report, and `.environment.json` metadata sidecar.
9. Emails the combined report inline when available and attaches the HTML report and consolidated CSV
   to `EMAIL_RECIPIENTS`, if set — this fires from the pipeline's
   `post { always { ... } }` block regardless of whether the overall build
   passed or failed, as long as the HTML report file exists (the subject line
   includes the build status). If the report could not be generated at all,
   a fallback notification with a link to the Jenkins build is sent instead.

Use `DRY_RUN=true` and `ROWS_LLM=1` to verify the unified LLM command and output
CSV/log generation on a Jenkins agent without consuming QAIC runtime. The
embedding, audio, and VLM stages have their own `ROWS_EMBEDDING`/`ROWS_AUDIO`/
`ROWS_VLM` and `RUN_EMBEDDING`/`RUN_AUDIO`/`RUN_VLM` params for the same
purpose.

The Jenkins pipeline has one LLM stage: `Run LLM`. It runs all eligible rows
from `llm_configs.csv`. Every LLM row is prefill/decode disaggregated and has
subfunctions, direct `pretrained_extra_args.weight_free=true`, and direct
`qaic_config.enable_blocking=true` in its prefill and decode overrides. Use
`ROWS_LLM` when a specific row subset is needed.

The Jenkins pipeline has one VLM stage: `Run VLM`. It runs all eligible rows
from `vlm_configs.csv`. Every VLM row is EPD and has subfunctions, direct
`pretrained_extra_args.weight_free=true`, and blocking enabled. Use `ROWS_VLM`
when a specific row subset is needed.

## HuggingFace Cache Configuration

The `HF_HOME` Jenkins parameter controls where HuggingFace model caches are stored:

- **Default (empty):** `${HOME}/.cache/huggingface` on the Jenkins agent
- **Custom value:** Any path, e.g., `/mnt/fast_storage/hf_cache`

This is useful when:
- The default cache location has insufficient space
- You want to use a faster storage device (NVMe, SSD)
- You want to share a pre-populated cache across multiple builds
- You need to isolate caches for different benchmark runs

The cache is bind-mounted into the Docker container at the same path, so models
downloaded during one run are available to subsequent runs without re-downloading.

Weight-free exports stage transformed checkpoints under `QEFF_CHECKPOINT_HOME`
(configured by the Jenkins pipeline beneath `QEFF_HOME`). This keeps the shared
HuggingFace cache read-only-safe.

## Device Group Configuration

Device groups are configured in the CSVs:

| Category | CSV device fields |
| --- | --- |
| LLM (PD) | `prefill_device_group`, `decode_device_group` |
| Embedding / audio | `additional_config.device_group` |
| VLM (EPD) | `encode_device_group`, `prefill_device_group`, `decode_device_group` |

The unified LLM stage always uses its CSV device groups. Jenkins retains
per-category overrides for embedding, audio, and per-stage VLM allocation.
Use a per-row dry run to validate the command before a hardware run.

### Manual Testing

Test the unified disaggregated LLM configuration manually:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name llm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_configs.csv \
  --output-csv test/test_llm.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir test \
  --base-dir . \
  --python-bin python3 \
  --server-ready-timeout-s 3600 \
  --dry-run
```

Test embedding with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_embedding_benchmark.py \
  --config-name embedding \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/embedding_configs.csv \
  --output-csv /tmp/test_embedding.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --server-ready-timeout-s 3600 \
  --device-group-override "0" \
  --dry-run
```

Test audio with device group override:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_audio_benchmark.py \
  --config-name audio \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/audio_configs.csv \
  --output-csv /tmp/test_audio.csv \
  --rows 1 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --server-ready-timeout-s 3600 \
  --device-group-override "0" \
  --dry-run
```

Test VLM with separate device group overrides:
```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_vlm_benchmark.py \
  --config-name vlm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/vlm_configs.csv \
  --output-csv /tmp/test_vlm.csv \
  --rows 3 \
  --latest-models-only false \
  --results-dir /tmp \
  --base-dir . \
  --python-bin python3 \
  --server-ready-timeout-s 3600 \
  --device-group-encode-override "0" \
  --device-group-prefill-override "1" \
  --device-group-decode-override "2,3" \
  --dry-run
```

## CSV Notes

Common columns:

- `model`, `server_type`, `client_type`, `host`, `port`
- `PL`, `GL`, `CL`, `num_prompts`, `max_concurrency`
- `backend`, `endpoint`, `dataset_name`, `ignore_eos`
- `server_extra_args`, `client_extra_args` for mode-specific flags not yet
  modeled as first-class columns

For `server_type=api_server`, used by applicable embedding and audio rows, the
runner builds `python -m vllm.entrypoints.openai.api_server`. Its
`additional_config` JSON is constructed from the corresponding CSV fields.

For `server_type=qaic_disagg`, the runner builds `python -m qaic_disagg` using
the PD columns such as `prefill_device_group`, `decode_device_group`,
`prefill_override_qaic_config`, and `decode_override_qaic_config`. The unified
LLM CSV uses this PD layout. VLM rows use the encode-stage columns
(`encode_port`, `encode_device_group`, `encode_max_num_seqs`, and
`encode_override_qaic_config`) as well, for their EPD layout.

The embedding CSV (`configs/embedding_configs.csv`) has four rows per model,
one per `pooling_method` (`mean`, `avg`, `cls`, `max`), passed through the
`additional_config` column's `override_qaic_config.pooling_method` key, along
with the client columns `tokenizer` and `random_range_ratio` used for
Hugging Face dataset benchmarks.

The audio CSV (`configs/audio_configs.csv`) has one row per whisper model and
uses the client columns `dataset_path`, `hf_subset`, `hf_split`, and
`hf_output_len` for the `hf` dataset backend.

The unified VLM CSV (`configs/vlm_configs.csv`) has one EPD row per model.
Every populated stage override explicitly sets `use_onnx_subfunctions=true`,
`pretrained_extra_args.weight_free=true`, and the rows set
`enable_blocking=true`. VLM device groups are split across encode, prefill,
and decode stages; its server-side `limit_mm_per_prompt` JSON field limits
multimodal input per request.

## Model Filtering

After `--rows` selects which CSV rows are in play, three model-name filters are
applied, in order:

1. **`SKIPPED_MODELS`** (always enforced, no override): a hardcoded set of
   models too large to run in this pipeline (>70B real params) — currently
   `zai-org/GLM-4.5` (~355B, MoE) and `hpcai-tech/grok-1` (~314B).
2. **VLM-only download skip list** (always enforced, no override):
   `VLM_SKIPPED_MODELS` in `vllm_vlm_benchmark.py` currently contains
   `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.6-27B`, and
   `Qwen/Qwen3.6-35B-A3B`, along with the two Gemma models
   `google/gemma-4-E2B-it` and `google/gemma-4-26B-A4B-it`, which have been failing model downloads in the nightly. This list only affects VLM
   benchmark rows.
3. **`LATEST_MODELS_ONLY`** (`--latest-models-only`, default `true`): when
   enabled, restricts runs to the curated `LATEST_MODELS` set defined in the
   runner script for the domain being run (`vllm_llm_benchmark.py`,
   `vllm_embedding_benchmark.py`, `vllm_audio_benchmark.py`, or
   `vllm_vlm_benchmark.py`). Set to `false` (Jenkins param `LATEST_MODELS_ONLY`)
   to run all non-skipped models in the CSV.

`zai-org/GLM-4.5` is in both LLM's `LATEST_MODELS` and `SKIPPED_MODELS` — the
skip-list always wins, so it is excluded regardless of the latest-only flag.

Skipped rows are logged to stdout with the applicable skip-list name and do
not appear in the output CSV, matching how `enabled=false` rows are already
excluded silently.
