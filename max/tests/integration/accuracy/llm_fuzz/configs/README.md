# `llm_fuzz/configs/`

Per-pipeline serve configurations for the llm-fuzz CI driver. Each
file is a shell fragment whose variables are sourced to build the
MAX Serve invocation and the matching fuzz subprocess.

## Schema

A config file sets the variables the driver expects after sourcing:

Serve flags:

- `batch_size` — passed as `--max-batch-size`
- `max_length` — passed as `--max-length`
- `extra_pipelines_args` — bash array of additional serve flags
  (e.g. `--ep-size`, `--data-parallel-degree`,
  `--kv-cache-format`)

llm-fuzz knobs (defaults; the workflow can override per dispatch):

- `model_profile` — required, passed as `--model-profile`
- `scenarios` — empty runs the tool's full default suite
- `k2vv_mode` — only meaningful when `scenarios` selects the
  kimi-k2vv suite

The model path is **not** set in this file. It comes from the matrix
entry (`max/tests/integration/accuracy/llm_fuzz_matrix.py`), which
carries CI-scheduling concerns: runner, GPU layout, instance type,
timeout, and the default model path. Keeping the matrix scoped to
"where/when" and the config scoped to "what" means a new pipeline
shape lands one config file plus one matrix entry.

## Adding a new entry

1. Drop a `.sh` file under `<owner>/<name>.sh` here.
2. Add a matching entry to `PIPELINES` in
   `max/tests/integration/accuracy/llm_fuzz_matrix.py` with the
   runner, GPU layout, instance type, and timeout.
3. Add the same `<owner>/<name>` string to the `pipeline` input's
   `options` list in `.github/workflows/llmFuzzAdHoc.yaml`, so the
   value is dispatchable.
4. If the config names a `model_profile` that doesn't exist yet,
   register it in `MODEL_PROFILES`
   (`max/tests/integration/accuracy/llm_fuzz/model_config.py`) —
   `fuzz.py` validates `--model-profile` against those keys.
