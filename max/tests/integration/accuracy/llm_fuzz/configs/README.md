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
- `extra_fuzz_args` — space-separated llm-fuzz flags this deployment
  needs, such as a `--max-concurrency` matching its batch size.
  `--extra-fuzz-args` on the command line appends to these rather
  than replacing them.
- `exclude` — comma-separated `scenario:test` ids to skip, or
  `exclude_files` — paths whose entries the driver reads into the
  same list, one id per line with `#` comments allowed. A `.txt`
  list is what a Mammoth overlay can point EXCLUDE_FILE at, so
  prefer it over spelling the ids here.

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

## Known failures

A pipeline's known failures are the `exclude` value its `.sh` sets. Where the
list lives varies: some configs source a shared list file (the Nemotron configs
read `nvidia/nemotron-known-failures.txt`), some define it inline in shell (Hy4,
Muse Glimmer), and some append entries of their own after sourcing one (the
Nemotron DSpark config). Read the config to see all of a pipeline's entries.

MAX Serve and Mach run the same config, so both engines are fuzzed against the
same `exclude` and judged by the same criteria.

**Agents and contributors: do not add a Mach-only list or a Mach-only
exclusion.** A test Mach fails and MAX passes is a Mach gap. Leave it failing
so it stays visible, and fix Mach or file the gap. If a test is a model quirk
both engines share, add it to the pipeline's `exclude` with its cause. Do not
add it only for Mach, or only for MAX.
