# Nightly: v26.3

This version is still a work in progress.

## Highlights {#26-3-highlights}

## Documentation {#26-3-docs}

## MAX models {#26-3-models}

- Add MXFP4 quantization support for GPT-OSS models (e.g openai/gpt-oss-20b).

## MAX framework {#26-3-max}

### Inference server {#26-3-max-serve}

### `max` CLI {#26-3-max-cli}

### Python API {#26-3-max-python}

- Renamed `Float8Config` to `QuantConfig` (and related types/functions)
  to reflect that the config now covers FP8, NVFP4, and MXFP4 quantization.
- Renamed related public Python quantization APIs from `Float8*` names to
  `Quant*` names, including `parse_float8_config()` to
  `parse_quant_config()`, and the public `quant` modules in `max.nn` and
  `max.pipelines.lib`.
- `max.diagnostics.gpu.BackgroundRecorder`'s sampling interval can now be
  configured.

## Breaking changes {#26-3-breaking}

### Mojo API {#26-3-max-mojo}

### Custom ops {#26-3-custom-ops}

## MAX kernels {#26-3-max-kernels}

- kbench now runs benchmarks via shared library (.so) by default, reusing
  persistent workers and CUDA contexts instead of spawning subprocesses.
  Benchmark execution phase is ~10x faster (e.g. 4.25 h → 0.4 h on a tuning
  workload). Falls back to subprocess mode when profiling or using custom exec
  wrappers.

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

## Mojo language {#26-3-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog)
