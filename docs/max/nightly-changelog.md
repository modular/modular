---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

### Inference server

### Server metrics

### `max` CLI

- Fixed LoRA and denoising-cache CLI flags replacing, rather than
  overriding, the matching `--config-file` section; `--enable-lora=false`
  now also disables LoRA that a recipe enabled, instead of being ignored.

### Python API

- Eager mode tensors will use the JIT by default. This unlocks fusion and
  shape specialization optimizations even for eager code, beating PyTorch
  performance in eager in the common case.

### C API

## MAX kernels

## Breaking changes

- Reworked `max.pipelines.PipelineArgs` and `PipelineConfig` construction
  around a single path and a single (nested) shape:
  - `PipelineArgs` now nests its runtime, sampling, and profiling fields in
    `runtime`, `sampling`, and `profiling` sub-configs
    (`PipelineRuntimeConfig`, `SamplingConfig`, and `ProfilingConfig`),
    matching the nested shape already used by recipes and `PipelineConfig`.
    Flat constructor kwargs for those fields (for example `max_batch_size=1`)
    are rejected; pass `runtime=PipelineRuntimeConfig(max_batch_size=1)`
    instead, and use the nested keys in config files validated into
    `PipelineArgs`. `PipelineArgs.from_flat_kwargs` (the CLI path) still
    accepts the flat spellings and routes them to the sub-configs.
  - Removed `PipelineConfig.from_flat_kwargs` and
    `PipelineArgs.from_pipeline_config`; `PipelineConfig.from_args` is the
    single way to construct a `PipelineConfig` from user input. Replace
    `PipelineConfig.from_flat_kwargs(...)` with
    `PipelineConfig.from_args(PipelineArgs.from_flat_kwargs(...))`.
  - `PipelineConfig.from_args` now also applies the model generation config's
    sampling defaults, applies `--model-override` entries, and resolves the
    speculative draft architecture, so programmatically constructed
    `PipelineArgs` behave the same as CLI invocations.
  - `PipelineRuntimeConfig` is now exported from `max.pipelines`.

- The legacy alias-buffer LoRA path has been removed. ModuleV3 LoRA (adapters
  passed as graph inputs) is now the only supported LoRA implementation.
  Serving a non-ModuleV3 architecture with `--lora-paths` now raises a clear
  error at startup instead of building a manager that never applies the
  adapters; serve the model's ModuleV3 variant (for example,
  `--prefer-module-v3`) to use LoRA adapters.

## Fixes

- Fixed `ops.group_norm()` raising `NotImplementedError` in eager mode on
  CPU. `group_norm` previously had a GPU-only kernel; it now has a CPU
  compute path too, so eager `group_norm` runs on CPU the same way
  `layer_norm`/`rms_norm` already do.

- Fixed the BF16 Expert Parallelism (EP) dispatch path failing to compile.
  The `ep.dispatch_async` kernel requires a `dispatch_scale_dtype` comptime
  parameter, but the BF16 branch of `call_ep_dispatch_async` only set
  `dispatch_fmt_str` and omitted the scale dtype, so any model using BF16 EP
  dispatch (for example, a non-quantized MoE) hit a graph-compile error. The
  BF16 branch now sets `dispatch_scale_dtype = float32` to match the kernel
  signature.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
