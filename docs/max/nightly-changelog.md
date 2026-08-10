---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

## MAX framework

- `--num-speculative-tokens` is now unset by default, and each speculative
  method resolves its own default: `eagle` and `mtp` keep drafting 2 tokens
  per step, while `dflash`-style block drafters (DFlash, DSpark) derive the
  draft checkpoint's trained block width. Explicit values are honored as
  before. Previously the flag defaulted to 2 for every method and block
  drafters overrode it at load time with a warning; a bare DFlash run now
  also sizes its KV cache draft headroom at the trained width instead of the
  old default.
- The graph compiler now fuses query/key RMSNorm followed by rotate-half RoPE
  into a single `rms_norm_rope` GPU kernel even when the RMSNorm upcasts to
  `float32`; numerics match the unfused graph.
- Added a `poison-all` mode to `MODULAR_DEBUG_DEVICE_ALLOCATOR` that fills
  every memory-manager allocation with a configurable NaN-pattern byte
  (`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_POISON_PATTERN`), so uninitialized
  device-memory reads trip differential tests without kernel instrumentation.
  Manual debugging aid, not a default.
- Added conda packages `max-benchmark`, `max-serve`, and `max-all`, plus a
  `max[all]` wheel extra, for parity with the existing wheel extras.
- Multimodal pipelines now compile their vision and language models in
  parallel via a shared `Module` container and `session.load_all()`, cutting
  compile/load time by up to 1.86x (Qwen3-VL-4B: 614s -> 428s).
- Made the compiled-model (MEF) cache key relocatable across install paths:
  absolute-path-valued pipeline options no longer enter the key, so a cache
  warmed under one install path hits under another.
- ModuleV3 weights are now sharded and transferred to devices inside the
  compiled graph rather than via eager ops, reducing per-GPU memory use
  (about 10 GiB for a DP-EP NVFP4 DeepSeek-V3).
- The VMM defragmenting allocator is now the default memory manager on NVIDIA
  GPUs, fixing external-fragmentation OOMs ("plenty free but no contiguous
  block"); override with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=0`. Also
  fixed the earlier opt-in being a silent no-op.
- Added a HIP-based VMM defragmenting allocator for AMD GPUs (opt-in via
  `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`) on MI300-series hardware.
- Coalesced consecutive Metal kernel launches into a single shared command
  buffer with a tunable op cap, reducing per-launch overhead on Apple GPUs;
  also restored Metal GPU execution aborted by an unimplemented
  driver-context stub.
- Improved expert-parallel MoE execution by running the shared expert on a
  side stream via `ops.side_stream`, overlapping it with the routed-expert
  computation.
- Allowed `float16`/`bfloat16` graphs to load `float32` checkpoint weights,
  with the weight adapter casting at load time.
- Improved multi-device startup latency by batching replay preface copies
  into a single submission.
- The vision encoder cache can store embeddings in fixed-size blocks,
  enabled by setting the `MAX_EXPERIMENTAL_VISION_CACHE_UTILIZATION`
  environment variable to a fraction in (0, 0.5] of the KV cache pool
  budget (`0`, the default, keeps the entry-count cache) on
  architectures whose memory planner reports a vision row spec (Gemma 4
  and Kimi K2.5). Capacity is a byte budget carved into 128-token
  blocks — a video spans many blocks and an image a few — so a
  video-capable model no longer collapses the cache to a handful of
  worst-case-video slots that starve image workloads.
- Vision embedding assembly during chunked prefill is now bounded by the
  active window: each step copies only the embedding rows whose
  placeholder tokens fall inside the chunk, with dense scatter indices,
  instead of rebuilding every image's rows with out-of-bounds sentinels.
  Per-chunk copy cost now scales with the chunk size rather than the
  request's total image tokens.

### Inference server

### Server metrics

### `max` CLI

- `max warm-interpreter-cache` now shows a live progress row per op family.

- Fixed `max warm-interpreter-cache` failing with a `ValueError` on a
  machine where an op family supports none of the available devices (for
  example, a GPU-only op family on a CPU-only machine). Such a family now
  warms as a no-op instead of aborting the whole command.

- Fixed LoRA and denoising-cache CLI flags replacing, rather than
  overriding, the matching `--config-file` section; `--enable-lora=false`
  now also disables LoRA that a recipe enabled, instead of being ignored.

### Python API

- Eager mode tensors will use the JIT by default. This unlocks fusion and
  shape specialization optimizations even for eager code, beating PyTorch
  performance in eager in the common case.

- `max.graph.ops.reduce_scatter_rms_norm` takes an optional `group_size`
  argument, matching `max.graph.ops.reducescatter.sum`: the devices split into
  contiguous groups of that many, each reducing independently, so the fused op
  also works under tensor-parallel-within-data-parallel topologies. It was
  previously full-world only and silently disabled itself whenever the
  tensor-parallel degree was smaller than the device count.

- `max.graph.ops.allgather_rms_norm` takes an optional `group_size` argument,
  matching `max.graph.ops.allgather`: the devices split into contiguous groups
  of that many, each gathering independently, so the fused op also works under
  tensor-parallel-within-data-parallel topologies. It was previously full-world
  only.

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

- Fixed DeepSeek-V3.2 and GLM-5.x pipelines ignoring `--max-length`: the
  resolved maximum sequence length was silently pinned to the DeepSeek
  default (163840) regardless of the flag or the checkpoint's advertised
  limit. These models also now size their rotary-embedding tables from the
  resolved maximum sequence length instead of the checkpoint's
  `max_position_embeddings`.

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
