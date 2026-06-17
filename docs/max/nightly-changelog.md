---
title: Nightly (v26.5)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added DiffusionGemma (`DiffusionGemmaForBlockDiffusion`), an
  encoder/decoder block-diffusion text model that generates 256-token
  blocks per step via an inner denoising loop. Supports NVFP4 and bfloat16
  weights; text-only for now.
- Added tool-calling and reasoning support to Qwen 3.5 / 3.6.
- Added support for the Ideogram 4 (`Ideogram4Pipeline`) text-to-image
  flow-matching diffusion transformer. The pipeline pairs a Qwen3-VL text
  encoder (run text-only, emitting concatenated intermediate hidden states)
  with a single-stream DiT that uses QK-RMSNorm, 3D MRoPE, SwiGLU, and AdaLN,
  and an asymmetric dual-branch classifier-free guidance scheme. FP8
  (`float8_e4m3fn`) checkpoint weights are dequantized to `bfloat16` at load.
  Serve via `/v1/responses`; benchmark with
  `--benchmark-task text-to-image`.

## MAX framework

### Inference server

- Fixed image requests failing with a 400 or 500 across all vision models. Two
  bugs in the shared image-resolution layer: `data:` URIs with unpadded or
  URL-safe base64 (sent routinely by clients and relays) were rejected by the
  strict decoder, and truncated, animated, or content-negotiated images (for
  example a `.jpg` URL that a host serves as WebP) passed the lazy header-only
  validation and then crashed later in the tokenizer's pixel decode with an
  unhandled error. Image payloads are now decoded tolerantly and validated with
  a full pixel decode that the tokenizer reuses (so each image is decoded only
  once), and undecodable content fails fast as a clean 400.
- Fixed intermittently-dropped Kimi K2.5 tool calls under reasoning-enabled
  `tool_choice="auto"`. The model often opens a tool-call section directly from
  inside its `<think>` block without emitting a closing `</think>` (an implicit
  end-of-reasoning, part of Kimi's interleaved-thinking design). The reasoning
  parser previously ended a reasoning span only on `</think>`, so the entire
  tool-call section was misclassified as reasoning and never reached the tool
  parser, so the response came back with empty `content` and the tool-call
  payload stranded in `reasoning`. Because whether the model emits `</think>`
  is sampling-dependent, the failure was flaky. The reasoning parser now also
  ends the span at `<|tool_calls_section_begin|>`, leaving the marker as
  content so the tool call is parsed correctly.
- Fixed a structured-output runaway: a `response_format` JSON schema that omits
  the root `"type"` (for example `{"properties": {"x": {}}}`, valid JSON Schema)
  previously compiled to a grammar that permitted a bare, unbounded top-level
  value, so a model that looped inside that value could never emit a terminator
  and generated until `max_length` (`finish_reason="length"`). Such schemas with
  an object-implying keyword (`properties`, `required`, `additionalProperties`,
  `patternProperties`) are now normalized to `"type": "object"` before grammar
  compilation, matching the behavior of xgrammar-based engines. A genuinely
  empty `{}` schema is still treated as "any value".
- Retuned the Prometheus/OpenTelemetry histogram buckets for MAX Serve metrics.
  Previously every histogram shared one millisecond-latency bucket range, which
  was inaccurate for non-latency metrics. Each histogram now uses bucket
  boundaries matched to its actual range (percentages bucket 0–100, token and
  occupancy counts use power-of-two buckets, batch size is fine-grained up to
  512, throughput and time metrics use appropriately wide ranges, and time
  metrics now extend out to 30 minutes). Quantile queries become more accurate;
  dashboards that hardcoded specific bucket boundaries may need updating.
- Changed `maxserve.cache.num_used_blocks` and `maxserve.cache.num_total_blocks`
  from counters to gauges. These report an instantaneous level, so a gauge is
  correct; as counters their exported values were meaningless. The Prometheus
  type changes to `gauge` and the exported series drops the counter `_total`
  suffix.
- Added `maxserve.cache.disk_blocks_read` and
  `maxserve.cache.disk_blocks_written` counters, reporting KV blocks read from
  and written to the disk cache tier when tiered (disk) KV caching is enabled.

### `max` CLI

### Python API

- **Preview (no-op today)**: `InferenceSession.profiling` is a new namespace
  that will control the libkineto-backed MAX profiler. The lifecycle methods
  are callable but do not yet produce trace files; the libkineto-backed
  Chrome-trace JSON output (compatible with
  [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis)) and the
  `session.debug.profiling_*` setter mirrors land in subsequent nightlies.
  The control surface is final: `session.profiling.start()` / `.stop()` /
  `.wait_for_trace()` and the read-only `.state` and `.is_enabled` properties.
  This API is orthogonal to the existing `session.gpu_profiling()` (NVTX/Nsight)
  path.

- `ProfilingConfig` gains six new fields for the libkineto profiler:
  `profiling_enabled`, `profiling_output_path`, `profiling_dynolog_enabled`,
  `profiling_warmup_steps`, `profiling_active_steps`, and
  `profiling_periodic_flush_seconds`.

- Eager execution in `max.experimental` now routes every realization through
  the `max.experimental.executor.Executor` abstraction. The out-of-the-box
  path is unchanged — graphs within the `MAX_INTERPRETER_MAX_OPS` threshold run
  on the interpreter and fall back to a cached compile otherwise — but it is
  now expressed as a new `CompositeExecutor` selected by
  `MAX_EAGER_EXECUTOR=composite` (the new default). The
  `MAX_USE_EAGER_INTERPRETER` environment variable has been removed; force
  compilation with `MAX_EAGER_EXECUTOR=compile` instead. The
  `EagerRealizationContext(use_interpreter=...)` argument is deprecated in
  favor of `EagerRealizationContext(executor=...)`.

- `max.nn.hooks.PrintHook` now supports `max.experimental.nn.Module`.

- Added `F.print`, which supports both single-device and multi-device tensors.

## MAX kernels

- Sped up GPU RMS norm on AMD CDNA4 (MI355X) for prefill-sized shapes. The
  warp-tiling path runs one row per block, so the per-thread SIMD width sets
  how many warps a row needs; on CDNA4, when there are enough rows to keep the
  GPU busy, using a 2x-wider per-thread SIMD halves the warps per row, which
  cheapens the block reduction and raises blocks-per-CU. This improves
  throughput by roughly 15-31% on shapes such as 8192x{2880,4096,5120,8192}
  and 4096x4096 (bfloat16), with no change to small-row shapes or other
  architectures.
- Fixed a rare illegal-instruction crash in the SM100 (Blackwell)
  flash-attention prefill kernels under chunked prefill with tensor
  parallelism. When the attention grid shared SMs with the tensor-parallel
  all-reduce collective under device graph capture, a consumer warp could read
  a stale tensor-memory base address and issue a tensor-core MMA against an
  invalid operand. The kernels now read the tensor-memory base once after it
  is published and carry it in a register, so there is no in-loop re-read to
  race.

## Breaking changes

## Fixes

- Fixed `max.nn.WeightNormConvTranspose1d` raising `AttributeError` when
  constructed with its default `has_bias=False`. The constructor
  unconditionally deleted the wrapped conv's `bias` attribute, which is only
  set when `has_bias=True`; the delete is now guarded.
- Fixed a GPU memory fault when benchmarking GPU layer norm: the benchmark's
  output lambda copy-captured the wrong tensor, so the actual output tensor was
  captured by reference and dereferenced as a host pointer on the device. This
  faulted on AMD GPUs (and was undefined behavior elsewhere). The lambda now
  captures the output tensor it writes to.
- Fixed `max.experimental.nn.Conv2d.forward` moving the weight to the
  input's device but leaving the bias behind, which failed with a device
  mismatch when the bias started on a different device than the input. The
  bias is now moved alongside the weight.

- Fixed a constrained-decoding bug that could intermittently drop grammar
  enforcement during speculative decoding with grammar-guided tool calling.
  The speculative bitmask walk advanced the matcher through draft tokens and
  restored it with `rollback`, but `rollback` does not correctly restore the
  matcher across certain tool-call structural tags (e.g.
  `<|tool_call_begin|>`). The walk now runs on a deep copy of the matcher,
  leaving the real matcher untouched.

## Mojo language
