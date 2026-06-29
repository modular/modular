---
title: Nightly (v26.5)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added GLM-5.2 (`GlmMoeDsaForCausalLM`) support, extending the existing
  GLM-5.1 sparse-attention architecture with cross-layer index sharing.
- Added Laguna (`LagunaForCausalLM`), poolside's decoder-only sparse-MoE
  language model. It uses sigmoid expert routing with a per-expert
  score-correction bias, a per-element softplus attention-output gate, and
  per-head QK-RMSNorm. Verified on `poolside/Laguna-M.1-NVFP4` (131B,
  compressed-tensors NVFP4 experts) on a single B200, including chat-template
  serving and tool calling. On GSM8K (0-shot) it scores ~0.81 with light
  sampling (`temperature=0.3` plus a frequency penalty); greedy decoding
  (`temperature=0`) is **not** recommended for this NVFP4 checkpoint, since it
  falls into repetition loops on a sizable fraction of prompts (dropping GSM8K
  to ~0.59). An experimental, not-yet-accuracy-validated FP8 KV cache (unscaled
  cast) is available behind `--kv-cache-format float8_e4m3fn`; the default bf16
  KV cache is the validated configuration.
- Added DiffusionGemma (`DiffusionGemmaForBlockDiffusion`), an
  encoder/decoder block-diffusion text model that generates 256-token
  blocks per step via an inner denoising loop. Supports NVFP4 and bfloat16
  weights; text-only for now.
- Added Nemotron-H (`NemotronHForCausalLM`), NVIDIA's hybrid Mamba-2 +
  attention + relu-squared-MLP decoder, with modelopt per-tensor FP8. Adds a
  new Mamba-2 SSD chunked-scan varlen prefill kernel (also used for decode as
  length-1 sequences). Verified on `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` on a
  single B200: random-weight logit-verify cosine 0.9999 vs HuggingFace, GSM8K
  strict-match ~0.70. Decode is optimized with an in-place SSM state-pool
  read-modify-write that writes only the active slots (+52% output tok/s at
  concurrency 32).
- Added tool-calling and reasoning support to Qwen 3.5 / 3.6.
- Added support for the Ideogram 4 (`Ideogram4Pipeline`) text-to-image
  flow-matching diffusion transformer. The pipeline pairs a Qwen3-VL text
  encoder (run text-only, emitting concatenated intermediate hidden states)
  with a single-stream DiT that uses QK-RMSNorm, 3D MRoPE, SwiGLU, and AdaLN,
  and an asymmetric dual-branch classifier-free guidance scheme. FP8
  (`float8_e4m3fn`) checkpoint weights are dequantized to `bfloat16` at load.
  Serve via `/v1/responses`; benchmark with
  `--benchmark-task text-to-image`.
- Added the `reasoning_split` chat-completion request field for MiniMax M3.
  It defaults to `true`, which keeps the existing behavior of returning the
  model's thinking in a separate `reasoning` field. Setting it to `false` folds
  the thinking back into the `content` field wrapped in `<think>...</think>`
  tags, matching the official MiniMax M3 endpoint. The field is a no-op for
  every other model.

## MAX framework

- Data-parallel (DP) serving now shares the prefix cache across replicas, so a
  multi-turn conversation gets cache hits even when a later turn is scheduled on
  a different replica than the previous one. GPU prefix-cache hits are served by
  a cheap device-to-device copy of the cached pages onto the assigned replica,
  and the CPU/disk offload tiers are now a single pool shared by every replica
  (a block offloaded by one replica can be loaded by another). As a result,
  `host_kvcache_swap_space_gb` now sizes one shared host pool of that size for
  the whole deployment, rather than allocating a separate pool of that size per
  replica.
- The graph compiler now fuses query/key RMSNorm followed by rotate-half RoPE
  into a single `rms_norm_rope` GPU kernel even when the RMSNorm is written "in
  float32" — that is, when a `bfloat16`/`float16` activation is upcast to
  `float32`, normalized, and cast back before RoPE. Previously the intervening
  `float32`-to-`bfloat16` downcast blocked the fusion and the idiom compiled to
  several separate elementwise kernels. The fused kernel now decouples its
  output dtype from its input dtype, so the reduction and weight/epsilon scaling
  stay in `float32` and only the result is produced in the activation dtype; the
  input upcast is absorbed by ordinary prologue fusion. Numerics match the
  unfused graph (the normalized value is rounded to the output dtype before
  RoPE).
- Added a `poison-all` mode to the `MODULAR_DEBUG_DEVICE_ALLOCATOR` environment
  variable for debugging uninitialized device-memory reads. Unlike the existing
  `uninitialized-poison` (which fills graph tensors with a type-aware, non-NaN
  sentinel and is detected by an instrumented load check), `poison-all` fills
  *every* memory-manager allocation — including internal scratch and other
  non-tensor buffers — with a raw byte (default `0xFF`, a NaN pattern for
  `float32`/`bfloat16`), so an uninitialized read propagates NaN into the output
  and trips existing differential tests without any kernel instrumentation. The
  fill byte is configurable via
  `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_POISON_PATTERN`, and the mode composes
  with `out-of-bounds` redzone checks. Because the NaN can also surface on
  legitimately-uninitialized allocation padding, it is a manual debugging aid
  rather than a default.

### Inference server

- Added an opt-in `emit_reasoning_content` server config. When enabled, chat
  completion responses emit a reasoning model's chain-of-thought under
  `reasoning_content` instead of `reasoning` (the two are never emitted
  together). This restores the `reasoning_content` field for clients that
  require it; it remains off by default, so responses emit `reasoning` only.
- Improved time-to-first-token for multimodal requests by making the image and
  video preprocessor reject and decode media more efficiently. Oversized media
  is now rejected before its bytes are fully materialized: an `http(s)` download
  is aborted as soon as the advertised `Content-Length` (or the streamed total)
  crosses the per-item cap, and a `data:` URI is rejected from its base64 length
  before it is decoded. Large `data:` base64 decoding now runs on a worker
  thread instead of blocking the server event loop, so one big payload no longer
  stalls other in-flight requests. Per-request video count and per-video byte
  limits are also enforced up front (mirroring the existing image limits).
- Reduced per-iteration latency for structured-output (constrained decoding)
  requests on speculative-decode models. The overlap pipeline now enqueues the
  asynchronous FSM-advance and bitmask compute once the next iteration's batch
  order is known, so the bitmask is written directly in the consuming batch's
  row order. This removes both the host-synchronization point that previously
  stalled the GPU-feeding thread when the batch composition changed between
  iterations and the device-side gather that earlier reconciled the order. The
  improvement applies across all six supported speculative-decode architectures
  (Kimi K2.5 MLA and MHA, DeepseekV3 MTP and Eagle3, Gemma 4 MTP,
  and EAGLE Llama 3).
- Constrained decoding (structured output) now unpacks the grammar bitmask on
  the GPU. The packed `int32` bitmask is transferred to device as-is and
  unpacked and applied to the logits in a single fused kernel
  (`apply_packed_bitmask`), instead of unpacking to a `bool` tensor on the CPU.
- Made numpy array transport across the API-server-to-model-worker request
  queue zero-copy. Large arrays (notably multi-image or high-resolution vision
  `pixel_values`) now ride out-of-band as their own ZMQ frame instead of being
  copied into the message body and then again through the socket, and the
  receiver decodes them as views with no copy. This is faster than both the
  previous copy and shared-memory transports at every payload size (for example
  ~5x faster than the copy path and ~2x faster than shared memory at 24-32 MiB
  in the transport microbenchmark), and removes the per-request shared-memory
  segment (and its sizing, leak, and page-fault costs) from this path entirely.
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
- Added opt-in SHA-256 KV-cache block hashing. A new `kv_cache_hash_algo`
  field on `KVCacheConfig` (default `ahash64`; opt-in `sha256` and
  `sha256_64`) threads through the pipeline and serve config, selecting a
  Mojo `block_hasher_sha256` and the matching `hash_request_tokens` SHA-256
  path. Chat-completion requests also accept an optional `cache_salt` field
  that scopes prefix-cache reuse to a single per-request KV chain. Default
  behavior is the same as the existing `ahash64` path.
- Added opt-in SHA-256 KV-cache block hashes through host-tier KV
  connectors. `NullConnector`, `LocalConnector`, and `TieredConnector` now
  accept 32-byte SHA-256 digests alongside 64-bit `ahash64` hashes. The
  `KVConnector` Protocol's `load` and `offload` take `Sequence[bytes]`
  block hashes and a `bytes | None` parent-sequence hash; the block
  manager coerces legacy `ahash64` int hashes to bytes (8-byte
  big-endian, signed) at the boundary, so a connector implementation only
  ever sees one hash shape. Connectors advertise what they accept via a
  new `supported_hash_algos: frozenset[KVHashAlgo]` property (default
  `frozenset({"ahash64"})`), which the block manager validates against
  the configured `kv_hash_algo` at startup so a mismatch fails fast with
  a clear remediation message. The disk tier names files `<hex>.bin` (16
  hex chars for 64-bit hashes, 64 hex chars for SHA-256 digests) and pins
  the algo in a `kv-disk-cache.meta.json` sidecar to refuse cross-algo
  reuse of a cache directory. `KVHashAlgo` is re-exported from
  `max.nn.kv_cache` for downstream consumers. Default behavior is
  unchanged.
- Extended SHA-256 KV-cache block hashes to the dKV (`DKVConnector`)
  external tier. `DKVConnector.supported_hash_algos` now advertises
  `frozenset({"ahash64", "sha256", "sha256_64"})`, and `load`/`offload`
  accept both 8-byte (`ahash64` / `sha256_64`) and 32-byte (full
  `sha256`) block hashes; 32-byte digests are truncated to their first
  8 bytes at the boundary into the unchanged `dkv_connector` Rust
  client, which continues to carry a `uint64 seq_hash` on the wire.
  Truncation is byte-identical to the existing `sha256_64` algorithm,
  so configuring MAX with `sha256` or `sha256_64` produces the same
  dKV key for the same logical block — no change to the dkv wire
  format, stored block identity, or `DKVExternalBlockMetadata`
  orchestrator hint shape. Default behavior is unchanged.

### `max` CLI

- The entrypoint for the CLI, formerly `max.entrypoints`, has been marked as
  private and moved to `max._entrypoints`. The CLI is still a public facing API,
  but the code within it is not.

### Python API

- Added `max.driver.set_virtual_cpu_target()` and `get_virtual_cpu_target()`.
  Set a fixed CPU codegen target (for example `"x86-64-v3"`, `"neoverse-n1"`,
  or `"generic"` for the most-portable baseline of the host arch family) before
  importing `max._interpreter_ops` so the eager interpreter's CPU kernel cache
  is compiled host-independently and can be shipped and reused across hosts of
  the same architecture family. Mirrors `set_virtual_device_target_arch()` for
  GPUs. Leaving it unset compiles for the build host's CPU, as before.

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

- The eager interpreter now compiles its matmul and unary-elementwise
  graph-compiler models lazily, per target on first dispatch, by default —
  bounding compile cost to the targets a program uses instead of JIT-compiling
  the full kernel library at import. Set `MAX_EAGER_OP_PRECOMPILE=1` to
  precompile the full matrix at import instead.

- Added a `max warm-interpreter-cache` command that batch-compiles the full
  eager interpreter model matrix into the on-disk cache for the current
  machine's devices and drops a stamp. A later lazy eager process on the same
  device set adopts the warm — one batched cache load instead of compiling each
  target on first use — so later programs start warm. Run it as a provisioning
  step (for example a Dockerfile `RUN`) on the target hardware. Pure
  optimization: if skipped, or on a different device set, dispatch compiles each
  target lazily.

- Added `max.experimental.nn.subgraphable` for `Module` subgraph compilation: a
  repeated block (via the `@subgraphable` class decorator, or the
  `subgraphable(layer)(x)` call form) lowers to one shared subgraph reused per
  call. Opt out per compile with `Module.compile(..., allow_subgraphs=False)`.

- `max.nn.hooks.PrintHook` now supports `max.experimental.nn.Module`.

- Added `F.print`, which supports both single-device and multi-device tensors.

- Added `max.graph.default_custom_extensions()` and the
  `default_custom_extensions_scope()` context manager. Paths registered as
  defaults are merged into the `custom_extensions` of every new `Graph`, so a
  backend can make its custom-op kernel library reachable from graphs built
  without an explicit `custom_extensions=` — including the eager-realization
  graph that backs `max.experimental` tensors. Empty by default.

- Moved the `max.entrypoints` package to be private. In doing so, we
  deprecated the `max.entrypoints.LLM` API and we'll introduce a new API
  for offline inference in a future release.

### C API

- Fixed `M_borrowTensorInto()` copying instead of borrowing a GPU input. When
  the borrowed pointer already lived on the target accelerator, the call
  allocated a fresh device buffer and copied into it, so in-place mutation of a
  `BufferType` model input was applied to the engine's private copy and never
  reflected back into the caller's buffer. Such pointers are now borrowed in
  place (zero-copy) on CUDA devices, matching the documented borrow semantics
  and the existing behavior for host inputs. Host pointers passed with a device
  spec are still staged via a host-to-device copy, as are device pointers on
  backends that do not yet implement in-place borrowing (AMD and Apple).

## MAX kernels

- The `TileTensor` layout type no longer takes an `element_size` parameter. A
  tensor's logical element width is now carried by its `Storage` parameter via
  `PointerStorage[element_width]` (default `PointerStorage[1]`), and
  `element_size` remains available as a derived comptime member. Code that
  passed `element_size=N` should now pass
  `Storage=PointerStorage[element_width=N]`, or use `TileTensor.vectorize()` to
  build the vectorized view.
- Apple silicon GPU support for running MAX models has been extended to M1 and
  M2 systems. Previously, the optimized matrix multiplication kernels for Apple
  silicon GPUs only returned correct results on M3 and newer systems. That has
  now been fixed for M1 and M2 systems, allowing many common MAX models to run
  correctly on them.
- The split-K decode attention kernel for Apple GPUs is now the default for
  token-generation attention, covering paged-KV-cache MHA and GQA decode for
  head dims that are a multiple of 32. It was previously opt-in;
  `MODULAR_ENABLE_APPLE_NAIVE_FA_DECODE=0` now opts out, falling back to
  `mha_gpu_naive`.
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

- Fixed inflated `maxserve.cache.h2d_blocks_copied` and
  `maxserve.cache.d2h_blocks_copied` telemetry on tiered and local KV cache
  deployments. The scheduler now resets connector transfer counters after each
  batch metrics sample so OpenTelemetry counters report per-batch deltas.
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

- Fixed slicing and `view()` on a `max.driver.DevicePinnedBuffer` silently
  returning a plain `Buffer`. The decayed type lost the pinned buffer's
  no-synchronization behavior, so a later `to_numpy()` on the slice triggered
  an unexpected device synchronization. Slices and views now preserve the
  `DevicePinnedBuffer` type.

## Mojo language
