---
title: Nightly (v26.5)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added GLM-5.2 (`GlmMoeDsaForCausalLM`) support, extending the existing
  GLM-5.1 sparse-attention architecture with cross-layer index sharing.
- Added multi-token prediction (MTP) speculative decoding for GLM-5.2
  (`UnifiedMTPGlm5_2ForCausalLM`). The baked-in NextN layer is served as a
  single-layer sparse-MLA draft (its own lightning indexer plus a paired
  `{mla, indexer}` KV cache); per `index_share_for_mtp_iteration`, the draft
  computes its top-k selection on the first MTP step and reuses it on the
  rest. Enabled automatically for GLM checkpoints that ship a NextN layer when
  speculative decoding is requested with no separate draft model. Validated on
  `zai-org/GLM-5.2-FP8` and `nvidia/GLM-5.2-NVFP4` across 8 B200s
  (`--speculative-method mtp`).
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
- Extended Nemotron-H with the Nemotron-3-Nano-30B-A3B hybrid MoE variant (a
  sigmoid-plus-bias top-6 router over 128 routed experts plus a shared expert)
  and enabled the Nemotron-H architecture on Apple silicon GPUs in bfloat16.
  The MoE path on Metal uses an integer-domain expert-gather index
  (`ops.floor_div`, avoiding a 64-bit-float divide) and a 32-bit
  `moe_create_indices` atomic (Apple GPUs lack 64-bit atomics), and adds an
  Apple FP4 (W4A16) decode GEMV with an f16-domain E2M1 decode plus a
  redesigned varlen causal-conv1d kernel. Verified on M5: the 30B-A3B MoE
  serves in bfloat16 at GSM8K 8-shot ~0.85.
- Added tool-calling and reasoning support to Qwen 3.5 / 3.6.
- Added tool-calling, reasoning, and structured-output (`response_format`)
  support to GLM-5.1 / GLM-5.2, enabled with
  `--tool-parser glm45 --reasoning-parser glm45 --enable-structured-output`.
  Reasoning uses `<think>`/`</think>`; tool calls use the model's native
  `<tool_call>…<arg_key>…<arg_value>…</tool_call>` format. With constrained
  decoding, tool-call arguments are constrained to each tool's JSON schema
  (declared keys, `required` properties, and per-property types — including
  nested objects/arrays, enums, numeric bounds, and string patterns), and the
  call sequence terminates on the model's turn-ender so it can't loop. Validated
  on `zai-org/GLM-5.2-FP8`.
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
- FLUX.2 diffusion pipelines now support both denoising-cache backends to skip
  redundant transformer passes during generation: `--taylorseer` (Taylor-series
  step skipping — the recommended default, with `balanced` and `fast` presets)
  and `--first-block-caching` (first-block-residual reuse — zero-tuning and
  data-adaptive). The two are mutually exclusive and both off by default. See
  the [image generation guide](/max/inference/image-generation).
- Gemma 4 with multi-token prediction (MTP) speculative decoding
  (`UnifiedMTPGemma4ForCausalLM`) now supports image and video input.
  Previously this path was served text-only: image tokens were ingested by
  the tokenizer but the vision encoder output never reached the language
  model, so image prompts were answered as if the model were blind. The
  vision encoder now runs during prefill and its projected soft-token
  embeddings are merged into the target model, matching the non-MTP Gemma 4
  path.

## MAX framework

- Added `--no-enable-tool-call-constrained-decode` (config key
  `sampling.enable_tool_call_constrained_decode`, default enabled) to decouple
  tool-call parsing from constrained decoding. When disabled, a configured
  `--tool-parser` still parses tool calls out of the generated text, but no
  server-generated grammar is produced and the bitmask constrained-decode path
  is skipped for tool calls. Note that with it disabled, `tool_choice=required`
  or a named function can no longer force a tool call. This is independent of
  `--enable-structured-output`, which continues to gate user-supplied
  `response_format` JSON schemas.
- Fixed the `code` label on the `maxserve_request_count` metric so it reports
  the HTTP status code actually returned to the client. The count is now
  recorded from the HTTP layer, so failures rejected before generation (for
  example a request with an unreachable image URL) are counted with their real
  status code instead of being labeled `200` or dropped entirely. Liveness and
  observability endpoints (`/health`, `/version`, `/ping`, `/metrics`) are not
  counted.
- Failed request submissions in the OpenAI-compatible serving endpoints now
  surface as HTTP error responses instead of a `200 OK` streaming response that
  carries an error payload. Request tokenization and the handoff to the model
  worker now complete before the streaming response headers are sent, so a
  failure at submission time (for example, a dead model worker) maps to an HTTP
  5xx (or 4xx for input errors). Errors that occur mid-stream, after the first
  chunk has been sent, are still serialized as an error event within the stream.
- Added `MAX_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S` to control how long the server
  waits for in-flight requests to finish after receiving `SIGTERM` before
  exiting (default 5 seconds). Raise it so long-running requests are drained
  rather than dropped during a rolling restart.
- Data-parallel (DP) serving now shares the prefix cache across replicas, so a
  multi-turn conversation gets cache hits even when a later turn is scheduled on
  a different replica than the previous one. GPU prefix-cache hits are served by
  a cheap device-to-device copy of the cached pages onto the assigned replica,
  and the CPU/disk offload tiers are now a single pool shared by every replica
  (a block offloaded by one replica can be loaded by another). As a result,
  `host_kvcache_swap_space_gb` now sizes one shared host pool of that size for
  the whole deployment, rather than allocating a separate pool of that size per
  replica.
- The dKV external KV-cache connector (`--kv-connector dkv`) now supports
  data-parallel (DP) serving and shares its prefix cache across DP replicas on
  the default single-tenant path, matching the `local` and `tiered` connectors.
  Every replica resolves to the same replica-agnostic store, and the stored
  block key carries no replica component, so a block offloaded through one
  replica is served to any other.
- The dKV external KV-cache connector now supports tensor parallelism
  (TP greater than 1) on the multi-tenant path for head-sharded (MHA/GQA), MLA
  (replicated-KV), and GQA head-replicated (`allow_kv_head_replication`) models.
  Each GPU handshakes its own per-shard store, and every KV load/offload fans
  out across the processing replica's shard clients with identical block ids and
  hashes; a block counts as loaded only once every shard has it. The store key
  reflects the KV-head slice each GPU holds: the TP rank when head-sharded, a
  single shared shard for MLA, and the head-group index under head replication.
- On the dKV multi-tenant tensor-parallel path, a KV load that returns
  differing block counts across a replica's per-GPU shard clients now drains
  the over-loading shards' in-flight device reads before returning the minimum
  count. This keeps a stray in-flight host-to-device copy (into a block the
  block manager frees because it did not land on every shard) from later
  clobbering a reallocated block. The drain host-completes the reads on the
  remote (NIXL) transport and enqueues a cross-stream ordering on the
  co-located same-host (CUDA) transport, so it closes the window on both. The
  common equal-count path is unchanged and pays no extra synchronization.
- The dKV external KV-cache connector (`--kv-connector dkv`) now requires a
  non-empty tenant identity (`MODULAR_DKV_TENANT_ID`, set by the deployment
  operator); the empty-tenant "default" path is removed. Both the connector and
  the dKV server now reject an unset/empty tenant rather than keying an unfenced
  shared store, so every deployment (single-tenant included) routes through the
  per-tenant region-sharded store — DP replicas of one tenant still share one
  store. Multi-cache models (speculative draft+target, quantized values+scales)
  now resolve on this path, folded into the handshake's `kv_config_hash`. A
  single-tenant node spanning more than one GPU must set the dKV server's
  `--fair-share-partitions` to its GPU count.
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
- Added a `max-benchmark` conda package for parity with the `max[benchmark]`
  wheel extra.

### Inference server

- Raised the maximum tool function name length from 64 to 1024 characters.
  Client-supplied tool names that legitimately exceed 64 characters are now
  accepted instead of rejected with a 400 error.
- Added vision encoder statistics to the scheduler's per-iteration batch log
  for multimodal models. Each batch line now includes a `Vision Encoder`
  clause reporting the number of images encoded this iteration versus served
  from the vision encoder cache (with the cache hit rate), and the image
  patches and vision tokens encoded. This makes it clear when a slow
  context-encoding iteration is driven by the vision encoder rather than the
  language model. The same values are exported as OpenTelemetry metrics under
  the `maxserve.vision.*` namespace. Applies to models backed by the shared
  vision encoder cache (Gemma 4, Kimi K2.5, and Gemma 4 MTP).
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

- Added `max.graph.ops.floor_div` (and `F.floor_div`), element-wise floor
  division matching Python `//`. Unlike `ops.div`, integer operands stay in
  the integer domain instead of being promoted to `float64`, so integer floor
  division compiles on backends without 64-bit float support (for example,
  Metal GPUs).

- Added `max.driver.set_virtual_cpu_target()` and `get_virtual_cpu_target()`.
  Set a fixed CPU codegen target (for example `"x86-64-v3"`, `"neoverse-n1"`,
  or `"generic"` for the most-portable baseline of the host arch family) before
  importing `max._interpreter_ops` so the eager interpreter's CPU kernel cache
  is compiled host-independently and can be shipped and reused across hosts of
  the same architecture family. Mirrors `set_virtual_device_target_arch()` for
  GPUs. Leaving it unset compiles for the build host's CPU, as before.

- **Preview (no-op today)**: `InferenceSession.profiling` is a new namespace
  that will control the libkineto-backed MAX profiler. The control surface is
  final and callable — `session.profiling.start()` / `.stop()` /
  `.wait_for_trace()` and the read-only `.state` / `.is_enabled` — but it does
  not record yet: libkineto-backed Chrome-trace JSON capture (compatible with
  Meta's [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis))
  lands in a later nightly. This API is orthogonal to the existing
  `session.gpu_profiling()` (NVTX/Nsight) path; see `max/docs/profiling.md`
  in the repository for the user guide.
- `ProfilingConfig` gains six new fields for the libkineto profiler, each
  configurable from Python through the matching `session.debug.profiling_*`
  setter or its `MODULAR_MAX_DEBUG_PROFILING_*` environment variable:
  `profiling_enabled`, `profiling_output_path`, `profiling_dynolog_enabled`,
  `profiling_warmup_steps`, `profiling_active_steps`, and
  `profiling_periodic_flush_seconds`.
- `profiling_output_path` accepts template variables (`{pid}`, `{rank}`) and a
  directory form (`/tmp/traces/`); the path is expanded per process (keyed on
  rank, PID, timestamp, and a sequence counter) so that, once trace capture is
  enabled, multi-process and multi-rank runs won't collide on a single fixed
  filename. `{rank}` resolves to `MODULAR_RANK` / `OMPI_COMM_WORLD_RANK` /
  `"0"`.
- `max.engine.ProfilingError` is a new exception type the profiler will raise
  to surface trace-write failures.
- Forking while the profiler is enabled is safe for host applications that
  embed `InferenceSession` and fork (Python `multiprocessing` with the `fork`
  start method, pre-fork servers, or a bare `os.fork()`) — child processes
  start with the profiler disabled and can call `start()` again; the parent
  retains its enabled state across the fork. (MAX Serve itself launches
  workers with the `spawn` start method and is unaffected.)

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

- GPU token sampling with `top_k >= 10` is now 2-4x faster. The softmax,
  temperature scaling, and min-p masking steps are fused into the top-k/top-p
  rejection-sampling kernel, eliminating an intermediate probability buffer
  and two kernel launches per sampling call. The dispatch threshold between
  the two-stage top-k kernel and the rejection-sampling kernel was lowered
  from `top_k = 32` to `top_k = 10` to match the new performance crossover.
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
- Enabled the low-latency (Lamport) all-reduce on B200 for small messages
  (up to 1 MiB at 2, 4, and 8 GPUs), where it beats the one-stage path by
  roughly 1.1-1.68x. The barrier-free protocol marks unwritten slots with a
  negative-zero sentinel, so its communication region is now initialized when
  pipeline signal buffers are allocated; without that the region read as
  already-written and produced non-deterministic results.

## Breaking changes

- Removed the `MAX_SERVE_METRIC_LEVEL` and
  `MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR` environment variables along with
  the `BASIC`/`DETAILED` metric-level distinction. MAX Serve now always emits
  its full set of metrics, so panels that previously required `DETAILED` (for
  example batch execution time) are populated in every deployment rather than
  only when detailed metrics were explicitly enabled. High-volume
  per-iteration scheduler metrics are still coalesced into a single
  cross-process flush, so there is no change in per-metric recording overhead.
  To record no metrics at all (previously `MAX_SERVE_METRIC_LEVEL=NONE`), set
  `MAX_SERVE_METRIC_RECORDING_METHOD=NOOP` or `MAX_SERVE_DISABLE_TELEMETRY=1`.
- Removed `InferenceSession.use_old_top_k_kernel()` and the
  `USE_OLD_TOP_K_KERNEL` environment variable. The legacy top-k sampling
  kernel this fallback selected has been deleted; the current two-stage
  top-k kernel is now used unconditionally.
- The `Input`, `Output`, `MutableInput`, `FusedInput`, and `FusedOutput`
  `IOSpec` values used in custom-op signatures (for example,
  `Tensor[Input, spec]`) are now static members of `IOSpec`
  (`IOSpec.Input`, `IOSpec.Output`, `IOSpec.MutableInput`,
  `IOSpec.FusedInput`, `IOSpec.FusedOutput`) instead of module-level aliases.
  Update custom-op call sites to qualify these names under `IOSpec`, for
  example `Tensor[IOSpec.Input, spec]`.
- The `compiler` Mojo package has been removed. It only re-exported 4 symbols
  from `extensibility`, please use that directly instead.

## Fixes

- Fixed the compiled-model cache (`.max_cache`) not invalidating when the
  Mojo kernel libraries change. The cache key previously content-hashed only
  the two built-in kernel packages, not the packages they import
  (`linalg`, `nn`, ...), so rebuilding after a kernel-source edit could
  silently serve a stale compiled model — for example during back-to-back
  kernel A/B benchmarking. The key now covers every Mojo binary package on
  the module import path, so kernel changes correctly trigger a recompile
  and clearing caches by hand is no longer needed.
- Fixed the structured-output grammar backend silently defaulting to
  `llguidance` instead of the intended global default `xgrammar` for models
  launched via `max serve`. The `--structured-output-backend` flag hardcoded
  `llguidance` as its default value, which shadowed the `None` "unset" sentinel
  the resolver relies on to apply the global default (`xgrammar`) or an
  architecture's pinned backend. The flag now defaults to unset, so any model
  without an explicit `--structured-output-backend` (and no architecture pin)
  correctly resolves to `xgrammar`.
- Fixed MiniMax-M3 tool-call grammar enforcement silently disabling itself
  when the model emits more than one tool-call section in a single response.
  Enforcement used to switch off once the first section closed, so a second
  section's start marker was rejected against the completed matcher
  (`Matcher rejected N token(s)…`) and the rest of the request ran
  unconstrained. Enforcement now stays on through the end of the turn: after
  the single tool-call section closes, only EOS is allowed, matching the
  model's chat template (all invocations in one section, followed
  immediately by end of turn).
- Fixed MiniMax-M3 streaming chat completions aborting with a 500 when the
  model emits a malformed tool call. The streaming tool parser now fails open
  like the non-streaming path: the raw tool-call text degrades to assistant
  content, tool parsing is bypassed for the rest of the request, and the
  stream terminates normally.
- Fixed a precision loss in the normalization ops where the `epsilon` value was
  carried in the input's dtype (for example `bfloat16`) before use. A small
  epsilon such as `1e-6` is not representable in `bfloat16`, so it was silently
  rounded. The `epsilon` for `rms_norm`, `layer_norm`, `group_norm`, and the
  fused residual, FP8-quantized, and distributed all-reduce variants is now
  carried as `float32` end to end — from the graph op through the graph
  compiler to the kernel. The Python `epsilon: float` argument is unchanged.
- Fixed MAX Serve crashing the model worker on the first host KV-cache
  offload/reload when run with `--kv-connector dkv`. The dKV connector had
  drifted out of sync with its client and no longer passed the required
  attention group on the load/offload path; it now supplies it, so the
  same-host prefix-cache path completes instead of raising.
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

- Fixed virtual-device mode on macOS. Previously the
  `max.driver.set_virtual_device_*()` settings had no effect on device
  creation: `Accelerator()` still took the real-hardware path, so requesting
  more devices than physically present failed and single-device
  cross-compilation silently used the real GPU. The virtual-device state now
  lives in a single shared library, so the setters and device creation always
  observe the same configuration on every platform.

- Fixed DeepSeek-V3.1-NVFP4 multi-token prediction (MTP) failing to load with
  `dispatch_quant_config must be specified when dispatch_dtype is not
  bfloat16` when expert parallelism was enabled. When a quantized model has no
  resolvable quantization config for its draft (BF16 NextN) weights, the draft
  config is now built with a bfloat16 dispatch dtype instead of constructing an
  invalid `EPConfig`.

## Mojo language
