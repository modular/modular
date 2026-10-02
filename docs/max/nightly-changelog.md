---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

- Added a [chat templates](/serve/chat-templates) guide that covers information
  about the `--chat-template` flag for the `max serve` CLI command.

## MAX models

- Added the MiMo-V2.6-Flash architecture (`MiMoV2ForCausalLM`) for NVFP4
  checkpoints such as `ProCreations/MiMo-V2.6-Flash-RL-NVFP4`. Its 39
  sliding-window layers use learned attention sinks, and its Q/K and V
  head dims (192 and 128) are padded to 256 for paged attention. The
  NVFP4 routed experts are repacked losslessly to MXFP4 and run with MXFP8
  activations; the dense projections are repacked losslessly to FP8 with
  128x128 block scales. The architecture needs two or more B200 (SM100)
  GPUs, since its weights (about 158 GiB) leave no room for the KV cache on
  one B200. Only NVFP4 exports load for now, not Xiaomi's FP8 checkpoint
  (`XiaomiMiMo/MiMo-V2.6-Flash-RL`).

- MiMo-V2.6-Flash (`MiMoV2ForCausalLM`) now supports speculative decoding
  with the DFlash drafter its checkpoint ships
  (`UnifiedDflashMiMoV2ForCausalLM`), with greedy and per-row sampled
  acceptance.

- GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) now serves
  `/v1/chat/completions` on 8 B200s, text-only. It pairs Kimi Delta Attention
  with sparse MLA whose indexer scores pools of four tokens rather than single
  tokens, and manifold-constrained hyper-connections in place of a plain
  residual add. Only the blockwise-FP8 checkpoint
  (`zai-org/GLM-5.3-Flash`) is supported.

- The fused Qwen3.5 speculative-decoding graph
  (`qwen3_5_with_mtp_graph`) now accepts M-RoPE positions, so speculative
  decoding composes with the vision path instead of excluding it. Without
  them the rotary fell back to a static table indexed by
  `cache_length + token_idx`, which is wrong for every token after an image:
  an image advances the position counter by its grid extent rather than by
  its soft-token count. The input covers the merged
  `[real, draft_1..draft_k]` window and is appended after the existing
  spec-decode signature, so earlier inputs keep their positions. It is
  declared only when the target runs M-RoPE, leaving text-only exports
  byte-identical, and a graph exported without it still serves text.

- Added video input to the Qwen3.5 and Qwen3-VL architectures, which
  previously rejected any request carrying a video part. Clips are sampled at
  2 fps, sized against a whole-clip pixel budget rather than a per-frame one,
  and patchified so that each patch row spans two consecutive frames. A clip
  becomes one vision encoder grid item but several placeholder runs in the
  prompt, each preceded by a `<T seconds>` timestamp label, so the model reads
  the temporal ordering from the prompt text. Send the clip's bytes in
  `videos` with a `video` content part in `messages`. An OpenAI-compatible
  request may instead carry a `video_url` part, which the server resolves and
  rewrites into that form.

## MAX framework

- Added `max.profiler.oneshot.cuda_profiler_region()`, a context manager that
  brackets a region with `cudaProfilerStart`/`cudaProfilerStop` so `nsys`/`ncu`
  capture only the wrapped region.

- The `max` CLI now allocates through jemalloc on Linux instead of glibc
  malloc. The graph compiler and the Mojo compiler run inside the CLI process,
  so the allocator the process starts with is the one they use. A cold
  `max warm-cache` compile is roughly 7 percent faster and its peak memory
  17 to 22 percent lower, measured on Llama 3.2 1B, Llama 3.1 8B and
  Gemma 3 12B on a 128-core host. The CLI restarts itself once at startup
  with the allocator preloaded, and model-worker subprocesses inherit it. Set
  `MODULAR_MAX_ALLOCATOR=system` to keep glibc malloc; sanitizer builds and
  non-Linux platforms are unaffected. Scripts that use `InferenceSession`
  directly are not affected either; to run one on jemalloc, add the installed
  `modular/lib/libjemalloc_preload.so` to `LD_PRELOAD` before starting Python.

- Added the `pre-jit` debug option (`MODULAR_DEBUG=pre-jit`,
  `[max-debug] pre-jit`, or `InferenceSession.debug.pre_jit`). It stops graph
  compilation once the Mojo for the graph has been emitted into
  `ir-output-dir`, skipping the kernel JIT, and `InferenceSession.compile()`
  raises `max.engine.CompilationStopped` in place of returning a model.
  `max warm-cache` treats that as success, so
  `MODULAR_DEBUG=ir-output-dir=<dir>,pre-jit max warm-cache ...` is a quick
  way to get the Mojo a model compiles to.

- Added the `uninitialized-read-mode` debug option
  (`MODULAR_DEBUG=uninitialized-read-mode=report`,
  `[max-debug] uninitialized-read-mode`, or
  `InferenceSession.debug.uninitialized_read_mode`). With
  `uninitialized-read-check` on, `report` prints each poisoned read and keeps
  running, so one run lists every offending load instead of stopping at the
  first. The default, `abort`, is unchanged. On Apple GPUs, `report`
  currently aborts without printing.
- Improved out-of-memory error messages for the VMM defragmenting device
  allocator. An allocation failure is now classified by cause -- genuinely out
  of memory, a shortfall recoverable from memory awaiting a stream synchronize
  or stranded in partially-used pages, memory held outside the manager's own
  accounting, virtual-address fragmentation, an unreserved arena, or a request
  that reached an unbacked region during a graph capture, where mapping is
  illegal -- and the message reports what each recovery lever may return, and
  names the setting that turns it, instead of a single often-misleading `free`
  figure. This applies to allocations that fall through to the device driver
  as well: such a failure previously surfaced the driver's refusal alone, and
  now leads with why the memory manager missed, carrying the driver's refusal
  as a note.
- `max serve` gained a `--cascade` flag that routes the request to the
  experimental Cascade server (`max.experimental.cascade.serve.main.serve`)
  instead of the standard API server + model worker. The resolved `PipelineArgs`
  is forwarded to the Cascade entrypoint, so `max serve --cascade --model ...`
  runs the Cascade server with the same model-selection flags. The Cascade
  deployment context is configurable from the same command: `--transport`
  (http/grpc), `--local-cpu-workers`, `--local-gpu-workers`, and
  `--remote-cpu-workers` / `--remote-gpu-workers` map onto `ContextConfig`.
  `--host` is now exposed for both paths and defaults `max serve --cascade` to
  `0.0.0.0`, matching `max serve`, instead of the Cascade entrypoint's
  `localhost` default.

- `max.experimental.sharding.PlacementMapping` and
  `DeviceMapping.to_placements()` are removed; use `DeviceMapping` and its
  `placements` attribute, which they aliased.

- `DeviceMesh.default()`, `DeviceMesh.is_single`, `DeviceMesh.is_simulated`
  and `DeviceMapping.to_mesh()` are removed; use `DeviceMesh.single(CPU())`,
  `mesh.num_devices == 1` and a new `DeviceMapping` on the target mesh.

- Added `max.experimental.sharding.auto_reshard` to control automatic
  resharding. It replaces `mode()`, `isolated_solver()`, `Solver`,
  `ReshardBehavior`, `GreedyReshard`, `NoReshard` and `PartialsOnly`; for
  example, `mode(NoReshard())` becomes `auto_reshard(mode="raise")`.

- `max.experimental.tensor.default_device()` accepts a `DeviceMesh` and
  replaces `max.experimental.sharding.mesh_context()`, which is removed.
  `defaults()` now returns the device as a `DeviceMesh`.

- `max.experimental.random.uniform()` and `gaussian()` accept a
  `DeviceMapping`.

- `max.experimental.nn.Module.to()` now only moves a module to one device.
  Build a multi-device module inside `default_device(mesh)` instead.

- `max.experimental.sharding` no longer re-exports `P`, `R`, `Action`,
  `PerShard`, `PerShardDim`, `Collective`, `ReduceOp`, `get_active_mesh`,
  `as_device_mapping`, `as_layout` or the `*_rule` functions.

- Added `max.experimental.sharding.Unknown`, the placement for per-device
  values with no known relation, and `Tensor.rebind_mapping()`, which
  relabels a tensor's placements without moving data. Ops on `Unknown` inputs
  run on each device's own shard and return `Unknown` results.
  `max.experimental.nn.common_layers.functional_kernels.local_map()` is
  removed: wrap the per-device function in `F.functional()` and claim its
  output's placement with `Tensor.rebind_mapping()`.
- Promoted pytree utilities out of experimental to stable `max.tree`.
- `max.experimental.nn.Module` is now a pytree: its attributes are its
  children, so `max.tree` functions such as `tree.map` and `tree.flatten` walk
  a module directly. `Module.compile` returns a
  `max.experimental.compilation.CompiledCallable`, which now also accepts a
  `Buffer` for a single-device input; the `max.experimental.nn.CompiledModel`
  wrapper and the `Module` parameter helpers it replaced (`apply_to_parameters`,
  `map_parameters`, `load_state`) are removed in favor of `max.tree`.
- Added the public `Tree` type alias in `max.tree` for nested pytree values.
  Layer subgraph inputs are annotated with `Tree[Any]` instead of the
  removed `SubgraphInput` alias.
- Every unified speculative-decoding architecture now passes its full
  `[batch_size]` per-row seed tensor to acceptance sampling, instead of only
  the architectures that opted in. Both verdicts consume it per row: the argmax
  verdict is itself a draw from the truncated target distribution — a draft
  token is accepted when it equals the sample — so the seed decides the
  committed token at every position, while `draft_proposal="sampled"` keys its
  accept coin per request and draft position. Keying each row off its own seed
  means a row's tokens no longer depend on its position in the batch, on how
  many requests share it, or on the batch's verify width. The committed
  distribution is unchanged either way, so this buys reproducibility rather
  than accuracy, and a single-row batch is unchanged.
- Device graph capture no longer aborts the model worker on AMD GPUs. With
  capture enabled, `max serve` now runs the matmul shapes that fell back to
  hipBLASLt on MAX's own kernels.
- Setting `MODULAR_DISABLE_VENDOR_FALLBACK=1` in the environment of
  `max serve` now disables the vendor-BLAS matmul fallback on any GPU.
  Previously the variable had no effect under `max serve`.
- Hardened decoding of client-supplied images. `Image.open` is now restricted
  to an explicit format allowlist (PNG, JPEG, WEBP, GIF, BMP, PPM, TIFF, TGA,
  and AVIF where the platform provides it), shrinking the native-decoder attack
  surface and keeping image bytes away from decoders such as EPS/Ghostscript.
- Media memory is now governed by a single knob, `MAX_SERVE_MAX_MEDIA_BYTES`
  (default 100 MiB). The fetched size of all resolved media (`http(s)://`,
  `data:`, and `file://` images and videos) for a request is bounded in
  aggregate by it, rather than each item being capped independently. No single
  image may *decode* to more than it either. That decoded size is estimated
  from the image header and rejected before the pixel buffer is allocated,
  which is what catches a decompression bomb (a small on-the-wire image that
  expands to hundreds of MB). This replaces both the former per-item
  `MAX_SERVE_MAX_BYTES` server cap and a separate decoded-pixel limit. It is
  deliberately separate from `MAX_SERVE_MAX_REQUEST_BYTES`, which bounds only
  the request body. A small body can name URLs the server then fetches, so
  raising one limit should not silently widen the other.
- Added dataset-agnostic image mixing to `max benchmark` (`--image-fraction`,
  `--image-count`, `--image-long-side`, `--image-aspect-ratio`, `--image-turn`),
  so any dataset (`sonnet`, `sharegpt`, `instruct-coder`, and so on) can have
  generated images mixed into a fraction of its requests or chat turns, not just
  the `random` dataset's existing fixed
  `--random-image-count`/`--random-image-size`. `--dry-run` reports an
  image-count distribution table (and a decoded image-long-side-pixel table)
  when a workload has images. See the
  [image mixing quick-start guide](https://github.com/modular/modular/blob/main/max/python/max/benchmark/benchmarking_mixed_images.md).
- `max benchmark`'s `--response-format` now reaches multi-turn workloads, which
  previously logged a warning and dropped it. A constrained request or turn runs
  without `ignore_eos`: a schema-shaped response ends where its schema is
  satisfied, and generating past that point makes the server drop enforcement
  for the remainder of the request. A chat session's running prompt length now
  charges what each turn actually generated rather than the length it drew.
- `max benchmark`'s `--response-format` now applies to a share of traffic rather
  than all of it: `--response-format-fraction` (default 1.0) sets the fraction
  of requests, or of eligible user turns, that are constrained, and
  `--response-format-turn` (`every`, `first`, `last`) chooses which turns of a
  chat session are eligible. The fraction is drawn per request and per turn, so
  at the default `every` it lands directly on the share of requests that set
  `response_format`; `first` and `last` narrow eligibility to one turn per
  session, so the realized request share is correspondingly lower.
- Added `--tools` and `--tools-fraction` to `max benchmark`, which send OpenAI
  tool definitions on any workload, multi-turn chat sessions included.
  Previously only datasets that carry their own tools sent any, and only on
  single-turn requests. `--tools` takes a `tools` list (inline JSON or
  `@file`); `--tools-fraction` (default 1.0) sets the fraction of requests, or
  of chat sessions, that carry it, and a selected session sends its tools on
  every turn. The rendered definitions are carved out of a session's first
  turn or a single-turn string prompt, so the input-length distribution still
  describes the whole prompt. Results report `tool_request_rate` and
  `tool_call_response_rate`: the share of requests that offered tools, and the
  share of those whose response called one.
- `max benchmark`'s image mixing now draws its selection from a private seeded
  RNG rather than the global one, so adding another augmentation ahead of it no
  longer changes which requests or sessions carry images. It also logs, over
  the turns it sampled, the newly-encoded images per request and the share of
  turns carrying one, the two shares a multi-modal workload is calibrated
  against.
  `augment_samples_with_images`'s `image_fraction` and `image_turn` arguments
  are renamed `fraction` and `turn` to match
  `augment_samples_with_response_format`; the `--image-fraction` and
  `--image-turn` flags are unchanged.
- `max benchmark` now reports its realized request mix in its own "Request
  Mix" section and `result_groups.request_mix`, rather than among the headline
  metrics in `result_groups.summary`. The group holds the structured-output and
  tool-calling rates, plus two new ones: `image_request_rate`, the share of
  requests whose payload carried an image (including images resent with a
  session's earlier turns), and `lora_request_rate`, the share routed to a LoRA
  adapter. Existing top-level keys in the result JSON are unchanged.
- Added a `Cat(v1:w1, v2:w2, ...)` categorical distribution for every
  `max benchmark` config field that accepts a distribution string (for
  example `--image-long-side`, `--image-count`, `--random-input-len`), so an
  explicit, empirically-measured distribution (such as image sizes measured
  from real production traffic) can be reproduced exactly instead of
  approximated with a parametric shape like `N`/`U`/`LN`. The `:weight`
  suffix is optional per entry (uniform when omitted), and weights don't need
  to sum to 1.
- Added `DeviceContext.wrap_host_memory()` (Mojo): makes a caller-owned host
  range device-accessible for as long as the returned `DeviceBuffer` lives. It
  grants access, not ownership, and the range must be addressed through that
  buffer rather than through the pointer passed in. CUDA, HIP and Metal only;
  Metal also requires a page-aligned base and a page-multiple length.
- Added an eager usage validator for `ModuleV3` and eager `Tensor` code. Can
  be enabled with a new `--eager-usage-validator` flag in the MAX CLI.
- Added `max.nn.HyperConnection` and `max.experimental.nn.HyperConnection`,
  the ModuleV2 and ModuleV3 forms of a manifold-constrained hyper-connection
  (mHC) site. The layer generalizes the residual connection: `hc_mult`
  residual streams run in parallel, and a learned gate collapses them into
  the sublayer's input and decides how the sublayer's output is written back
  across them. It owns the `hc_fn`, `hc_base`, and `hc_scale` weights, and
  returns `(post, comb, collapsed)` so the caller drives the residual update.
  Float32, GPU-only.

### Inference server

- A model with recurrent state, such as Qwen3.5 and the other hybrid
  architectures on the recurrent cache group, can now use a KV connector.
  The host and disk tiers hold each state checkpoint beside the KV blocks of
  the boundary it was taken at, and a prompt that has left the device cache
  resumes from the deepest checkpoint the tiers still hold. Configuring
  `--kv-connector-config` beside such a model used to be refused at startup.

- `--draft-proposal sampled` now works with block speculative decoding
  (DFlash, DFlash2 and DSpark drafts). The draft samples each proposal at the
  request's temperature, top-k and top-p and hands the verifier the
  distribution it sampled from, so acceptance runs true speculative sampling
  instead of typical acceptance. Previously these drafts ignored the flag and
  kept proposing their argmax.

- A speculative architecture that can't sample its draft now refuses
  `--draft-proposal sampled` at startup rather than silently drafting by
  argmax.

- Added `--tool-call-policy`, which sets how tool-call arguments are
  constrained:

  - `force_unconstrained`: no tool-call grammar.
  - `force_strict_false`: only the tool-call envelope is constrained;
    arguments are free-form, ignoring each tool's `strict` field.
  - `force_strict_true_*`: every tool is constrained to its argument schema.
  - `default_strict_true_*`: tools are constrained to their argument schema
    unless the request sends `strict: false`.
  - `default_strict_false_*` (default): tools are constrained to their
    argument schema only when the request sends `strict: true`.

  The `*` suffix is one of `and_best_effort` or `and_reject_unsupported`, which
  controls what happens to a schema that the grammar cannot enforce.

  The default, `default_strict_false_and_best_effort`, matches OpenAI's Chat
  Completions API.

- By default, Kimi, MiniMax-M2, GLM-4.7, and Gemma-4 no longer fail a request
  whose `strict: true` tool schema has unenforceable keywords; they compile it
  best-effort, unless the tool-call policy selects different behavior.

- Removed `--enable-tool-call-constrained-decode`. Use
  `--tool-call-policy=force_unconstrained` instead of setting it to false.

- Added `--prefill-coalesce-min-pending` (default 0, off): under in-flight
  batching, hold pending fresh prefills until that many can share one mixed
  step instead of admitting them one by one. With data parallelism the count
  is the total across all replicas, not one replica's queue. Mixed steps
  forfeit device graph capture for their decode rows, so coalescing admissions
  keeps more decode steps on the captured fast path, trading a bounded
  prefill-admission delay for lower decode latency at high concurrency. That
  delay is at most the same number of decode steps, or
  `--prefill-coalesce-max-held-steps` when set. Mid-prefill chunked
  continuations are never held.

- Added `--prefill-coalesce-max-held-steps` (default 0): a cap on how many
  decode steps in a row a held prefill waits before it is admitted, no matter
  how few are queued. Without it, `--prefill-coalesce-min-pending` does both
  jobs: it sets the queue depth that releases a prefill, and the number of
  steps after which one is released anyway. So the queue depth could not be
  raised without also making prefills wait longer. This flag splits the two.
  At 0 it falls back to `--prefill-coalesce-min-pending`, and it does nothing
  while that is 0.

- Added `--max-request-input-tokens` (default 0, off): a ceiling on how many
  prefill tokens one request may draw from the batch's context-encoding budget
  in a single step, applied by chunking. Without it a long prefill can claim
  the whole `--max-batch-input-tokens` budget step after step while short
  requests queue behind it, so the cap trades the long request's time to first
  token for the interactivity of the short ones it no longer blocks. It
  requires chunked prefill and is inert without it, and
  `--chunked-prefill-min-chunk-size` must not exceed it.
- MAX now honors `OTEL_SDK_DISABLED` and, for metrics,
  `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` and `OTEL_EXPORTER_OTLP_ENDPOINT`, so
  a deployment that set the latter for traces now sends metrics there too,
  and loses them if that endpoint is gRPC.

- MAX Serve can now export spans over OTLP/gRPC. Set
  `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` (or the generic
  `OTEL_EXPORTER_OTLP_PROTOCOL`) to `grpc`, and set
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to a gRPC receiver such as
  `http://<host>:4317`. Keep the `http://` scheme for a plaintext
  collector: without it the OTel SDK dials TLS. The protocol defaults to
  `http/protobuf`, so existing deployments are unchanged, and metrics and
  logs still export over HTTP.

- The scheduler's `max.phase.prefill` and `max.phase.decode` spans now nest
  under the request's `max.request` span on non-streaming chat and completions
  requests, instead of beside it or, without a `traceparent`, in a trace of
  their own.

- Added `--prefill-schedule-interval` (default 1, every step): admit prefill
  work only on every Nth scheduler step, leaving the steps in between entirely
  to decode. Data-parallel ranks advance in lockstep, so prefill on any one
  rank stalls the whole group; scattering it across steps pays that stall
  repeatedly, while concentrating it onto a shared cadence pays it once. A step
  with no decode work on any replica admits prefill regardless, rather than run
  an empty batch. The cost is delayed prefill admission, bounded at N-1 steps.
  Unlike prefill coalescing, mid-prefill chunked continuations are held too:
  the budget emits one chunk per step, so exempting them defeats the cadence.

- Structured and constrained output now uses xgrammar exclusively. The
  `llguidance` backend has been removed; `--structured-output-backend`
  no longer accepts `llguidance` as a value.

- `/v1/completions` now honors `reasoning_split`. By default the server's
  reasoning parser still hides the reasoning span from `text` and counts it in
  `usage.completion_tokens_details.reasoning_tokens`. A request that sends
  `reasoning_split: false` skips the parser, so `text` and `logprobs` cover
  every generated token, reasoning span included, as in vLLM.

- Added correlation IDs to structured log records on every route:
  `request_id`, previously always empty, and, while tracing is enabled with
  `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` set, `dd.trace_id`, except on a probe
  that arrives without a `traceparent`. Structured logging is on by default in
  the MAX container images; elsewhere, set `MODULAR_STRUCTURED_LOGGING=1`.

- Populated `batch_id` in structured log records that a text-generation model
  worker logs during a forward pass while tracing is enabled. It was previously
  always empty, and still is on disaggregated prefill and decode workers.

- Added `--adaptive-speculative-widths` to pick how many drafts
  speculative decoding verifies by measured decode tokens per second,
  instead of a fixed width. Takes a list such as `1,3,5`, or `all`. Each
  width adds graph capture time at startup.

- With `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` set, MAX Serve starts an HTTP
  server span for each request except health, version, ping and metrics
  probes, such as `/v1/health`. It continues any inbound `traceparent` and
  parents the request's `max.request` span, so the request's logs carry a
  trace ID without a `traceparent`. It ends when the response body finishes,
  so a streamed response's span covers the whole stream.

### Server metrics

- Added counters for how much traffic uses tool calling and structured
  output: `maxserve.tool_call.requests` (the request declared tools, tagged
  `choice`), `maxserve.tool_call.responses` (its response actually contained
  a tool call), and `maxserve.structured_output.requests` (tagged `kind`).
  The first two together show how often a declared tool inventory is used.
- Added `maxserve.tool_call.tools_per_request`, a histogram of how many tools
  a request declared. Tool schemas are rendered into the prompt, so this is
  the explanatory variable behind a client's prompt length and grammar
  compile cost.
- Added the `maxserve.media.*` family, which accounts for the media work a
  multimodal request pays for before it reaches the model. Volume and origin
  come from `maxserve.media.items` (tagged `media_kind` and `source`),
  `maxserve.media.items_per_request` and `maxserve.media.item_size`; the cost
  of turning a reference into bytes from `maxserve.media.resolve_time`; the
  codec cost from `maxserve.media.image_size` and the
  `maxserve.media.image_decodes` / `maxserve.media.image_decode_ms` counter
  pair, both tagged `format`, whose quotient is the mean decode cost for a
  format; and refusals from `maxserve.media.rejections` (tagged `reason`).
  Cache pressure comes from `maxserve.media.preprocess_cache_evictions`,
  `maxserve.media.preprocess_cache_size` and
  `maxserve.media.preprocess_cache_capacity` (all tagged `media_kind`), which
  separate a low preprocess-cache hit rate caused by a workload with no
  repeats from one caused by a cache thrashing at its byte budget. The
  resolve and decode time is inside `maxserve.time_to_first_token`, where it
  was previously unattributable.
- `maxserve.vision.preprocess_cache_hits` and
  `maxserve.vision.preprocess_cache_misses` are now recorded on every
  architecture with a preprocess cache, not only on those the API server can
  ask at admission whether an image is already preprocessed. Elsewhere the
  pair sat at zero. On those other architectures it comes from the
  tokenizer's own cache lookup after tokenization rather than from the
  admission peek, so the two windows differ and the rate isn't comparable
  across architectures.
- Added `maxserve.cache.connector_loads_refused` and
  `maxserve.cache.connector_offload_blocks_dropped` for the tiered KV
  connector. The first counts loads its host and disk tiers refused, each
  served by recomputing the blocks instead. The second counts offloaded
  blocks the host pool had no room for, which rises when blocks pinned for
  in-flight transfers starve the pool, before the hit rate falls.

### `max` CLI

### Python API

- Added `max.experimental.custom.declare`, which declares a custom op's
  input and output signature once, as an immutable `custom.CustomOp`, and
  runs eagerly or inside a `Graph`. Output dims may be symbolic,
  parameterized, or left to the kernel's shape function to determine at run
  time.
- `max.experimental.custom.declare` now raises `ValueError` if the op's own
  `custom_extensions` (or the process-wide defaults) don't register the
  kernel.
- `max.nn.moe.StackedMoE` can now run MXFP4 experts W4A8 on SM100 GPUs,
  including under tensor parallelism: pass `mxfp8_activations=True` with an
  MXFP4 `quant_config`. The expert scales are then declared in the grouped
  matmul's interleaved layout (see
  `max.nn.moe.interleaved_block_scales_shape`). The new `router_dtype` and
  `combine_dtype` options run the router and the weighted combine in float32.
- Added `max.nn.moe.SigmoidTopKRouter`, the sigmoid top-k router with an
  expert score correction bias (`noaux_tc` with one expert group) that
  MiniMax-M2, HY-V3, and MiMo-V2 share.

### C API

## Kernels and GPU programming

- Reworked the `gated_delta_conv1d_fwd` GPU kernel (Gated DeltaNet
  causal conv, Qwen3.5/3.8 and GLM-5/Kimi-Delta prefill pass 1) from
  one thread per channel walking the sequence serially to one thread
  per token tile. A 2048-token prefill chunk at the Qwen3.8 TP2 conv
  width drops from ~650 us to ~32 us per layer, and 8192 tokens from
  ~4.3 ms to ~112 us, bit-identically, with decode shapes unchanged.

- The KDA / gated-DeltaNet chunk prefill now routes the production
  configuration (fp32 output, bf16 `K_FIRST` state pool, original softplus
  gate, probability beta, GVA head grouping on SM100) to the two-kernel bt16
  pair instead of the fused kernel: a prep pass materialises gate-rebased
  tiles per chunk group, then a chain kernel walks chunks per
  (sequence, value-head) with tcgen05 MMA and drains fp32 output directly.
  At the Qwen3.5 prefill shape (16 key / 48 value heads, 128 dims, 2048
  tokens per forward) the op drops from 0.59 ms to 0.23 ms per layer, and the
  fused kernel remains the fallback for other configurations.

- Added `max.nn.state_space.kda_chunk`, the chunk-blocked form of the KDA /
  gated-DeltaNet recurrence, alongside the existing `kda_decode`. Its launcher
  can now reach the fused Blackwell kernel instead of only the three-stage scan
  pipeline: the selection guard asked `std.sys.info` whether the compile target
  was an SM100 GPU, which is false in a host-side launcher, so the fused branch
  had been compiling as dead code. At Qwen3.5 geometry the launcher runs 1024
  tokens in 0.315 ms against the pipeline's 1.853 ms. The fused path is opt-in
  through a compile-time parameter and off by default.

- `TileTensor` gained `as_span()`, returning a `Span` over the tensor's
  elements.

- `TileTensor` gained `unsafe_ptr()`, returning the raw scalar `Pointer` its
  engine exposes for the tensor's storage. Element loads and stores on the
  tensor go through that pointer.

- Added `max.nn.kernels.keyed_uniform`, which draws one uniform value in
  `[0, 1)` per row of a seed tensor. Every row is its own Philox key, so a
  row's value is a function of its seed alone, not of the row's position or of
  what else shares the launch. `ops.random.uniform` keys the whole tensor off
  index 0 of the graph seed and walks the flat element index as its Philox
  counter, so it cannot express that. Speculative decoding's accept coin uses
  it to draw one uniform per (request, draft position).

- `TileTensor.slice` now accepts an `Int` in place of a slice literal, which
  fixes that dimension to the given index and drops it from the result, the
  way `squeeze` would. The output rank is the number of slice arguments, so
  `t.slice[1:3, 2, 0:4]()` returns a rank-2 view of a rank-3 tensor. Any axis
  may be dropped, not only trailing ones, and each surviving axis keeps its
  own stride. Bounds stay compile-time and are now range-checked against the
  parent's static shape at compile time. A view of a fully static tensor
  remains fully static: its extents and strides are `ComptimeInt`, and the
  base offset is carried in the view's engine as a `ComptimeInt` rather than
  computed at runtime.

- `TileTensor` now slices through subscript syntax: `t[batch, 0:n, :]` fixes
  the batch axis, narrows the next one to a runtime subrange, and keeps the
  last whole. Every dimension takes exactly one argument. A subscript is a
  view whenever one of its arguments is a slice, and an element load
  otherwise, so existing indexing is unchanged. Compile-time and runtime
  indices share the path: an `Idx[n]` index on an axis with a compile-time
  stride is folded into a `ComptimeInt` component of the view's offset, and
  only the rest is computed at runtime. Slice bounds are runtime values, so
  a sliced extent is runtime too -- `:` means `0:dim`, not a marker. Strides
  are always inherited whole.

- Deprecated `TileTensor.as_immut()` in favor of `TileTensor.as_imm()`, which
  returns the same immutable view and matches the naming of `Pointer.as_imm()`.

- Added `max.nn.kernels.hyper_connection_gates`, a fused kernel for the
  Manifold-Constrained Hyper-Connections (mHC) gate computation. It replaces
  the sigmoids, the softmax, and the Sinkhorn-Knopp projection an mHC site
  runs between its stream projection and its stream collapse -- roughly 40
  small elementwise and reduction launches at 20 Sinkhorn iterations -- with
  a single launch that keeps the whole `hc_mult` x `hc_mult` mixer in one
  warp's registers. Float32, GPU-only. It supersedes
  `max.nn.kernels.mhc_split_sinkhorn`, which is removed.

- `max.nn.kernels.grouped_matmul_block_scaled` and
  `max.nn.kernels.grouped_matmul_blocked_swiglu` accept an optional
  `a_row_scales`: one `bfloat16` scale per activation row, multiplied into
  that row's output together with its expert scale (before the SwiGLU in the
  fused op). It dequantizes NVFP4 activations that were quantized with a
  per-token tensor scale. NVFP4 on SM100 only.

- Added `layout.tmem_engine.TMemEngine`, a `TensorEngine` that lets a
  `TileTensor` view Blackwell Tensor Memory (TMEM). A TMEM tile's layout places
  elements on the 128-lane by 512-column grid lane-first: a lane stride of `1`
  and a column stride of `TMEM_NUM_LANES` (`128`), so the whole accumulator is
  `(128, 512):(1, 128)` and nested layouts express the lane placement of other
  MMA shapes. The engine encodes grid positions into the hardware's
  lane-and-column addresses itself. TMEM has no pointer, so the engine's data
  path is `TileTensor.copy_from` in either direction: a copy between a TMEM row
  and a register or shared-memory tile moves consecutive columns of the lane the
  calling thread owns through the warp-collective `tcgen05.ld` or `tcgen05.st`
  in the `32x32b` shape, one instruction per power-of-two chunk of at most 64
  registers, and one wait per 64-column slice. The `copy_from_async` and
  `copy_to_async` engine methods and the tile-level `tmem_copy_async` issue the
  same instructions without waiting, so several copies can share one
  `TMemEngine.wait_store` or `wait_load`; an async row is capped at 64 columns,
  so a wider row is tiled into one copy per slice. A thread's view of a warp's
  `(32, N)` tile is its row 0, since the hardware adds the lane to the warp base
  address. The engine supports 4-byte element types and requires an SM100
  target.

- Added `TensorEngine.copy_to`, the source side of a copy.
  `TileTensor.copy_from` now calls it on the source tensor's engine, and its
  default forwards to the destination engine's `copy_from`, so existing
  engines are unaffected. An engine whose storage has no pointer, such as
  `TMemEngine`, overrides it to run its own load loop.

## Breaking changes

- `max.gpu.primitives.warp.reduce()` and `lane_group_reduce()` now take the
  reduction function as a runtime closure argument instead of a compile-time
  `capturing` parameter. The shuffle function stays a compile-time parameter:

  ```mojo
  def add[dtype: DType, width: SIMDLength](
      x: SIMD[dtype, width], y: SIMD[dtype, width]
  ) -> SIMD[dtype, width]:
      return x + y

  warp.reduce[shuffle_down](val, add)  # was: warp.reduce[shuffle_down, add](val)
  ```

- Text generation now stops on every `eos_token_id` in a model's
  `generation_config.json`, as Hugging Face `generate` does, in addition to
  the tokenizer's EOS token and the `eos_token_id` in `config.json`. This
  applies to `TextTokenizer`, `TextAndVisionTokenizer`, and the
  architecture tokenizers built on them. Before, those extra ids were only a
  default for `stop_token_ids`, so a request that set its own
  `stop_token_ids`, or that bypassed the model's sampling defaults, could
  generate past them. Models whose `generation_config.json` lists more EOS
  ids than `config.json` (Llama 3.1 Instruct, for example) may now end some
  responses sooner. `stop_token_ids` still adds to the set, and
  `ignore_eos` still disables all of it.

- `TensorEngine` drops its `load` and `store` requirements, along with the
  `DefaultEngine`, `DevicePointerEngine`, and `StaticOffsetEngine`
  implementations. Loads and stores go through the raw pointer `unsafe_ptr`
  returns: `Engine.load[width=w, alignment=a](storage, offset)` becomes
  `Engine.unsafe_ptr(storage).load[width=w, alignment=a](offset)`, and
  `TileTensor.unsafe_ptr()` exposes the same pointer at the tensor level.

- `TileTensor`'s two runtime slicing methods are replaced by subscript
  syntax, and the `All` marker they used is removed along with them.
  `t.slice(batch, All, All)` becomes `t[batch, :, :]` and
  `t.slice((0, h), (0, w))` becomes `t[0:h, 0:w]`. `All` was a `CoordLike`
  that stood for a dimension rather than a coordinate, which is why `Coord`
  carried a `contains_slices` member to reject it; both are gone. Where the
  indices are compile-time values, prefer `slice[]()`, which keeps the view
  fully static.

- `MAX_SERVE_TRACE_PREFILL_BATCH` is renamed `MAX_SERVE_TRACE_BATCH` and now
  brackets the model-execute call on the decode scheduler as well as prefill.
  A dispatch line with no matching completion identifies which worker is stuck
  inside one call, which the prefill-only form could not do for a decode-side
  stall. Log lines name the role, as `Dispatching decode batch: ...`, and
  an execute that raises closes its bracket with `Failed ... batch` so it
  is distinguishable from one that never returned.

- `KVCacheMetrics` drops `nixl_read_blocks_local`, `nixl_read_blocks_remote`,
  and the `remote_read_ratio` property computed over them. No code path ever
  populated either field, so the ratio returned `0.0` for every caller. Six
  populated counters are added in their place: `dkv_peer_attaches`,
  `dkv_peer_attach_failures`, `dkv_peers_dropped`, `dkv_peer_loads`,
  `dkv_peer_load_failures`, and `dkv_hints_rejected`. They are zero unless the
  external KV-cache connector is in use.

- `max.driver.DevicePinnedBuffer` is removed. Allocate
  `Buffer(dtype, shape, device=gpu, usage=Usage.STAGING | Usage.UNTRACKED)`
  instead; `UNTRACKED` keeps the old behavior, where reads never wait on
  device copies. Replace `isinstance(b, DevicePinnedBuffer)` with `b.pinned`
  to check for page-locked memory, or `Usage.UNTRACKED in b.usage` to check
  whether reads skip the hazard wait.

- MAX Serve now exports spans only when `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
  is set. It no longer sends them to Modular's collector by default, and
  `OTEL_EXPORTER_OTLP_ENDPOINT` alone no longer turns tracing on. To keep
  exporting spans, set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, for example to
  `http://collector:4318/v1/traces`.

- The `//` operator on `TensorValue` and `max.experimental.tensor.Tensor`
  now matches `ops.floor_div`: floor division of two integer tensors returns
  an integer tensor of the operand dtype instead of `float64`. Float operands
  are unaffected. Code that relied on the float result should cast
  explicitly, for example `(x // n).cast(DType.float64)`.

- MAX now builds against NIXL 1.5.0, whose libfabric (EFA) transport adds a
  mandatory connection handshake that a NIXL 1.3.0 peer never answers. A MAX
  built on 1.5.0 cannot exchange KV cache transfers over libfabric with one
  built on 1.3.0, so disaggregated prefill and decode have to be upgraded
  together.

## Fixes

- `ops.floor_div` on mixed integer dtypes such as `uint8 // int16` now
  applies the floor correction for the promoted signed dtype, so `7 // -2`
  returns `-4` instead of `-3`.

- Fixed `max serve` with `MAX_SERVE_KERNEL_TRACE_LEVEL=kernel` never writing
  its libkineto kernel trace. The trace is now written when the server stops,
  provided the model worker shuts down within its 5 second grace period.

- `max.experimental.functional.relu()` now runs eagerly on float `Tensor`
  inputs.

- Fixed a regression where indexing a buffer -- loading it and then gathering
  rows out of it, as a paged KV cache does -- allocated and copied the entire
  source buffer on every execution instead of reading only the rows requested.
  For a large buffer this inflated memory use and could fail with an
  out-of-memory error. The load now folds into the gather so it reads the
  buffer directly.

- Fixed a prefill worker crash on `response_format` JSON schema requests under
  disaggregated serving. The decode worker admits the request and forwards it
  to prefill, but a prefill worker launched without `--enable-structured-output`
  rejected the schema from inside its forward pass and exited, which then
  stalled decode. Prefill never enforces a grammar now: the decode worker
  discards prefill's token for a constrained request and samples the first
  token under its own matcher, so the schema is still honored.

- Fixed a crash serving Gemma 4, Gemma 3 multimodal, Pixtral, and Qwen 3
  embedding with a batch larger than one. Each of these architectures accepted
  `max_batch_size` and never handed it to `PipelineModel`, which left the base
  at its default of 1 while the scheduler dispatched full batches. The model
  input buffers a batch stages into are sized from that value, so the first
  multi-context step overran them and the model worker crashed. The
  architectures pass it up now.

- Fixed `max generate` crashing after the first token for Gemma 4 and
  Idefics3 models, whose tokenizers rejected the CLI's token list on decode.

- Fixed the same `max generate` decode crash for the other vision-language
  models: Gemma 3 multimodal, InternVL, Kimi K2.5, Pixtral, Qwen2.5-VL and
  Qwen3-VL.

- The functional kernel wrappers in `max.experimental.nn.common_layers` now
  open a realization context, so eager attention with a paged KV cache runs.

- Fixed streaming `/v1/completions` sending an unreadable frame, or ending
  the response abruptly, when a request failed part-way through a stream.
  The error now arrives as an `error` object the OpenAI client surfaces as
  an `APIError`, matching `/v1/chat/completions`.

- Fixed `sampling_params.seed` not reproducing. The batch-slot fix above
  briefly salted each request's RNG key with a hash of its request id, which
  the server mints fresh per HTTP request, so two identical requests carrying
  the same pinned seed drew different tokens and a replayed request never
  reproduced its own output. The key is the seed and the generated-token count
  again; requests that pin the same seed and have generated the same number of
  tokens now draw in lock step, which is what pinning a seed asks for.

- Fixed sampled tokens depending on which batch slot a request occupied. Every
  draw from the fused token sampler — ordinary decode as well as speculative
  verification — mixed the physical batch position into its RNG counter, so a
  request that was preempted and re-admitted into a different slot, or that
  simply shared a step with a different set of requests, drew a different token
  from an unchanged seed. A request's RNG key is now derived from its own seed
  and generated-token count alone, and the batch position no longer reaches the
  sampler at all. No distribution
  changes, but the exact token emitted for a given seed does move, so output
  pinned against a previous build will differ. Requests that pass the same
  `seed` stay independent of one another, which was previously true only on one
  of the three sampling routes.

- Fixed a negative-bound slice of a concatenated value silently returning that
  value rotated instead of the region asked for. When two slices together cover
  one axis, the graph compiler rewrites them into a single `split`, whose
  results are consecutive chunks in order. It ordered those chunks by the raw
  start constant, but slicing takes numpy-style bounds, so a negative start
  counts back from the end of the axis: `x[..., -8:]` carries a start of `-8`,
  which sorted ahead of a slice starting at `0` and swapped the two regions.
  Writing `concat(x[..., :-rope_dim], x[..., -rope_dim:])` over a value built by
  a `concat` therefore produced that value rotated left by `rope_dim`, on CPU
  and GPU alike, with no error and with every operator individually correct.
  Slice bounds are now normalized against the axis before the chunks are
  ordered, and a set of slices that does not actually tile the axis is left
  alone rather than rewritten.
- Fixed the tiered KV cache connector leaking its `max_kv_tiered_*` disk
  offload directory on almost every shutdown. Deleting it relied on the model
  worker unwinding cleanly, which it never does: the worker is stopped with
  `SIGTERM`, so its cleanup hooks were skipped and each run left its offload
  tree behind. An auto-created offload directory now holds an exclusive lock
  for as long as its server lives, and startup deletes any such directory no
  live process still holds — so a directory abandoned by a `SIGTERM`,
  `SIGKILL`, or OOM-kill is reclaimed on the next start instead of being
  reported to an operator. Directories left by earlier versions carry no lock
  and are still only reported, not deleted.
- Fixed a decode-engine crash in disaggregated (prefill/decode) serving under
  memory pressure. When the decode node could not allocate KV blocks for a
  new request whose prompt partly hit the prefix cache, it re-queued the
  request with its token window still advanced past the cached prefix while
  handing the cached blocks back. If those blocks were evicted before the
  retry, the block manager asserted (`Expected at least N blocks to store KV
  for M committed tokens, but only 0 are assigned`) and the model worker
  died. A KV cache allocation that fails now leaves the request exactly as it
  found it, in both the paged and the Jenga cache managers.
- Fixed a tool whose parameter schema used `const` or `enum` with an object or
  array value producing an uncompilable grammar, for models whose tool-call
  format frames arguments in XML markers rather than JSON (GLM, MiniMax, Qwen,
  DeepSeek). The literal's own JSON quotes were read as grammar syntax, so the
  request failed with `Grammar provided in request cannot be compiled`. Such a
  literal now keeps its JSON form inside the value markers. The tag-keyed
  formats, which spell an object as nested tags and so have no JSON form to
  fall back on, reject it with an explicit message instead.
- Fixed a GLM tool-call argument pinned by `const` or `enum` to a string that
  reads as a number or a boolean (`"2.0"`, `"true"`) being reported as that
  number or boolean. GLM emits argument values bare, so the parser recovers the
  type from the schema, and it consulted `type` and the string facets but not
  `const` or `enum` — neither of which usually carries a sibling `type`. A
  conforming tool call therefore read back as a schema violation.
- Fixed a structured-output `integer` field being unable to accept a number
  written with an exponent, such as `1e80`. JSON Schema reads an integer as a
  number with a zero fractional part, so `1e80` is one, and it is how a model
  writes a very large integer — but the grammar's integer terminal had no
  exponent suffix, so the `e` was masked out mid-number and the model was
  steered into the nearest legal continuation (`1e80` became `180`). The
  terminal now admits an exponent whose sign is `+` or absent. A negative
  exponent stays rejected, since `1e-5` is a legal JSON number but not an
  integer. This affects `response_format` schemas as well as tool calls.

- Fixed identical tokens producing different reduce-scatter sums depending
  on which batch slot they occupied. The P2P reduce-scatter kernel rotated
  the order in which it summed its peers' buffers by the destination rank,
  to stagger NVLink traffic. A reduce-scatter's destination rank is a
  function of the row index, so the same token summed in a different
  order from one shard to the next -- a legal float reassociation, but one
  bfloat16 ULP of difference that downstream layers amplified into
  diverging sampled output when two identical requests shared a batch at
  different slots (measured on MiniMax-M3 TP4: 113 of 150 greedy decode
  steps disagreed at temperature 0). Every destination now accumulates its
  peers in the same canonical rank order, in the standalone kernel, the
  fused reduce-scatter + RMSNorm kernel, and the grouped relay path, so an
  element's sum depends on its inputs alone. Outputs of existing
  multi-GPU batches may shift within rounding noise; the add order was
  never a contract.

- Fixed Qwen3.5 returning fluent nonsense on every prompt, text included.
  Its linear-attention layers all read and wrote one shared recurrent state
  row rather than one each, and nothing about that was visible from outside,
  so the model loaded and served as usual.

- Fixed the scheduler refusing to admit queued prefill on models with a
  recurrent state cache while KV cache memory was still free.

- Fixed `top_k` on Apple GPUs returning stale memory past the 32nd output of a
  row that a single threadgroup reduces, which a large `k` forces. Tied values
  in rows of up to 2048 elements also come back smallest index first now, as on
  other GPUs, instead of grouped by threadgroup.

- Fixed KV cache transfers over NIXL on InfiniBand hosts aborting the process
  when a NIXL agent shut down, on glibc's `mutex->__data.__owner == 0`
  assertion. The CUDA and ROCm verbs builds of the NIXL UCX plugin bound part
  of the mlx5 API to an outdated compatibility ABI, so UCX's completion queue
  doorbell writes landed on a mutex inside libmlx5. The plugins now link
  libmlx5 and bind its current ABI.

- Fixed PyTorch raising `HIP error: peer access is already enabled` on AMD GPUs
  in a process that had set up MAX across several GPUs more than once, for
  example by creating a second multi-GPU `InferenceSession`. MAX accepted the
  repeat as success but left HIP's last error set, and PyTorch reported it from
  its next kernel launch. MAX now clears that error.

- Fixed `max.phase.prefill` spans never ending and `max.phase.decode` spans
  never starting, so with tracing enabled neither was exported for a completed
  request and the model worker kept every such request's prefill span in
  memory.

- Fixed `DeviceContext.execution_time()` and `execution_time_iter()` on Apple
  GPUs reading the host clock without waiting for the timed work, so GPU
  benchmarks on Metal reported enqueue time instead of execution time. They now
  return the GPU time between the start and stop points, as on NVIDIA and AMD
  GPUs.

## Mojo language
