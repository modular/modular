---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Fixed unbounded host-memory usage in Gemma 4 video pre-processing: the
  server now decodes only the sampled frames of a video instead of
  materializing every frame before sampling, bounding peak memory at the
  sampled frame count (previously a long clip could transiently allocate
  ~100 GB in the API server process).
- Added GLM-5.2 (`GlmMoeDsaForCausalLM`) support, extending the GLM-5.1
  sparse-attention architecture with cross-layer index sharing.
  - Added multi-token prediction (MTP) speculative decoding for GLM-5.2
    (`UnifiedMTPGlm5_2ForCausalLM`), serving the baked-in NextN layer as a
    single-layer sparse-MLA draft; enabled automatically for GLM checkpoints
    that ship a NextN layer with `--speculative-method mtp`.
  - Added tool-calling, reasoning, and structured-output (`response_format`)
    support to GLM-5.1 / GLM-5.2, enabled with `--tool-parser glm45
    --reasoning-parser glm45 --enable-structured-output`.
  - Fixed a GLM-5.1-FP8 crash caused by a shared-experts dtype mismatch.
  - The GLM-5.2 B200 recipe now serves the checkpoint's full 1M-token
    context window (`max_length: 1048576`, previously pinned to 163840).
    The pin existed because the wider window cost ~33% decode throughput on
    long-context workloads; the sparse-attention indexer now does work
    proportional to actual sequence lengths (per-layer kernel cost measured
    flat across frozen bounds), and a weekly long-context serving benchmark
    tracks the end-to-end throughput at this configuration.
- Added multi-token prediction (MTP) speculative decoding for Inkling
  (`UnifiedMTPInklingForConditionalGeneration`), serving the checkpoint's
  chained dense draft depths; enabled automatically for Inkling checkpoints
  that ship `mtp_config` with `--speculative-method mtp`.
- Added Laguna (`LagunaForCausalLM`) support for
  `poolside/Laguna-M.1-NVFP4`, including tool calling.
- Added DiffusionGemma (`DiffusionGemmaForBlockDiffusion`) support for
  `google/diffusiongemma-26B-A4B-it` (bfloat16) and
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4`; text-only for now.
- Added Nemotron-H (`NemotronHForCausalLM`) support, NVIDIA's hybrid
  Mamba-2 + attention decoder, with modelopt per-tensor FP8 and a new
  Mamba-2 SSD chunked-scan varlen kernel.
  - Extended Nemotron-H with the Nemotron-3-Nano-30B-A3B hybrid MoE variant
    and enabled the architecture on Apple silicon GPUs in bfloat16.
  - Enabled NVIDIA's official FP8 Nemotron-H checkpoints on Apple silicon
    (previously crashing or producing all-zero logits) and sped up
    Nemotron-H decode on Apple M5 by ~41-81%.
  - Added support for serving `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` on
    Apple silicon via a tiled simdgroup-MMA grouped-FP8 (W8A16) MoE matmul,
    decoding faster than bf16 at concurrency with half the weight memory.
- Fixed the `max_batch_size` handling for Nemotron-H.
- Added support for the `detail` parameter on image and video content
  parts in chat requests.
- Added Ideogram 4 (`Ideogram4Pipeline`) support, a text-to-image
  flow-matching diffusion transformer; serve via `/v1/responses`.
  - FP8 checkpoint weights run hot projections on native FP8 GEMMs (~24%
    faster end-to-end on MI355).
- Added support for `amd/Kimi-K2.7-Code-MXFP4` on AMD GPUs.
- Expanded Gemma 4 support:
  - Added DSpark speculative decoding for Gemma 4 12B
    (`UnifiedDSparkGemma4ForCausalLM`), DeepSeek's block-drafting method:
    a small draft transformer drafts a 7-token block per step. Enabled with
    `--draft-model-path deepseek-ai/dspark_gemma4_12b_block7
    --speculative-method dflash --num-speculative-tokens 7`.
    - Fixed the draft applying full rope instead of the checkpoint's
      partial rotary factor (0.25), which was costing roughly 10% of the
      draft acceptance rate.
  - Added DSpark speculative decoding for Gemma 4 31B
    (`UnifiedDSparkGemma4_31BForCausalLM`), serving `google/gemma-4-31B-it`
    with the vLLM speculators-format draft
    `RedHatAI/gemma-4-31B-it-speculator.dspark` (llama-style causal draft
    block, pruned 32k draft vocabulary mapped through the checkpoint's d2t
    table). Enabled with the `gemma4_31b_dspark.yaml` recipe or
    `--draft-model-path RedHatAI/gemma-4-31B-it-speculator.dspark
    --speculative-method dflash`. An explicit `--num-speculative-tokens` is
    honored: values below the trained 7 truncate the causal draft block
    prefix-stably, values above run as extrapolation with a warning and
    degrading acceptance; unset defaults to the trained 7.
  - Added DFlash speculative decoding for Gemma 4 31B
    (`UnifiedDflashGemma4_31BForCausalLM`), serving `google/gemma-4-31B-it`
    with the z-lab block-diffusion drafter `z-lab/gemma-4-31B-it-DFlash`: a
    5-layer noncausal draft block drafts 15 tokens per step from six target
    hidden-state taps. Enabled with the `gemma4_31b_dflash.yaml` recipe or
    `--draft-model-path z-lab/gemma-4-31B-it-DFlash --speculative-method
    dflash`. The draft width is pinned to the drafter's trained
    `block_size - 1`; a mismatching `--num-speculative-tokens` is overridden
    with a warning. NVFP4 target checkpoints (`nvidia/Gemma-4-31B-IT-NVFP4`)
    are supported via the `gemma4_31b_dflash_nvfp4.yaml` recipe.
  - Gemma 4 31B DSpark now supports structured output (JSON schemas and
    tool-call grammars, enforced on the target verify pass; a
    grammar-violating draft is rejected at its position) and Gemma 4
    thinking: reasoning content is split out of responses, and relaxed
    acceptance during the thinking phase can be enabled with
    `use_relaxed_acceptance_for_thinking`.
  - Renamed the Gemma 4 12B DSpark architecture to
    `UnifiedDSparkGemma4_12BForCausalLM` (module
    `max.pipelines.architectures.unified_dspark_gemma4_12b`), so the two
    Gemma 4 DSpark architectures are named by model line.
  - Sped up Gemma4-12B DSpark decode by up to ~1.3x via a packed wide-N
    shallow-K GEMV, a single-pass streaming argmax kernel, and device graph
    capture.
  - Gemma 4 with MTP speculative decoding (`UnifiedMTPGemma4ForCausalLM`)
    now supports image and video input; previously the vision encoder output
    never reached the language model, so image prompts were answered as if
    the model were blind.
  - MTP speculative decoding now samples recovered tokens from the residual
    distribution when stochastic acceptance rejects a draft token,
    preserving the target distribution for argmax draft proposals.
  - Added structured-output and tool-calling support via the xgrammar
    backend, covering Gemma 4's special tool-call format.
  - Added float16 support, with the logit softcap and vision pooler run in
    fp32.
  - Added tensor-parallel support for the MoE variant.
  - Video inputs now route through the shared `VisionEncoderCache`, so a
    repeated clip is served from cache with no re-encode.
  - Video decoding now runs on a worker thread, so concurrent requests
    overlap video decode.
  - Improved vision-batch serving latency by concatenating embeddings
    on-device instead of round-tripping through host numpy.
  - Fixed the MoE expert-router softmax being computed in `bfloat16`
    instead of `float32`, which degraded MoE quality.
  - Fixed image/video position and scatter indexing desyncs under chunked
    prefill, which could corrupt vision embeddings on multimodal prompts
    split across chunks.
  - Fixed crashes in multi-device serving and multi-image batches by making
    `merge_per_device_buffers` rank-agnostic.
  - Fixed reasoning being dropped after tool results.
  - Fixed a vision-batch crash caused by constructing a `Device()` instead
    of `CPU()` for host tensors.
- Expanded DeepSeek-V3 ModuleV3 support:
  - Added NVFP4 (modelopt) weight support, running experts, dense MLPs,
    and the attention output projection on SM100 block-scaled FP4 matmul
    kernels.
  - Added data-parallel + expert-parallel (DP-EP) and multi-GPU
    tensor-parallel + expert-parallel (TP+EP) serving. Note: `Tensor.to`
    no longer implicitly calls `F.distributed_broadcast`; call it
    explicitly where needed.
  - Fixed the FP8 adapter by casting f32 normalization gammas, resolving a
    dtype mismatch.
- Expanded Kimi K2.5 support:
  - Kimi with DFlash speculative decoding
    (`UnifiedDflashKimiK25ForCausalLM`) now supports image input;
    previously the vision encoder was not compiled, so image prompts were
    answered as if the model were blind.
  - Added support for combining Kimi tool calling with
    `response_format=json_schema` on the xgrammar constrained-decoding
    backend.
- Expanded FLUX.2 support:
  - FLUX.2-klein bf16 checkpoints on Apple M5 GPUs now default to int8
    W8A8 quantization, ~1.45x faster end-to-end than bf16 on
    FLUX.2-klein-4B at near-lossless quality; set
    `APPLE_FLUX2_INT8_W8A8=0` to opt out.
  - NVFP4 checkpoints can now opt into an int8 W8A8 requant at load on
    Apple M5 with `APPLE_FLUX2_INT8_W8A8=1`, ~2.56x faster end-to-end
    than the default W4A16 path on FLUX.2-dev.
  - Diffusion pipelines now support two denoising-cache backends to skip
    redundant transformer passes: `--taylorseer` (recommended default,
    with `balanced` and `fast` presets) and `--first-block-caching`; the
    two are mutually exclusive and both off by default.
- Expanded Qwen support:
  - Added tool-calling and reasoning support to Qwen 3.5 / 3.6.
  - Added `Qwen/Qwen3.8-27B` support in bfloat16 on the existing
    `Qwen3_5ForConditionalGeneration` architecture, covered by logit
    verification against the torch reference.
  - `Qwen3_5ForConditionalGeneration` now serves across multiple GPUs.
    Tensor parallelism splits the attention heads, the gated-DeltaNet key and
    value heads, and the per-device linear-attention state pools; both mixers
    reject a device count that would not divide their head counts evenly.
  - `Qwen3_5ForConditionalGeneration` now supports device graph capture.
  - Added multi-token prediction (MTP) speculative decoding for Qwen3.8
    (`UnifiedMTPQwen3_5ForConditionalGeneration`), fusing the target, the
    baked-in MTP head and a recurrent-state rollback into one graph, selected
    for Qwen3.5-family checkpoints that ship an MTP head with
    `--speculative-method mtp`. Rejecting a speculated token cannot be undone
    by rewinding a KV length pointer when the layer is recurrent, so the graph
    snapshots the gated-DeltaNet conv and recurrent pools before verifying and
    replays the two state kernels over the accepted rows. The graph is served
    through the Mach engine; MAX compiles and exports it but does not run it.
  - Fixed a `Qwen3EmbeddingModel` crash.
- Added `--state-pool-dtype`, which overrides the storage dtype of a hybrid
  model's recurrent state pools (SSM and linear-attention conv and recurrent
  state). It defaults to the model's compute dtype. `float32` makes a
  speculated generation follow the same state trajectory as an unspeculated
  one -- the recurrence rounds to the pool dtype at each call boundary, so a
  lossy pool makes the trajectory depend on how speculation chunked the
  sequence -- at roughly double the per-request state memory (Qwen3.8-27B:
  74.8 to 149.6 MiB per seated request).
- Added per-request LoRA adapter support: `LoRALinear` and
  `StackedLinearLoRA` extend LoRA to standalone and fused-QKV projections,
  with `LoRAManager.apply` swapping target layers in a model.
- Improved Eagle3 speculative-decoding performance by removing a redundant
  concatenate in the draft path.
- Fixed Step-3.5-Flash accuracy and performance.
- Fixed the EAGLE3 MHA draft `lm_head` all-gather in pure tensor-parallel
  mode.

## MAX framework

- Added `max.pipelines.lib.MemoryPlan`, the result of memory planning when a
  pipeline is loaded: the effective `planned_max_length`, `max_batch_size`,
  `max_batch_total_tokens`, KV-cache budget, and device specs the pipeline
  and its schedulers consume.
- Renamed `MemoryEstimator.estimate_memory_footprint` to
  `MemoryEstimator.plan_from_sizes`, after the `MemoryPlan` it returns. Use
  `MemoryEstimator.plan` instead to plan from a `PipelineConfig` alone;
  `plan_from_sizes` is for callers that have already computed the weight,
  activation, and signal-buffer sizes.
- The sequence-length rule now runs once, when the config is built:
  `config.model.max_length` holds the resolved length and
  `PipelineArgs.max_length` keeps what the user asked for.
  `ArchConfig.initialize` receives that length instead of deriving it
  (`max_seq_len` is now a required keyword argument), and memory planning
  may only lower it, on the plan.
  `PipelineModel.calculate_max_seq_len`,
  `ArchConfigWithAttentionKVCache.user_provided_max_length` and
  `model_max_seq_len` are removed; architectures own the rule, so Mistral,
  Mistral3 and Pixtral now bound `max_length` on their configs.

- Memory planning no longer writes its planned `max_length` and
  `max_batch_total_tokens` back onto the pipeline config. After startup,
  `PipelineConfig.model.max_length` keeps the construction-resolved value
  and `PipelineConfig.runtime.max_batch_total_tokens` keeps the
  user-provided value (`None` when unset); the effective
  values live on `MemoryPlan`.
- `PipelineModel` now requires the `memory_plan` constructor argument
  (keyword-only; constructing a pipeline model without a plan raises a
  `TypeError`), and `PipelineModel.max_seq_len` is a read-only view of the
  plan's `planned_max_length` rather than a stored copy with a config
  fallback.
- Made `MemoryEstimator.free_memory`, `static_memory_size`,
  `available_kv_cache_memory`, and `max_supported_sequence_length` private.
  They are steps within a memory plan rather than useful on their own, and
  the values they produced are now available on `MemoryPlan`.
- The block-based vision encoder cache now shards its storage across
  devices instead of replicating every entry on each one. The same
  `--vision-cache-utilization` fraction buys the same cache capacity while
  reserving only `1/n_devices` of it per device; the remainder stays with
  the KV cache. Cache hits gather rows to each device in one batched
  submission.
- Added opt-in token-balanced CE scheduling across data-parallel replicas.
  With `--dp-ce-balance-timeout-ms` >= 0 (default -1 = off), new context
  encoding requests wait in an unbound pool and are placed by a per-step
  planner that prices them at their post-prefix-cache length (a read-only
  probe of each replica's device cache and the shared host/disk tiers) and
  binds them to the least-loaded replica when first scheduled. Unbalanced CE
  work may be deferred up to the timeout while its replica runs decode
  instead, until per-step occupancy reaches `--dp-ce-balance-threshold`
  (default 0.8). A below-threshold step with CE work on two or more replicas
  still runs immediately with each replica's chunk size reduced to the
  balance level, so only the excess defers
  (`--dp-ce-balance-enable-dynamic-chunk-size`, default on; skipped when the
  balance level is under half the CE chunk target, where the extra chunks
  would cost more than the imbalance).
- Added `--chunked-prefill-min-chunk-size` (config key
  `runtime.chunked_prefill_min_chunk_size`, default 0 = off) to set a floor,
  in tokens, on any chunk created by chunked prefill. When splitting a
  request against the CE token budget, the cut is moved earlier so that
  neither the chunk nor its remainder is smaller than the floor; if no legal
  cut point exists within the remaining budget, the request is left unsplit
  for a later step. This avoids degenerate slivers (for example an 8-token
  tail chunk after an 8192-token budget cut) that pay a full step's overhead
  and re-read the request's entire context in attention for almost no
  progress.
- Fixed non-streaming chat completions leaking a literal structural tool-call
  marker (for example `<tool_call>`) into `message.content` when a
  `max_tokens` truncation landed mid tool-call block. The response now
  surfaces only the content before the marker, with
  `finish_reason == "length"`.
- Added an experimental `--fold-sampler-into-graph` option (default off) that
  folds greedy token selection (argmax) into the captured forward graph, so a
  single device-graph replay materializes the sampled token instead of a
  separate sampler submission with a blocking readback. Applies to all-greedy
  decode batches on architectures that emit the folded token output (currently
  Nemotron-H); non-greedy requests fall back to the separate sampler.
- Added a `max-pending-futures` config (default 1, the classic
  overlap-scheduler depth of one forward in flight per request). Request
  bookkeeping now tracks unrealized future-token placeholders with a counted
  model instead of a single-sentinel check, and setting the value to 2 enables
  experimental schedule-ahead decoding: two forwards in flight per request,
  with the next step's input token realized on-device from the folded sampler
  output. Behavior at the default depth is unchanged.
- Fixed the serve CLI dropping the `fold-sampler-into-graph`,
  `max-pending-futures`, and greedy-sampling gate settings on their way to the
  model worker, which silently disabled the folded greedy sampler. With the
  flags threaded through, `--fold-sampler-into-graph` removes the per-token
  blocking sampler submission and substantially improves decode latency on
  architectures that support it.
- Added `max.engine.read` for loading a compiled-model artifact (a `.mef`
  file) without an `InferenceSession`. The resulting `CompiledModel` can
  be initialized on any session via `InferenceSession.init`. It replaces
  `InferenceSession.read`, which has been removed.
- Image generation responses on the Open Responses endpoint now report
  `usage`: token counts stay at 0 and a new `usage.image_generation_details`
  block carries `width`, `height`, `megapixels`, `steps`, and `image_count`,
  measured from the actual generated images rather than the requested
  dimensions. Previously `usage` was always `null`. (An interim nightly
  reported the raw pixel count as `output_tokens`; that encoding is replaced
  by `image_generation_details`.)
- Added `InferenceSession.read` for loading a compiled-model artifact (a
  `.mef` file) previously saved with `CompiledModel.export_mef`. It accepts a
  path or a binary file-like object (such as `io.BytesIO`), deserializes
  without invoking the graph compiler, and returns a `CompiledModel` ready to
  pass to `InferenceSession.init`.
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
- Added request-queue backpressure to MAX serve via two cooperating caps. The
  `--max-queue-size` flag (env var `MAX_SERVE_MAX_QUEUE_SIZE`, cap *N*) bounds
  the request queue to the model worker; once it is full, new requests are
  rejected immediately with HTTP 429 instead of being enqueued. The
  `--max-pending-requests` flag (env var `MAX_SERVE_MAX_PENDING_REQUESTS`, cap
  *M*) stops the worker from draining the request queue once its pending
  (prefill) queue is *M* deep, so the request queue actually backs up under
  load. Together they form a self-calibrating mechanism that sheds load to keep
  latency within SLAs and naturally accounts for long requests holding batch
  space. Both default to unbounded. Rejections are observable via the existing
  `maxserve.request_count` metric with `code="429"`.
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
  `host_offload_max_gb` now sizes one shared host pool of that size for
  the whole deployment, rather than allocating a separate pool of that size per
  replica.
- The dKV external KV-cache connector (`--kv-connector-config '{"type":
  "dkv"}'`) now supports
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
- The dKV external KV-cache connector (`--kv-connector-config '{"type":
  "dkv"}'`) now requires a
  non-empty tenant identity (`MODULAR_DKV_TENANT_ID`, set by the deployment
  operator); the empty-tenant "default" path is removed. Both the connector and
  the dKV server now reject an unset/empty tenant rather than keying an unfenced
  shared store, so every deployment (single-tenant included) routes through the
  per-tenant region-sharded store — DP replicas of one tenant still share one
  store. Multi-cache models (speculative draft+target, quantized values+scales)
  now resolve on this path, folded into the handshake's `kv_config_hash`. A
  single-tenant node spanning more than one GPU must set the dKV server's
  `--fair-share-partitions` to its GPU count.
- Added `MODULAR_MAX_RELEASE_FREE_HOST_MEMORY`, an opt-in serving knob that
  returns free host-allocator pages to the OS once model compilation finishes,
  before graph capture. Graph compilation leaves tens of GiB free-but-unreturned
  in glibc's per-thread arenas, which glibc never reclaims on its own; setting
  this variable to any non-empty value calls `malloc_trim(0)` at that point.
  On Gemma 4 31B this returns ~24 GiB of anonymous RSS per model worker in
  ~1.4s. Unset by default, and a no-op on platforms without `malloc_trim`.
- Setting the `MODULAR_MAX_RELEASE_HOST_WEIGHTS` environment variable to
  `1` frees the host copies of checkpoint weights once the GPU holds
  them, returning the full checkpoint size in host RSS. GPU deployments
  of graph-API architectures only; weights that execute on CPU must not
  be released.
- Chat completions now honor `reasoning_effort`; previously only an explicit
  `chat_template_kwargs.reasoning_effort` had any effect and the standard
  fields were silently ignored. An effort of `none` disables thinking, and
  values set directly in `chat_template_kwargs` still win.
- `--num-speculative-tokens` is now unset by default, and each speculative
  method resolves its own default: `eagle` and `mtp` keep drafting 2 tokens
  per step, while `dflash`-style block drafters (DFlash, DSpark) derive the
  draft checkpoint's trained block width. Explicit values are honored as
  before. Previously the flag defaulted to 2 for every method and block
  drafters overrode it at load time with a warning; a bare DFlash run now
  also sizes its KV cache draft headroom at the trained width instead of the
  old default.
- VLM tokenizers can now cache preprocessed media, so an image or video resent
  on a later conversation turn skips the resize, rescale and patchify (and for
  video, the whole decode) instead of redoing it. Keyed on the same
  raw-encoded-bytes digest the vision encoder cache uses, and bounded by host
  bytes rather than entry count: `--max-vision-preprocess-cache-bytes` and
  `--max-video-preprocess-cache-bytes` each default to 10 GiB, their combined
  size is capped at a quarter of the memory the process may use (a cgroup grant
  where there is one), and `0` disables either. The budget is a ceiling rather
  than a reservation -- the cache grows into it and evicts to stay under it --
  and on a host with less than 80 GiB the cap scales both down proportionally
  rather than overcommitting. Entries unused for
  `--max-media-preprocess-cache-idle-seconds` (default 300, `0` disables) are
  dropped on the next cache lookup or insert, so a burst of distinct media does
  not hold host memory for the life of the process. Enabled for Gemma 4 images
  and video, Kimi K2.5 images, and Qwen2.5-VL and Qwen3-VL-MoE images.
- Added `max.driver.begin_launch_trace()` and
  `max.driver.take_launch_trace()`, exposing the launch trace recorded by the
  runtime on CUDA and HIP devices. The trace lists the operations enqueued
  across all streams — kernel launches (name, grid/block dimensions, shared
  memory), memory copies, and memsets — in one enqueue-ordered list of
  `max.driver.LaunchTraceEntry` values, each with a `stream_index` identifying
  its stream and a deterministic, address-free `semantic_hash`. Because it is
  process-global, work enqueued on streams the caller has no handle to (such as
  a compiled graph's internal stream) is captured too. Intended for tests and
  debugging that assert which device work a code path enqueues and on which
  stream. The `max.driver.launch_trace()` context manager wraps the pair and
  always stops recording on block exit, even if the block raises.
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
- The vision encoder cache now stores embeddings in fixed-size blocks.
  Capacity is a byte budget carved into 128-token blocks — a video spans
  many blocks and an image a few — so a video-capable model no longer
  collapses the cache to a handful of worst-case-video slots that starve
  image workloads. The budget is set with the new
  `--vision-cache-utilization` flag, a fraction of the KV cache pool
  budget (default `0.05`; `0` disables caching). The previous
  entry-count cache and its `--max-vision-cache-entries` flag are
  removed.
- Vision embedding assembly during chunked prefill is now bounded by the
  active window: each step copies only the embedding rows whose
  placeholder tokens fall inside the chunk, with dense scatter indices,
  instead of rebuilding every image's rows with out-of-bounds sentinels.
  Per-chunk copy cost now scales with the chunk size rather than the
  request's total image tokens.
- Added `DeviceBuffer.unsafe_host_ptr()` to the Mojo `max.gpu.host` API. On
  devices with unified memory (Apple silicon), it returns a CPU-addressable
  pointer to the buffer, so the host can read a kernel's output after
  `DeviceContext.synchronize()` without an `enqueue_copy` round trip. Reads
  through it are uncached, so it suits small control records rather than bulk
  readback. A CPU device returns the buffer's own pointer, since its
  allocations are host memory already; devices whose memory is not
  CPU-addressable raise.

### Inference server

- `/v1/responses` now fetches client-supplied `input_image` URLs through the
  same media resolver as `/v1/chat/completions`, so the two paths share one
  byte cap and one error mapping. Previously the responses path had its own
  downloader with no size limit, meaning an arbitrarily large image could be
  fetched and base64-expanded in memory, and its failures echoed the
  underlying network error back to the client. The inlined `data:` URI's MIME
  type is now sniffed from the fetched bytes instead of guessed from the URL,
  and content that is not a decodable image is rejected with a 400 rather than
  inlined as an image.

- GLM models now map `reasoning_effort` onto the two thinking levels their
  chat template can express, instead of forwarding it verbatim. The template
  reads only `high` as a distinct level and treats every other value as
  maximum effort, so passing the value through inverted the scale: `low` and
  `medium` requested maximum reasoning while `high` requested less than they
  did. Every effort other than `none` (which disables thinking), `max` (the
  template's own top level, still addressable directly) and `xhigh`
  (OpenRouter's name for that same top level) now selects the lower level, so
  an unrecognized value degrades to less reasoning instead of silently maxing
  out. Requests that set no effort are unaffected.

- Structured-output JSON grammars can be made whitespace-tolerant, per
  architecture via `default_structured_output_any_whitespace`.
  - GLM 5 models default to whitespace-tolerant `response_format` grammars.

- Structured-output grammar compilation now runs off both serving hot
  paths. A new request's grammar matcher (from `response_format` JSON
  schemas or tool-call grammars) is built on a worker thread while the
  request waits for admission instead of on the scheduler's decode
  thread, and the API server's admission-time schema validation runs off
  the event loop instead of freezing in-flight streaming responses. A
  cold multi-second compile of a complex schema now delays only that
  request instead of stalling inter-token latency for every active
  request.

- MAX Serve no longer drops uvicorn's log records. The console, file, and
  OTLP handlers filter on an allowlist of logger prefixes that omitted
  uvicorn, which owns the HTTP error log, so an exception escaping the ASGI
  application, a malformed request, and the cancellation of in-flight
  requests when the shutdown drain expires all went unreported. The uvicorn
  logger stays at `WARNING`, so the per-request access log remains
  suppressed.

- Added `MAX_SERVE_HTTP_KEEPALIVE_TIMEOUT_S` to control how long an idle HTTP
  connection is held open, defaulting to 120 seconds (previously hard-coded to
  5 seconds). A server that retires idle connections sooner than its clients do
  always wins the close race, and a close landing just as a pooled client
  writes its next request reaches that client as a TCP reset rather than a
  response. A client cannot replay a POST body, so it surfaces the reset
  instead of retrying. Keep this above the idle-connection timeout of every
  client that pools connections to MAX Serve.

- An unhandled server error now returns the standard OpenAI `error` envelope as
  JSON rather than a bare `text/plain` `Internal Server Error`. The
  request-session middleware runs outside Starlette's exception middleware, so
  raising from it bypassed the app's exception handler and reached
  `ServerErrorMiddleware`, which replies in plain text and then re-raises,
  prompting uvicorn to close the connection under a client that was owed a
  response.

- Speculative decoding takes `--draft-proposal sampled` (default `argmax`,
  unchanged). The draft model samples its proposal under the request's
  temperature/top-k/top-p and keeps the distribution it drew from, so
  verification runs true speculative sampling — accept on the
  `p_target/q_draft` ratio, recover from `max(p_target - q_draft, 0)` — rather
  than the typical-acceptance approximation, and the emitted tokens follow the
  target model's distribution.

### Server metrics

- Fixed the speculative-decoding per-position acceptance-rate histogram
  (`maxserve_spec_decode_acceptance_rate_per_position`) understating
  acceptance: decode batches that performed zero verifications published a
  full row of 0% observations, diluting every position's average. Such
  batches now contribute nothing, matching the acceptance-length histogram's
  population. The batch log line also shows the acceptance length including
  the bonus token next to the accepted-drafts-per-step value, since the two
  conventions are easy to confuse.

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

- `max.experimental.nn.Module.compile` reuses precompiled MEFs when the session
  has them, so a ModuleV3 model can be compiled where no accelerator is attached
  and initialized where one is. `max.experimental.support.set_export_mefs`
  records each compiled graph into a directory, and
  `max.experimental.support.set_precompiled_mefs` initializes those artifacts
  instead of compiling. `InferenceSession.compile_reusing_mefs` is the same
  half-step for callers that trace a graph and initialize it themselves.

- Eager mode tensors will use the JIT by default. This unlocks fusion and
  shape specialization optimizations even for eager code, beating PyTorch
  performance in eager in the common case.

- `max.experimental.sharding.NamedMapping` takes its mesh from the enclosing
  `mesh_context()` when none is passed, so a layer can name the axis it shards
  along without being handed a mesh. Its `original_spec` and
  `original_unreduced` properties are removed.

- Added `max.experimental.tree_utils`, pytree utilities over nested `list` /
  `tuple` / `namedtuple` / `dict` and any class declaring the tree protocol:
  `__tree_flatten__` with either `__tree_unflatten__` or `__tree_empty__`, and
  an optional `__tree_setattr__`. There is no registry and no decorator, so a
  type opts in by declaring the methods. `flatten` and `unflatten` carry a
  value across a flat boundary, `leaves`, `paths` and `nodes` read it, `map`
  builds a new tree, and `update` writes path-keyed values into an existing one
  in place. Every walk takes `leaf`, saying where it stops, and `shared`,
  saying whether a value reachable by two paths is one object or two. Import
  the module as a namespace: `from max.experimental import tree_utils as tree`.

- Added `max.experimental.compilation`, three transforms over plain
  callables. `stage(fn)(*args, **kwargs)` traces `fn` into a `max.graph`
  that can be printed and inspected as MLIR. The arguments are `fn`'s own,
  except that each tensor is given as a `TensorType`. This partially
  evaluates `fn`: the tensor types become graph inputs, and every other
  argument is evaluated during tracing. `compile(fn, weights=...)(*args,
  **kwargs)` stages the same way and compiles the graph; the result is
  callable on real tensors. Weights and device memory load only on the
  first call, so `export_mef` can save the compiled graph to a file without
  loading either. `as_subgraph(fn)` returns a drop-in replacement for `fn`
  that, during tracing, calls one shared subgraph instead of inlining its
  body, so a stack of identical layers compiles once.

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

- `max.driver.Buffer` now implements `__str__`, so `str(buffer)` and
  `print(buffer)` show the buffer's data formatted like a numpy array, followed
  by its `dtype`, `shape`, and `device`. `repr(buffer)` still returns the
  metadata-only representation.

- `max.nn.sampling.AcceptanceSampler` and
  `max.nn.sampling.stochastic_acceptance_sampler` take a `draft_proposal`
  argument. The default, `"argmax"`, is unchanged: the draft proposes
  deterministically and verification runs typical acceptance. With
  `"sampled"`, the caller passes the distribution the draft sampled from, so
  verification runs the real `p_target / q_draft` ratio test and recovers
  rejected positions from `max(p_target - q_draft, 0)`; temperature, top-k and
  top-p then all apply to the draft-verification distribution, where
  `"argmax"` applies only temperature. Sampled mode is GPU-only, needs a
  static `vocab_size`, and cannot be combined with relaxed thinking-phase
  acceptance, whose rule assumes the drafted token is the draft's argmax.

### C API

## MAX kernels

- The MLA sparse-attention indexer (DeepSeek V3.2, GLM 5.x) now does work
  proportional to each row's actual key count instead of the batch's
  `max_cache_length` metadata. Inside captured decode device graphs that
  metadata is baked at capture time — with a 1M-token maximum sequence length
  it sits orders of magnitude above the tokens a batch actually holds — and
  the indexer paid a full-width `-inf` score fill, a full-width top-k scan,
  and a key-tile-per-CTA scorer grid per layer per step at that frozen
  bound. The bitonic top-k kernels now clamp each row's scan to its live
  causal range, the score-buffer fill is skipped on the SM100 scorer path
  (which writes every live slot itself), and the SM100 scorer's key-split
  route now covers the tensor-parallel head counts (4 and 8) with its part
  count capped at a fixed number of waves, so the grid is sized to the
  hardware rather than to the metadata bound while per-CTA loop bounds come
  from the runtime cache lengths. At the GLM 5.2 MTP decode shape (batch 8,
  width 6, 76k-token context, 4 heads per rank) with metadata frozen at 1M,
  one indexer layer drops from 0.89 ms to 0.10 ms on B200, matching its
  cost at a bound sized to the runtime lengths; shapes without a metadata
  gap are unchanged except a small fixed per-call cost for the row-bounds
  clamp (~4% on a batch-256, 4k-context decode).
- Sped up GPU token sampling by about 4% per output token when the largest
  `top_k` in the batch is below 10, by removing a device synchronize from
  `fused_token_sampling_gpu`. The synchronize backed a check that raised on an
  all-NaN logits row. Such a row now yields an arbitrary in-range token rather
  than an error. Set `max-debug.assert-level` to `all` to restore the check, or
  use `max-debug.nan-check` to locate NaN logits.
- Fixed expert-parallel dispatch dropping half of every token belonging to an
  expert that only one communication SM serves, which surfaced as NaN logits.
  The block-scaled wire formats (NVFP4 and MXFP8) copy a token tile as two
  column halves claimed separately, and the claim loop stopped as soon as a
  claim covered the last token, so the remaining half was never copied unless
  a second SM happened to be on the same expert. Since experts are assigned
  round-robin over the communication SMs, this began once a device held more
  experts than half that count — 74 per device on a B200, so a 896-expert MoE
  over eight devices returned NaN while 512 experts stayed correct.

- The SM100 grouped block-scaled matmul accepts MXFP4 weights against MXFP8
  activations (W4A8), so a quantized MoE can feed its packed 4-bit experts
  straight to the tensor cores rather than dequantizing them to bfloat16
  first. This removes MAX's per-forward `mxfp4_dequant` over the routed expert
  stack, and it keeps the weights at their 4-bit footprint in global memory,
  which matters most at expert counts where a bfloat16 copy of the stack does
  not fit. A new `unpack_fp4` option on the NVIDIA TMA descriptor helpers,
  backed by the `TensorMapDataType.PACKED_FP4_ALIGN16B` tensor-map type, pads
  the weights into the byte-addressed form the tensor cores read as the copy
  engine lands them in shared memory.

- The joint top-k/top-p sampling kernel can now also return the masked,
  renormalized distribution it drew from, exposed as
  `max.nn.kernels.topk_fused_sampling_with_dist`. Speculative decoding needs
  that distribution to build a rejection residual, and reads the sampled
  token's own probability out of it -- a value that has to agree with the
  sampler's accept decision, so it comes from the sampling kernel rather than
  a separate softmax. The existing single-output path is unchanged.
- Added `max.nn.kernels.topk_topp_masked_probs`, which computes a row's
  top-k/top-p masked renormalized softmax without sampling and without a
  sort. Speculative decoding verification reads the target's masked
  probability of each drafted token and builds its rejection residual from
  this one tensor, in the same form the draft sampler emits its proposal
  distribution.
- The fused gumbel-argmax sampling kernel takes a `from_probs` parameter,
  exposed as `max.nn.kernels.gumbel_argmax_from_probs`: each row's score is
  `ln(p) + gumbel` over unnormalized probabilities, drawn with noise the
  kernel generates from a per-row seed. This enables sampling a speculative
  decoding rejection residual `max(p_target - q_draft, 0)` that the caller
  builds in graph ops. GPU-only, non-Apple.

## Breaking changes

- `max.pipelines.PipelineArgs` is now immutable: assigning to one of its
  top-level fields after construction raises a pydantic `ValidationError`.
  Construct it with the values you need. Its sub-configs (`runtime`,
  `sampling`, etc.) are unchanged for now.

- `max.pipelines.lib.LoRAConfig` and `max.pipelines.lib.ProfilingConfig` are
  now immutable (pydantic `frozen=True`); assigning to a field after
  construction raises a `ValidationError`. Construct with the desired values.
- The KV cache connector is now configured as a single object: its type moved
  onto `--kv-connector-config` as a `type` field, and the separate
  `--kv-connector` flag is removed. Replace `--kv-connector rust_tiered` with
  `--kv-connector-config '{"type": "rust_tiered"}'`, and in a recipe set
  `model.kv_cache.kv_connector_config.type`. `host_kvcache_swap_space_gb` is
  renamed `host_offload_max_gb` to match `disk_offload_max_gb`, and both now
  default to sizing their tier from the device page pool (twice it on host,
  three times on disk) rather than to a fixed 50 GiB. Dict-valued `kv_cache`
  flags now merge field-wise over a config file's value instead of replacing
  it, so overriding one connector field on the command line keeps the rest --
  previously a partial override reset the connector type and silently disabled
  offloading.

- Renamed `max.driver.DeviceStream` to `DeviceQueue` and
  `Device.default_stream` to `Device.default_queue`; the old names were
  removed. The driver models work submission as a command queue; a stream
  is one backend's implementation of that queue. Method, property, and
  argument names (`Buffer.stream`, `stream=`, `native_stream_handle`) are
  unchanged.

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
- `--max-vision-cache-entries` is replaced by `--vision-cache-utilization`,
  a fraction of the KV cache pool budget for the vision encoder cache
  (default `0.05`; `0` disables caching). The cache is block-based, so an
  entry count no longer describes its capacity; configs setting the old
  flag must convert to a pool fraction.

- The legacy alias-buffer LoRA path has been removed. ModuleV3 LoRA (adapters
  passed as graph inputs) is now the only supported LoRA implementation.
  Serving a non-ModuleV3 architecture with `--lora-paths` now raises a clear
  error at startup instead of building a manager that never applies the
  adapters; serve the model's ModuleV3 variant (for example,
  `--prefer-module-v3`) to use LoRA adapters.

- Removed the parametric `max.benchmark.bencher_iter_custom[fn](bencher, ctx)`
  overloads and unused `bencher_iter_custom_multicontext()`. Pass the launch
  closure as a value: `bencher_iter_custom(bencher, fn, ctx)`.

- Removed `max.algorithm.reduce_boolean()`, which took its `reduce_fn` and
  `continue_fn` as `capturing` compile-time parameters and had no callers. Use
  `max.algorithm.reduce()` with a boolean accumulator, or write the early-exit
  loop directly.

- Removed the parametric `max.algorithm.parallelize[func](num_work_items, ...)`
  and `max.algorithm.parallelize_over_rows[func](shape, axis, grain_size, ...)`
  overloads that took a `capturing` closure as a compile-time parameter. Pass
  the body as a unified closure in the first runtime argument instead:
  `parallelize(func, num_work_items, ...)` and
  `parallelize_over_rows(func, shape, axis, grain_size, ...)`. Closure bodies
  drop `@__parameter` / `@__copy_capture` in favor of an explicit capture list,
  for example `def body(start: Int, end: Int) {imm}:`.

- Removed the parametric `capturing` overloads of
  `DeviceContext.execution_time[fn](num_iters)`,
  `DeviceContext.execution_time_iter[fn](num_iters)`, and
  `DeviceContext.enqueue_cpu_function[fn]()`. Pass the closure as a runtime
  argument instead: `execution_time(fn, num_iters)`,
  `execution_time_iter(fn, num_iters)`, and `enqueue_cpu_function(fn)`. Nested
  closures passed this way are unified closures, so replace `@__parameter` and
  `@__copy_capture(x)` with an explicit capture list such as `{imm}` or
  `{var x, imm}`.

- `PipelineRegistry.retrieve_factory` now returns a `RetrievedPipeline`
  dataclass with `tokenizer`, `factory`, and `memory_plan` fields instead of
  a `(tokenizer, factory)` tuple, so callers can reach the memory plan
  computed during retrieval. Replace tuple unpacking with attribute access.
  `PipelineRegistry.retrieve` is unchanged.

- The serving surface now reads the planned sequence length and batch token
  budget from the memory plan instead of re-reading them from the pipeline
  config. `TokenGenerationSchedulerConfig.from_pipeline_config`,
  `start_model_worker`, the scheduler loaders, and the startup log helpers
  (`log_basic_config`, `log_pipeline_info`) take the memory plan as a
  parameter. Resolved values are unchanged.

- Renamed `MemoryPlan.max_length` to `MemoryPlan.planned_max_length` to
  distinguish the plan's value from the user intent on
  `PipelineArgs.max_length` and the construction-resolved
  `PipelineConfig.model.max_length`, which keep their names.

## Fixes

- Fixed run-to-run nondeterminism of `layer_norm`, `rms_norm`, and other
  Row-API rowwise reductions on Apple Silicon GPUs: a block that reduced
  several rows re-used its shared-memory strip across row iterations
  without ordering the trailing broadcast read against the next combine's
  first store. Model outputs on Metal (for example FLUX.2 image
  generation) are now byte-identical across runs; NVIDIA and AMD codegen
  is unchanged.

- On Apple Silicon, a missing Metal Toolchain (a separate download since
  Xcode 16) now surfaces `xcrun`'s own error, which names the fix
  (`xcodebuild -downloadComponent MetalToolchain`), instead of the opaque
  "Please submit a bug report." message.

- Fixed tool-call requests failing with HTTP 400 (`anyOf branch and base
  schema both set "description"`) on models whose grammar compiles in strict
  mode (GLM-5.x, Gemma 4). The xgrammar JSON-schema converter's `anyOf`
  base-merge now skips annotation-only keywords (`description`, `title`,
  `default`, `examples`, `$comment`, `deprecated`, `readOnly`, `writeOnly`)
  instead of rejecting them as branch/base conflicts; they carry no grammar
  constraint.

- Fixed a race that enforced structured-output grammars during a reasoning
  model's thinking span.

- Fixed structured output and constrained tool calling being silently ignored
  on the Kimi K2.5-family pipelines when serving with DFlash speculative
  decoding (`--speculative-method dflash`). The unified DFlash graph compiled
  without the constrained-decoding bitmask inputs. The graph now binds the
  bitmask inputs and applies the grammar mask across every speculative position,
  matching the EAGLE speculative-decoding pipelines.

- Fixed GPT-OSS, OLMo 3, and OLMo 2 ignoring `--max-length`. The server
  accepted prompts up to the checkpoint's own length limit, and sized the KV
  cache for that limit, whatever you asked for and whatever memory allowed.
  Both now use the length you set. Runs that pass no `--max-length` are
  unaffected.

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

- Fixed CPU `argmax`/`argmin` reductions returning a wrong index for reduce
  axes of 256K+ elements, for example an argmax over a `[1, 2097152]` tensor,
  where the row's reduction fans out across multiple CPU workers.

- Fixed the distribution the top-k/top-p sampler emits for speculative
  decoding (`emit_dist`) being under-normalized when a `min_p` mask removes
  weight and the row passes top-p at the first trial: the row was scaled by
  the unmasked softmax mass instead of the masked kept mass, so it summed to
  less than one and skewed the rejection residual. The sampled token stream
  was and remains unchanged.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
