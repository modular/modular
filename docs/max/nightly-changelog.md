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
  - Fixed a `Qwen3EmbeddingModel` crash.
- Added per-request LoRA adapter support: `LoRALinear` and
  `StackedLinearLoRA` extend LoRA to standalone and fused-QKV projections,
  with `LoRAManager.apply` swapping target layers in a model.
- Improved Eagle3 speculative-decoding performance by removing a redundant
  concatenate in the draft path.
- Fixed Step-3.5-Flash accuracy and performance.
- Fixed the EAGLE3 MHA draft `lm_head` all-gather in pure tensor-parallel
  mode.

## MAX framework

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

- Eager mode tensors will use the JIT by default. This unlocks fusion and
  shape specialization optimizations even for eager code, beating PyTorch
  performance in eager in the common case.

- `max.experimental.sharding.NamedMapping` takes its mesh from the enclosing
  `mesh_context()` when none is passed, so a layer can name the axis it shards
  along without being handed a mesh. Its `original_spec` and
  `original_unreduced` properties are removed.

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

- Fixed tool-call requests failing with HTTP 400 (`anyOf branch and base
  schema both set "description"`) on models whose grammar compiles in strict
  mode (GLM-5.x, Gemma 4). The xgrammar JSON-schema converter's `anyOf`
  base-merge now skips annotation-only keywords (`description`, `title`,
  `default`, `examples`, `$comment`, `deprecated`, `readOnly`, `writeOnly`)
  instead of rejecting them as branch/base conflicts; they carry no grammar
  constraint.

- Fixed structured output and constrained tool calling being silently ignored
  on the Kimi K2.5-family pipelines when serving with DFlash speculative
  decoding (`--speculative-method dflash`). The unified DFlash graph compiled
  without the constrained-decoding bitmask inputs. The graph now binds the
  bitmask inputs and applies the grammar mask across every speculative position,
  matching the EAGLE speculative-decoding pipelines.

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
