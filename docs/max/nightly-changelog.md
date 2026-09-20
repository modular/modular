---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

- Added a [chat templates](/serve/chat-templates) guide that covers information
  about the `--chat-template` flag for the `max serve` CLI command.

## MAX models

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

- Added the `pre-jit` debug option (`MODULAR_DEBUG=pre-jit`,
  `[max-debug] pre-jit`, or `InferenceSession.debug.pre_jit`). It stops graph
  compilation once the Mojo for the graph has been emitted into
  `ir-output-dir`, skipping the kernel JIT, and `InferenceSession.compile()`
  raises `max.engine.CompilationStopped` in place of returning a model.
  `max warm-cache` treats that as success, so
  `MODULAR_DEBUG=ir-output-dir=<dir>,pre-jit max warm-cache ...` is a quick
  way to get the Mojo a model compiles to.

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

### Inference server

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

- Added `--prefill-schedule-interval` (default 1, every step): admit prefill
  work only on every Nth scheduler step, leaving the steps in between entirely
  to decode. Data-parallel ranks advance in lockstep, so prefill on any one
  rank stalls the whole group; scattering it across steps pays that stall
  repeatedly, while concentrating it onto a shared cadence pays it once. A step
  with no decode work on any replica admits prefill regardless, rather than run
  an empty batch. The cost is delayed prefill admission, bounded at N-1 steps.
  Unlike prefill coalescing, mid-prefill chunked continuations are held too:
  the budget emits one chunk per step, so exempting them defeats the cadence.

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

### `max` CLI

### Python API

- Added `max.experimental.custom.declare`, which declares a custom op's
  input and output signature once, as an immutable `custom.CustomOp`, and
  runs eagerly or inside a `Graph`. Output dims may be symbolic,
  parameterized, or left to the kernel's shape function to determine at run
  time.

### C API

## Kernels and GPU programming

- `TileTensor` gained `as_span()`, returning a `Span` over the tensor's
  elements.

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

## Breaking changes

- `KVCacheMetrics` drops `nixl_read_blocks_local`, `nixl_read_blocks_remote`,
  and the `remote_read_ratio` property computed over them. No code path ever
  populated either field, so the ratio returned `0.0` for every caller. Six
  populated counters are added in their place: `dkv_peer_attaches`,
  `dkv_peer_attach_failures`, `dkv_peers_dropped`, `dkv_peer_loads`,
  `dkv_peer_load_failures`, and `dkv_hints_rejected`. They are zero unless the
  external KV-cache connector is in use.

## Fixes

- Fixed a crash serving Gemma 4, Gemma 3 multimodal, Pixtral, and Qwen 3
  embedding with a batch larger than one. Each of these architectures accepted
  `max_batch_size` and never handed it to `PipelineModel`, which left the base
  at its default of 1 while the scheduler dispatched full batches. The model
  input buffers a batch stages into are sized from that value, so the first
  multi-context step overran them and the model worker crashed. The
  architectures pass it up now.

- The functional kernel wrappers in `max.experimental.nn.common_layers` now
  open a realization context, so eager attention with a paged KV cache runs.

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

## Mojo language
