---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

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
- Added the public `Tree` type alias in `max.tree` for nested pytree values.
  Layer subgraph inputs are annotated with `Tree[Any]` instead of the
  removed `SubgraphInput` alias.
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

### Inference server

- Added `--prefill-coalesce-min-pending` (default 0, off): under in-flight
  batching, hold pending fresh prefills until that many can share one mixed
  step instead of admitting them one by one. Mixed steps forfeit device graph
  capture for their decode rows, so coalescing admissions keeps more decode
  steps on the captured fast path, trading a bounded prefill-admission delay
  (at most the same number of decode steps) for lower decode latency at high
  concurrency. Mid-prefill chunked continuations are never held.

### Server metrics

### `max` CLI

### Python API

### C API

## Kernels and GPU programming

- Added `max.nn.kernels.keyed_uniform`, which draws one uniform value in
  `[0, 1)` per row of a seed tensor. Every row is its own Philox key, so a
  row's value is a function of its seed alone, not of the row's position or of
  what else shares the launch. `ops.random.uniform` keys the whole tensor off
  index 0 of the graph seed and walks the flat element index as its Philox
  counter, so it cannot express that. Speculative decoding's accept coin uses
  it to draw one uniform per (request, draft position).

## Breaking changes

- `KVCacheMetrics` drops `nixl_read_blocks_local`, `nixl_read_blocks_remote`,
  and the `remote_read_ratio` property computed over them. No code path ever
  populated either field, so the ratio returned `0.0` for every caller. Six
  populated counters are added in their place: `dkv_peer_attaches`,
  `dkv_peer_attach_failures`, `dkv_peers_dropped`, `dkv_peer_loads`,
  `dkv_peer_load_failures`, and `dkv_hints_rejected`. They are zero unless the
  external KV-cache connector is in use.

## Fixes

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

## Mojo language
