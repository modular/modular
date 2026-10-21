---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Models

## MAX framework

### Inference server

### Server metrics

## `max` CLI

## Python API

### C API

## Kernels and GPU programming

## Breaking changes

- Deprecated `LayoutTensor` and `LayoutTensorIter` in the `layout` package.
  Use `TileTensor` and `TileTensorIter` instead.

- Removed the `LayoutTensor` overloads of `sum()` and `max()` from
  `layout.math` and of `create_tensor_tile()` from `layout.tma_async`. Use the
  `TileTensor` overloads instead.

- `TMATensorTile`, `TMATensorTileArray`, `TMATensorTileIm2col`, and
  `SplitLastDimTMATensorTile` now take their shapes as flat `Coord` values
  instead of `IndexList[rank]`, and no longer take a separate `rank`
  parameter.

- The `rms_norm` functions and GPU kernels in `nn.normalization`, the KV cache
  `rms_norm` ops, and `gemv_and_partial_norm` now take `epsilon` and
  `weight_offset` as `Float32` compile-time parameters instead of runtime
  arguments.

- Removed the `extensibility.foreach[func, target=...](output, ctx)` overloads
  that took a `capturing` closure as a compile-time parameter. Pass the body
  as a closure in the first runtime argument instead:
  `foreach[target=target](body, output, ctx)`.

- Removed the `--rejection-sampling-strategy` flag, the `SpeculativeConfig`
  field it set, and the `RejectionSamplingStrategy` type. The flag had no
  effect, because each architecture's acceptance sampler chooses the
  acceptance rule.

- Renamed the `--eplb-replicas-per-gpu` option to
  `--eplb-redundant-experts-per-gpu` (`eplb_redundant_experts_per_gpu` in
  config files). The value and its effect are unchanged.

- Removed the `mojo_tiered` KV connector type. Use `rust_tiered` instead. A
  `--kv-connector-config` that sets `mojo_tiered` now fails with an error.

- The legacy KV cache manager (`MODULAR_USE_LEGACY_KV_CACHE=1`) no longer
  supports KV connectors or disaggregated inference, and requesting either
  now fails at startup. Disaggregated inference with a cache that has
  non-full-attention layers also fails instead of falling back to the legacy
  manager.

- `max benchmark` no longer strips a leading `@` from a file path, whether it
  is passed to `--response-format`, `--tools`, `--extra-body` or
  `--agentic-tool-profiles` or named in a `--workload-config` YAML. Pass the
  bare path instead: `--tools path/to/tools.json`.

- Removed the `DeviceQueue` overloads of `max.driver.Buffer.to()` and
  `Buffer.copy()`. Pass a `Device` (or, for `to()`, a sequence of devices)
  instead: `buffer.to(device)` and `buffer.copy(device=device)`.

- DSpark speculative decoding now has its own method: serve a DSpark draft
  with `--speculative-method dspark` instead of `dflash`. DSpark drafts a
  block at a time like DFlash but runs a different draft step, so a DSpark
  draft under any other method now fails at startup with a message naming
  the method to use.

- Changed shape functions registered with
  `@extensibility.register_shape_function()`
  to require tensor-trait arguments. Use `Some[Tensor]` for shapes,
  `Some[TileTensorable]` for element values, or `TensorTuple` for variadic
  inputs. Legacy tensor aliases remain supported on `execute()`.

- Removed `max.pipelines.GenerateMixin` and the `generate()` and
  `generate_async()` methods it gave text generation pipelines. Build
  contexts with `pipeline.tokenizer.new_context()` and run them with
  `GenerateIterator.from_contexts()`, or pass token ids to
  `max.pipelines.lib.generate.generate()`. `TextGenerationPipelineInterface`
  now declares `pipeline_config` and `tokenizer` as abstract properties, so
  custom subclasses must implement them.

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

- The reduction callbacks in the `max.algorithm` package now take a `Coord`
  instead of an `IndexList[rank]`, and no longer carry a `rank` parameter.
  This changes the `input_fn` and `output_fn` contracts of `sum()`, `min()`,
  `max()`, `mean()` and `product()`, along with `row_reduce()` and
  `reduce_launch()` in the GPU backend. The coordinate holds the same values
  as before, so results are unchanged:

  ```mojo
  # was: def input_fn[width: Int, rank: Int](idx: IndexList[rank]) -> ...
  def input_fn[width: Int](idx: Coord) -> SIMD[dtype, width]:
      return input.load[width=width](idx)
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

- `max.nn.moe.MoE` and `MoEQuantized` take the expert activation as one
  `activation=` argument, a `GatedActivation` from `max.nn.activation`, in
  place of `use_swigluoai`, `swiglu_alpha`, `swiglu_limit`, and
  `gated_activation_fn`. Replace `use_swigluoai=True, swiglu_alpha=a,
  swiglu_limit=l` with `activation=GatedActivation("swigluoai", alpha=a,
  limit=l)`, and `gated_activation_fn=fn` with
  `activation=GatedActivation(gated_activation_fn=fn)`. The fused SwiGLU
  kernel wrappers `max.nn.kernels.grouped_matmul_blocked_swiglu` and
  `grouped_dynamic_block_scaled_matmul_amd_swiglu_quant`, and
  `max.nn.comm.ep.ep_kernels.fused_silu` and `fused_silu_quantized`, take the
  same `activation=` argument.

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

- The deprecated `max.kv_cache` module is removed. It only re-exported
  `max.pipelines.kv_cache` and emitted a `DeprecationWarning` on import. Import
  `PagedKVCacheManager`, `DummyKVCache`, `load_kv_manager`, and the other names
  it re-exported from `max.pipelines.kv_cache` instead.

- The `sha256_64` value of `kv_cache_hash_algo` is removed, and a config that
  sets it now fails validation at startup. It kept the first 8 bytes of each
  block's SHA-256 digest but chained the next block onto the full digest, so a
  request whose hash chain grew after its first block (decode past a block
  boundary, chunked prefill) raised an error. Use `sha256` instead. The dKV
  connector keys `sha256` blocks by the first 8 bytes of the digest, so dKV
  keys are unchanged.

## C API

## Kernels and GPU programming

- The deprecated `LayoutTensor` and `LayoutTensorIter` types have been removed
  from the `layout` package, along with the `to_layout_tensor()` and
  `lt_to_tt()` conversions. Use `TileTensor` and `TileTensorIter` instead.
  `ThreadScope` now lives in `layout.tile_io`.

## Tools

- Added the ability to install `max-serve` and `max-benchmark` as Conda extras,
  using `max[extras=[serve]]` and `max[extras=[benchmark]]` respectively.
  `max[extras=[all]]` installs both plus `mojo`.

## Fixes

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
