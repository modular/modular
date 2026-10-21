---
title: MAX nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Models

- GLM-5.1, GLM-5.2, and GLM-5.3 now default `clear_thinking` to `False`, which
  keeps the `reasoning_content` a client sends back for earlier turns in the
  prompt. GLM-5.3 previously defaulted it to `True`, and GLM-5.1's and
  GLM-5.2's templates clear earlier reasoning when it is unset. This matches
  Z.ai's Coding Plan endpoint, and
  [Z.ai's thinking-mode guide](https://docs.z.ai/guides/capabilities/thinking-mode)
  recommends it for coding and agent workloads. Pass `clear_thinking=True` in
  the chat template options to drop earlier reasoning.

## Inference server

### Server metrics

- Added `maxserve_request_client_disconnect_cancellations_total`, which
  counts streamed responses cut short by a client disconnect and completion
  requests abandoned while waiting for a result, split by `phase`. A
  mid-stream disconnect was previously invisible:
  `maxserve_request_count_total` records it under the status code already
  sent.

## `max` CLI

## Python API

- Added `max.driver.Buffer.memset(value)` to fill a contiguous buffer or
  slice in place with a byte value, without allocating a source buffer.

- Added the `ops.CoordinateTransformMode` and `ops.RoundMode` enums.
  `ops.resize_linear`, `ops.resize_nearest`, and their `max.experimental`
  functional counterparts take them for `coordinate_transform_mode` and
  `round_mode` in place of the integer codes 0-3. The integer codes still
  work.

- Added `max.driver.float32_to_bfloat16_as_uint16()`,
  `max.driver.bfloat16_as_uint16_to_float32()`, and
  `max.driver.float32_array_to_buffer()` for moving bfloat16 data between
  NumPy and a `Buffer`. `max.pipelines.lib` still re-exports
  `float32_to_bfloat16_as_uint16()` and `float32_array_to_buffer()`.

- Removed the `max.pipelines.lib.bfloat16_utils` module. Import its functions
  from `max.driver` instead.

- Removed `GraphPipelineModel`, `ModuleV3PipelineModel` and
  `MultiGraphPipelineModelWithKVCache` from `max.pipelines.lib`. Subclass
  `PipelineModel`, return the module from `graph_specs()` under
  `GraphRole.ENCODER`, and load it with `load_graphs()` instead.

- `Buffer.to(device)` for a buffer on another queue of that device requires an
  `after` event and no longer syncs implicitly. Mojo's `reassign_ownership_to()`
  is removed; use `enqueue_wait_for()` and keep buffers alive until synced.

- Removed `max.nn.FusedMLP`. Use `max.nn.MLP`, which takes the separate
  `gate_proj` and `up_proj` weights a checkpoint ships and fuses their matmuls
  in the graph.

- Removed `max_single_alloc_size()` and `run_healthcheck()` from the Mojo
  `DeviceContext`. On NVIDIA and AMD GPUs the health check still runs when a
  device context is created. Python's `Device.max_single_alloc_size` is
  unchanged.

- `generate()`, `GenerateIterator`, `AutoModel.generate()` and
  `AutoModel.generate_iter()` take `messages` and `images`, so they run chat
  and image prompts on models such as Gemma 3 and InternVL.
  `GenerateResult.prompt_lengths` gives each prompt's length. `AutoModel`
  turns the vision encoder cache off by default.

- Added `AutoProcessor` to `max.pipelines`, which turns chat conversations
  with inline images into the `messages` and `images` that
  `AutoModel.generate()` takes.

- `AutoPipeline` now takes a chat conversation, or a list of them, with
  images inline, and returns only the new text for each one.

## C API

## Kernels and GPU programming

- The deprecated `LayoutTensor` and `LayoutTensorIter` types have been removed
  from the `layout` package, along with the `to_layout_tensor()` and
  `lt_to_tt()` conversions. Use `TileTensor` and `TileTensorIter` instead.
  `ThreadScope` now lives in `layout.tile_io`.

- `TileTensor.tile()` on a nested layout now indexes each mode first leaf
  first, as `crd2idx` and the CuTe `local_tile` operation do: tile `c` of size
  `t` starts at element `c * t` of the mode, and the tile keeps the leading
  leaves it spans, with their strides. Before, it treated the first leaf of
  each mode as the tile grid and the second as the fragment. To extract
  fragments, put the fragment first in each mode, as `blocked_product()` does:
  `((3, 2), (4, 2))` rather than `((2, 3), (2, 4))`. A tile size that doesn't
  split a nested mode on leaf boundaries is now a compile-time error.

- Removed the `max`, `min`, `sum`, `product` and `mean` overloads in
  `max.algorithm` that take an input shape, an `input_fn`, an `output_fn` and
  a `reduce_dim`. To reduce along an axis, use the graph ops (`ops.sum`,
  `ops.max`, `ops.min`, `ops.mean`, `ops.prod`), or call `map_reduce` once
  per row. The `Span` overloads and the 1-D `sum` and `mean` overloads that
  take a length and an `input_fn` are unchanged.

## Tools

- Added the ability to install `max-serve` and `max-benchmark` as Conda extras,
  using `max[extras=[serve]]` and `max[extras=[benchmark]]` respectively.
  `max[extras=[all]]` installs both plus `mojo`.

## Fixes

- `max serve` now runs prompts given as token ids, with no images, on Gemma 3,
  InternVL, Idefics3, Pixtral and Qwen 2.5 VL. These requests used to fail in
  the tokenizer.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
