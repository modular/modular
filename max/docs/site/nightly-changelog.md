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

- InternVL and Gemma 3 multimodal models now return log probabilities, so
  `logprobs` in `max serve` and `output_logprobs` in `generate()` work for
  them.

- Added support for
  [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
  on the `LagunaForCausalLM` architecture. The checkpoint interleaves 12
  full-attention layers with 36 sliding-window layers on a 512-token window,
  gates attention output per head rather than per element, and mixes expert
  precisions: NVFP4 routed experts in layers 1-39 and bfloat16 ones in layers
  40-47, with the dense first layer and every shared expert in bfloat16
  throughout. Attention type, query-head count, RoPE scaling and MLP precision
  are now all resolved per layer, so `poolside/Laguna-M.1-NVFP4` keeps its
  uniform layout.

## Inference server

- `/v1/completions` now returns a 400 error for `echo` with `logprobs` on a
  server started without `--enable-echo`, when the model computes log
  probabilities. The request used to crash the model worker. Other models
  still return no log probabilities, and `echo` without `logprobs` has no
  effect.

- Qwen 3.5 now passes every `chat_template_kwargs` entry to its chat template,
  such as `add_generation_prompt` or `preserve_thinking`. It used to keep only
  `enable_thinking`, which still defaults to on.

- MAX Serve's span attributes now sit under the entity they describe:
  `max.request_id` is now `max.request.id`, `max.batch_id` is `max.batch.id`,
  and the other request and batch attributes move under `max.request.*` and
  `max.batch.*`. `kernel_agg:<name>` spans carry the kernel name in
  `max.kernel.op` instead of `max.kernel.name`; per-kernel spans keep
  `max.kernel.name`. Update trace queries that filter on the old names.

### Server metrics

- `maxserve_batch_execution_time_milliseconds` now measures the executor
  call when overlap is off. The former full scheduler-step duration is
  available as `maxserve_scheduler_batch_step_time_milliseconds`. Batch
  throughput uses the executor interval and counts returned output tokens,
  including prefill outputs and accepted speculative tokens. See the
  [batch timing contract](/serve/metrics#batch-scheduler) for overlap timing.
  Intermediate prefill chunks returning no tokens now contribute zero to
  the CE generation-throughput histogram, shifting its distribution on
  chunked-prefill workloads. The benchmark harness's
  `prefill_batch_execution_time_ms` and `decode_batch_execution_time_ms`
  columns use `maxserve_batch_execution_time_milliseconds`, so MAX's
  synchronous values now exclude scheduler work; comparisons with older
  builds are not like for like.

- Added `maxserve_request_client_disconnect_cancellations_total`, which
  counts streamed responses cut short by a client disconnect and completion
  requests abandoned while waiting for a result, split by `phase`. A
  mid-stream disconnect was previously invisible:
  `maxserve_request_count_total` records it under the status code already
  sent.

## `max` CLI

## Python API

- A custom op whose Mojo kernel is registered wrong, or doesn't match its
  call, now raises `max.graph.KernelRegistrationError`, a `ValueError`
  subclass. The message points at the Mojo code the way the Mojo compiler
  does: the file, line and column, the source line and a caret. A custom op
  package's kernels are now checked when it loads, so a misplaced
  `DeviceContext` argument fails there instead of while compiling the
  generated code.

- Source tracebacks (`InferenceSession.debug.source_tracebacks`) now stop at
  your code's call into the graph API, so errors point at the lines that
  created each op rather than at MAX's implementation of it. Set
  `source_tracebacks_hide_internal = False` to keep every frame. Errors
  that `Model.execute` raises are now `max.engine.ModelExecutionError`, a
  `RuntimeError` subclass that is also a `ValueError`, so existing
  `except ValueError` handlers keep working.

- `max.nn.Transformer`, `DistributedTransformer`, `logits_postprocess()` and
  `distributed_logits_postprocess()` return a
  `max.nn.graph_groups.DecoderOutputs` of named groups instead of a tuple.
  Read the logits as
  `outputs.logits["logits"]` rather than `outputs[1]`, and pass the outputs to
  `Graph.output(outputs)` rather than unpacking them. The graph's outputs keep
  their order, and `Graph.output_tree` returns them by name.

- `ops.matmul()` takes an `out_dtype` keyword argument that sets the result
  dtype. The inputs keep their own dtypes and are no longer cast to a common
  one, so `bfloat16` inputs with `out_dtype=DType.float32` keep the float32
  accumulator. Without `out_dtype`, `ops.matmul()`, the `@` operator and
  `max.experimental.functional`'s `matmul()` require both inputs to share a
  dtype, which the result keeps, so `max.nn.Linear` raises when its input and
  weight dtypes differ.
  `max.experimental.nn.Linear` casts its input to the weight's dtype, so a
  float32 input to a bfloat16 `Linear` now gives bfloat16 instead of float32.

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

- `generate()`, `GenerateIterator`, `AutoModel.generate()`,
  `AutoModel.generate_iter()` and `AutoPipeline` take `chat_template_kwargs`,
  which reach the chat template of every `messages` row, such as
  `{"enable_thinking": False}` to turn off Qwen 3.5's thinking.

- `images` on `generate()`, `GenerateIterator`, `AutoModel.generate()` and
  `AutoModel.generate_iter()` takes PIL images as well as encoded bytes, and
  the tokenizer uses a PIL image without decoding it again. `AutoProcessor`
  now passes a copy of each PIL image instead of encoding it.

- `AutoProcessor` and `AutoPipeline` now read images from http(s) URLs, `data:`
  URIs and base64 data, including OpenAI `image_url` parts. URLs are fetched
  with a 30 second timeout and a 100 MiB limit per image.

- Errors from `AutoModel.generate()` and `AutoModel.generate_iter()` now name
  `AutoModel`, not `generate()`. An image part that holds its own image, or
  an `image_url` part, now raises `ValueError` pointing to `AutoProcessor`,
  even when `images` gives the row the right count.

- Added `generate_images()` in `max.pipelines.lib.generate_images`, which runs
  an image generation pipeline such as FLUX.2 in your own process and returns
  one list of PIL images per prompt, in prompt order. Each prompt runs as the
  request `max serve` builds for it, so the model's defaults are the same as in
  serve.

- `AutoModel` now loads image generation models such as FLUX.2, and
  `AutoModel.generate(prompt=...)` runs them and returns a
  `GenerateImagesResult`. Text models keep the same `generate()` call and
  result. An argument for the other kind of model raises a `ValueError` that
  names it. Image models need `max_batch_size=1` and no device graph capture.
  `AutoPipeline` runs image models too, through
  `AutoModel.generate(prompt=...)`, and returns a prompt's list of images.
  On an image model, `AutoModel.generate_iter()` points to
  `AutoModel.generate(prompt=...)`, and `generate()` points to
  `generate_images()`.

- Removed the `blocking_sync`, `disable_timing` and `interprocess` parameters
  from the Mojo `DeviceContext.create_event()`, along with `EventFlags`. Events
  never record timing; use `DeviceContext.execution_time()` to time kernels.

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

- `TileTensor.load_linear()` and `TileTensor.store_linear()` have been removed.
  Use `TileTensor.load()` and `TileTensor.store()` with a `Coord`, which also
  work on nested layouts: `t.load_linear[4](IndexList[2](r, c))` becomes
  `t.load[4](Coord(r, c))`. On CPU, `load()` and `store()` default to the
  element's alignment rather than the full vector's, so pass `alignment=` to
  keep a wider one.

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

- `max serve` now returns a 400 for a prompt given as token ids, on
  completions, chat `prompt_tokens` or embeddings, when an id is below 0 or
  past the model's vocabulary. Such a request used to fail its whole batch.
  `generate()`, `GenerateIterator` and `AutoModel.generate()` raise
  `InputError` for such an id in `input_ids`. Embeddings requests the
  tokenizer rejects, such as a too-long prompt, now return a 400, not a 500.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases/).
