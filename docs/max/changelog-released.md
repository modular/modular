---
title: What's new
description: Release notes for each version of the Modular platform.
toc_max_heading_level: 2
---

Here's everything you should know about what's changed in each release.

## Nightly: v26.2

This version is still a work in progress.

See how to [install the nightly
release](/max/packages#install).

<!-- INSERT HERE : This line is required for post-process-docs.py -->

## v26.1 (2026-01-29)

### Highlights {#26-1-highlights}

The eager-style [`Tensor`](/max/api/python/tensor#max.tensor.Tensor) and
[`Module`](/max/api/python/nn/module#max.nn.module.Module) APIs are
now the primary API for model development, providing a PyTorch-like development
experience:

```python
from max import functional as F
from max.tensor import Tensor
from max.dtype import DType

x = Tensor.constant([1.0, -2.0, 3.0, -4.0, 5.0], dtype=DType.float16)
y = F.relu(x)
print(y)
# Tensor([1 0 3 0 5], dtype=DType.float16, device=Device(type=gpu,id=0))
```

If you want explicit control over the graph structure, you can
still build models with the [`Graph`](/max/api/python/graph/Graph) APIs.

For more details, see the [model developer guide](/max/develop/).

### Documentation {#26-1-docs}

- The fully refactored [MAX LLM book](https://llm.modular.com/) is now designed
so the code you write in each exercise incrementally builds upon the last one,
until you've built an executable GPT-2 model with the MAX Python API.

- New model developer guide introduces [eager-style
programming](/max/develop/), [tensor APIs](/max/develop/tensors), and [data
types](/max/develop/dtypes). Much more is coming soon.

- New guide to [profile MAX on GPUs with `nsys`](/max/gpu-system-profiling).

- Extended [documentation for
`kbench`](https://github.com/modular/modular/tree/main/max/kernels/benchmarks/autotune#kbench-a-benchmarking-toolkit-for-mojo-kernels),
a Python tool to benchmark, autotune, and analyze MAX kernel performance.

### MAX models {#26-1-models}

- [Gemma3](https://builds.modular.com/models/gemma-3-it/27B) now supports
vision input (multimodal) in the 12B and 27B variants, including support for
local file paths and structured output. Learn more in the [image to text
guide](/max/inference/image-to-text).

- Added `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3-VL-2B-Instruct`
  model architectures.

- Removed Llama 3.2 Vision (`Llama-3.2-11B-Vision-Instruct`) architecture support.
  Use other vision models such as Pixtral, InternVL, Qwen2.5-VL, and Gemma3.

### MAX framework {#26-1-max}

- All Python wheels are now hosted at `https://whl.modular.com/nightly/simple/`.
  If using `uv`, change `--index-url` to `--index`, and if using `pip`, change to
  `--extra-index-url`. For precise commands, see the
  [install guide](/max/packages#install).

#### Inference server {#26-1-max-serve}

- Improved scheduling to achieve higher KVCache utilization and batch sizes. By
default, MAX now schedules a context encoding (CE) request only if KVCache
memory is less than 95% full _after_ allocating blocks for that request or if
no active requests exist. You can adjust this watermark value (`0.95`) with
[`--kvcache-ce-watermark`](/max/cli/serve#--kvcache-ce-watermark-kvcache_ce_watermark).
Beware that increasing it causes more preemptions.

- When running models with data-parallelism (DP), the semantics of max batch size
  has changed. For example, when specifying `--data-parallel-degree 8` and
  `--max-batch-size 32` it previously meant that each data-parallel replica could
  have at most 4 requests for an aggregate max batch size of 32. We changed this
  so that now the CLI flag specifies the max batch size per replica. This means
  the aggregate max batch size of the above values is 8*32=256 requests.
  This aligns with vLLM and other inference engines.

- `--max-ce-batch-size` is now deprecated. The cap on batch size is now uniform
  between context encoding and token generation phases of text generation. Use
  `--max-batch-size` instead.

- The API server now returns chunked tokens from the model worker, reducing overhead
  and significantly improving throughput for small models and decode-heavy
  workloads.

- Server stats collection (`collect_server_stats`) is now enabled by default for
  serving benchmarks.

#### `max` CLI {#26-1-max-cli}

- The `max generate` command now applies the model's chat template internally
  when using `--prompt`. This more closely aligns with how users typically prompt
  a model for testing and ensures special tokens are properly filtered from
  output.

- Added tracing flags to `max benchmark` for `nsys` profiling:

  - `--trace`: Enable tracing of the benchmark run (currently NVIDIA GPUs only)
  - `--trace-file`: Path to save the trace file
  - `--trace-session`: Optional session name for tracing

  Requires the server to be run under `nsys launch`. Using
  `--gpu-profiling detailed` is recommended.

#### Python API {#26-1-max-python}

- The eager-style [`Tensor`](/max/api/python/tensor#max.tensor.Tensor) APIs are
now the primary API for model development, providing a PyTorch-like development
experience.

  We moved the eager-style tensor APIs out of `experimental` and
  reorganized the `max.nn` module to make the eager module
  system the primary API (`nn.module_v3` is now `nn.module`).

  The previous [`max.nn`](/max/api/python/nn/) components are still available
  for backward compatibility in [`max.nn.legacy`](/max/api/python/nn/legacy/).

- Renamed `max.driver.Tensor` to
[`max.driver.Buffer`](/max/api/python/driver#max.driver.Buffer) to clarify that
it represents a low-level memory buffer, not a tensor. The
[`max.tensor.Tensor`](/max/api/python/tensor#max.tensor.Tensor) class remains
the primary tensor type.

- Added `forward()` method to
[`Module`](/max/api/python/nn/module#max.nn.module.Module) to compute the
output—it behaves the same as invoking the object as a callable (the
`__call__()` method).

- `accelerator_count()` now returns a non-zero value when called on an Apple
  silicon system. This means you can use this code:

  ```python
  device = CPU() if accelerator_count() == 0 else Accelerator()
  ```

  And it defaults to using the available Apple silicon GPU. As a consequence,
  MAX graphs should in most cases be dispatched to run on Apple silicon GPUs.
  Note that most MAX models do not yet work on Apple silicon GPUs due to
  missing hardware-specific kernel pathways and other support, but this is an
  important step towards enabling MAX more broadly on Apple silicon GPUs.

- Added `max.nn.module.rope` containing rotary embedding implementations,
[`RotaryEmbedding`](/max/api/python/nn/rope/RotaryEmbedding) and
[`TransposedRotaryEmbedding`](/max/api/python/nn/rope/TransposedRotaryEmbedding).

- Added
[`ArchConfig`](/max/api/python/pipelines/interfaces#max.pipelines.lib.interfaces.ArchConfig)
and `ArchConfigWithKVCache`. Going forward, models that register with the MAX
architecture registry must define a config that implements this protocol

- Added `ops.complex.mul` for multiplying complex-valued tensors

- Added `calculate_virtual_device_count()`, `calculate_virtual_device_count_from_cli()`,
  `load_max_buffer()` to [`max.driver`](/max/api/python/driver/).

- Added [`TokenBuffer`](/max/api/python/interfaces#max.interfaces.TokenBuffer)
for token management.

- Renamed `prefill_chunk_size` to `max_batch_input_tokens`
  and `max_batch_context_length` to `max_batch_total_tokens`
  in [`PipelineConfig`](/max/api/python/pipelines/config/#max.pipelines.lib.config.PipelineConfig)
  and `TTSConfig` classes to better reflect their purpose in batch memory
  management.

  The corresponding CLI flags have also been renamed:
  `--prefill-chunk-size` is now `--max-batch-input-tokens` and
  `--max-batch-context-length` is now `--max-batch-total-tokens`.

- Fixed `max.driver.Buffer.to(stream)` to not copy (it return reference to
  the same tensor) when the stream is on the same device, even for GPU-pinned
  host memory.

- Removed deprecated `max.nn` convolution classes: `Conv2dV1`, `Conv1DV1`,
  `Conv3DV1`. Use `Conv2d`, `Conv1D`, `Conv3D` instead.

- Removed deprecated `max.nn` layer classes: `LinearV1`, `QLinearV1`,
  `GPTQLinearV1`, `MLPV1`, `EmbeddingV1`, `LayerNormV1`, `RMSNormV1`. Use
  `Linear`, `GPTQLinear`, `MLP`, `Embedding`, `LayerNorm`, `RMSNorm` instead.

- Removed `max.engine.MojoValue`

- Removed the deprecated `custom_ops_path` parameter from
  [`InferenceSession.load()`](/max/api/python/engine#max.engine.InferenceSession.load).
  Instead use the `custom_extensions` parameter.

- Added `graph.ops.shard_and_stack()`

- Removed unused `graph.weights.PytorchWeights`

### MAX kernels {#26-1-max-kernels}

- Improved performance for Hopper Matmul when using skinny M shapes. In particular
when M is between 2 and 64, we see a significant performance boost for specific
shapes ranging between 10 - 40%.

- Added swapAB optimization to Hopper Matmul, performs B x A and does a tranposed
write to C. This helps when you need more granularity in the M dimension.

- Refined `create_stream` API: all streams are now non-blocking (`blocking`
  argument has been removed). Explicitly use `DeviceEvent` and `synchronize()`
  wherever necessary.

<!-- Please place Layout/LayoutTensor changes under "Library changes" in the
     **Mojo changelog**, since the layout package is packaged with and
     documented alongside Mojo. -->

### Mojo language {#26-1-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog)

## v25.7 (2025-11-20)

### Highlights {#25-7-highlights}

- The MAX Python API is now [fully open-sourced on
GitHub](https://github.com/modular/modular/tree/main/max/python/max)!

  As we expand our [model
  repository](https://builds.modular.com/?category=models), we're making
  significant progress on these APIs to simplify the effort to build
  production-ready GenAI models in Python. Some APIs are still experimental,
  but you can [build an LLM with it today](https://llm.modular.com).

### Documentation {#25-7-docs}

- New online book to [build an LLM from scratch with
MAX](https://llm.modular.com), using our **experimental model APIs**. This is a
guided lesson to building GPT-2 with our Python API, explaining each component
of the transformer model along the way. Like the Python APIs, the book is a
work in progress—please [report any issues in
GitHub](https://github.com/modular/max-llm-book/issues).

- All the planned parts of [GPU Puzzles](https://puzzles.modular.com/) are now
complete! Support for Apple silicon GPUs is also making [steady
progress](https://puzzles.modular.com/howto.html#gpu-support-matrix).

- Tutorials on docs.modular.com are now integrated into the
[Guides](/max/intro) section, indicated with a book icon in the left
navigation.

- The [`max` CLI docs](/max/cli/) are now generated from [the CLI
source](https://github.com/modular/modular/blob/main/max/python/max/entrypoints/pipelines.py).

### MAX models {#25-7-models}

- Gemma3 now supports logprobs.

### MAX framework {#25-7-max}

- Added support for bfloat16 models running on GPUs with ARM-based CPU hosts,
such as Grace Hopper (GH200) and Grace Blackwell (GB200).
- Updated minimum NVIDIA GPU driver requirement to 580.

#### `max` CLI {#25-7-max-cli}

- [`max benchmark`](/max/cli/benchmark) can now run LoRA benchmarking for
supported models and target modules.

- `max benchmark --collect-gpu-stats` can now collect AMD
GPU statistics.

- `max serve --do-penalties` was renamed to `--enable-penalties` and enabled by
default. To disable penalties, you can specify
[`--no-enable-penalties`](/max/cli/serve#--enable-penalties---no-enable-penalties)

#### Python API {#25-7-max-python}

- Added support for Python 3.14.

- Removed support for Python 3.9.

- All MAX Python API modules are now **open-sourced**. In addition to those
previously released, we've added `driver`, `dtype`, `engine`, `experimental`,
`interfaces`, `kv_cache`, `mlir`, `nn`, `profiler`, `support`, `torch`, and
more [in our GitHub
repo](https://github.com/modular/modular/tree/main/max/python/max).

- Added [`max.profiler`](max/api/python/profiler) module with the
[`Tracer`](/max/api/python/profiler#max.profiler.Tracer) class to create and
manage profiling spans based on runtime conditions, and the
[`@traced()] decorator to profile a whole function.

- Added [`max.diagnostics.gpu`](/max/api/python/diagnostics/gpu) APIs to expose
  common GPU statistics as might be reported by `nvidia-smi` or `rocm-smi`.

- Added the [`max.kv_cache`](/max/api/python/kv_cache/) package, which provides
APIs to manage key-value caches used in transformer models. Not to be confused
with the existing [`max.nn.kv_cache`](/max/api/python/nn/kv_cache/) package that
includes kernels for KV caching.

- Removed the `KVCacheManager` class and combined it with the single
[`PagedKVCacheManager`](/max/api/python/kv_cache/paged_cache/cache_manager#max.kv_cache.paged_cache.cache_manager.PagedKVCacheManager)
implementation. During merger, `prefetch()` was renamed `maybe_reserve()`.

- Added
[`NullKVCacheManager`](/max/api/python/kv_cache/null_cache_manager#max.kv_cache.NullKVCacheManager)
for compile-only mode, which avoids GPU memory allocation when compiling models
without a physical GPU present.

- Added
[`ResetPrefixCacheBackend`](/max/api/python/kv_cache/paged_cache/tp_cache_manager#max.kv_cache.paged_cache.ResetPrefixCacheBackend)
and
[`ResetPrefixCacheFrontend`](/max/api/python/kv_cache/paged_cache/tp_cache_manager#max.kv_cache.paged_cache.ResetPrefixCacheFrontend)
classes for coordinating prefix cache resets between frontend and backend
components.

- Added more APIs for text-to-speech (TTS) models such as
[`AudioGenerationInputs`](/max/api/python/interfaces#max.interfaces.AudioGenerationInputs)
and
[`AudioGenerationOutput`](/max/api/python/interfaces#max.interfaces.AudioGenerationOutput)

- Changed
[`LoRAConfig.max_num_loras`](/max/api/python/pipelines/lora_config#max.pipelines.lib.lora_config.LoRAConfig.max_num_loras)
default to `1` (was `100`).

- New [`RequestID`](/max/api/python/interfaces/#max.interfaces.RequestID) class
replaces previous type alias to provide better type safety and consistency
across the API.

- Removed `InputContext` and replaced it with the modality-output specific
[`TextGenerationContext`](/max/api/python/interfaces/#max.interfaces.TextGenerationContext)
and
[`EmbeddingsContext`](/max/api/python/interfaces/#max.interfaces.EmbeddingsContext).

- Added
[`ImageMetadata`](/max/api/python/interfaces/#max.interfaces.ImageMetadata) and
[`VLMTextGenerationContext`](/max/api/python/interfaces/#max.interfaces.VLMTextGenerationContext).

- Added [`max.nn.comm`](/max/api/python/nn/comm/) with `Allreduce` and
`Signals` for peer-to-peer communication in allreduce.

- [`ops.gather()`](/max/api/python/graph/ops#max.graph.ops.gather) no longer
has a default `axis`, it must be specified explicitly (better matching PyTorch
and NumPy).

- [`Graph.add_subgraph()`](/max/api/python/graph/Graph#max.graph.Graph.add_subgraph)
has been updated to take a `devices` argument. This allows subgraphs to take
advantage of device-aware work scheduling.

#### Mojo API {#25-7-max-mojo}

- Renamed the `tensor_internal` package to `tensor` and removed the
previous `tensor` stub—the API behaves the same but the [Mojo `tensor`
docs](/mojo/kernels/extensibility/tensor/) moved.

### Mojo language {#25-7-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming and `Layout`/`LayoutTensor` changes, see the [Mojo
changelog](/mojo/changelog).

## v25.6.1 (2025-10-10)

Fixes a latency regression due to a top-k algorithm change and a couple
other benchmarking bugs.

## v25.6 (2025-09-22)

- [Highlights](#25-6-highlights)
- [Documentation](#25-6-docs)
- [MAX models](#25-6-models)
- [MAX framework](#25-6-max)
  - [Inference server](#25-6-max-serve)
  - [`max` CLI](#25-6-max-cli)
  - [Python API](#25-6-max-python)
- [MAX kernels](#25-6-kernels)
- [Mojo language](#25-6-mojo)

### Highlights {#25-6-highlights}

- MAX delivers **state-of-the-art performance on NVIDIA Blackwell** (B200)!

  We've been describing our Blackwell bring-up over a series of blog posts, and
  we recently published [Part 4: Breaking
  SOTA](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota),
  in which we share our latest matmul benchmarks compared to NVIDIA's cuBLAS
  library.

- MAX provides **industry-leading performance on AMD MI355X**!

  In a matter of weeks, we got MAX running on the brand new MI255X system and
  have already produced early benchmarks that go head-to-head with Blackwell.
  If you have access to an MI355X, you can try it yourself today by following
  our [quickstart guide](/max/get-started).

- Benchmarking endpoints is easier than ever before the new [`max
benchmark`](/max/cli/benchmark) command, which accepts YAML
configuration files so you can easily share and reproduce your benchmarks.

### Documentation {#25-6-docs}

- Our new [quickstart guide](/max/get-started) lets you pick the model
architecture and size you want, and then shows you how to deploy it and run our
open-source benchmarking script, all from the `max` CLI.

- We updated and simplified the [benchmarking
tutorial](/max/deploy/benchmark) to use the new `max benchmark`
command.

### MAX models {#25-6-models}

- Added the
[gpt-oss](https://github.com/modular/modular/tree/modular/v25.6.0/max/pipelines/architectures/gpt_oss)
model architecture (GPU, bfloat16).
[Try GPT-OSS now](https://builds.modular.com/models/gpt-oss-20b-BF16/20B).

### MAX framework {#25-6-max}

- Added device-aware work scheduling for AsyncRT: work items can now specify a
  `deviceHint` to route execution to specific worker threads based on device
  affinity, improving multi-device performance.

- Improved code quality by enabling large set of RUFF lints, including
  [flake8-annotations (ANN)](https://docs.astral.sh/ruff/rules/#flake8-annotations-ann)
  which now enforces Python type annotations for new contributions.

#### Inference server {#25-6-max-serve}

- Added support for data parallelism in Llama models. To enable this feature,
  use the `--data-parallel-degree` option:

  ```sh
  max serve --model $MODEL_ID --data-parallel-degree 2 --devices gpu:0,1
  ```

- Metrics for each context encoding and token generation batch are now logged
  to the console periodically. We can override the default frequency (3 seconds)
  of such logs via setting the `MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S` flag.
  For example, setting `MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S=0` will log
  metrics for all batches.

- Improved error messages when pulling a model that requires more RAM than
  what's available or when there won't be enough RAM left for the KV cache.

#### `max` CLI {#25-6-max-cli}

- Added the `max benchmark` subcommand that runs a suite of benchmarks and
  collects performance metrics on a model server. This command provides
  convenient packaging/installation for our open-source
  [`benchmark_serving.py`](https://github.com/modular/modular/tree/main/benchmark#benchmark-max)
  script and accepts all the same options.

- Added `--chat-template` to the CLI for passing a custom chat templates
  defined in Jinja2 template files.

- Renamed the `--allow-safetensors-weights-float32-to-bfloat16-cast` flag to
  `--allow-safetensors-weights-fp32-bf6-bidirectional-cast`, which supports
  automatic bidirectional dtype casts when needed.

- The `max generate` command now supports `--top-k`, `--temperature`, and
  `--seed` flags.

- Changed `--num-warmups` behavior. Previously, it ran the model on the prompt
  `N` times, generating until reaching a stop condition each time. Now it runs
  the model for `N` steps, generating `N` new tokens as a warmup.

- Added the `--model` option as a preferred alternative to `--model-path`. They
  behave the same.

- Deprecated `--pad-to-multiple-of`.

- Removed the previously deprecated `--model-name`. Use `--served-model-name`
instead.

#### Python API {#25-6-max-python}

- Removed the previously deprecated `KVCacheStrategy.CONTINUOUS` and all
  associated classes (including `ContinuousBatchingKVCacheManager`).

- Added [`ops.fence`](/max/api/python/graph/ops#max.graph.ops.fence), a pure
  identity operation that prevents the async runtime from reordering operations
  across it. This operation is essential for implementing cross-device
  synchronization.

- Removed `PipelineConfig.max_new_tokens`. Use
  [`SamplingParams.max_new_tokens`](/max/api/python/pipelines#max.pipelines.SamplingParams)
  instead.

- Added
  [`logits_processor`](/max/api/python/interfaces/#max.interfaces.SamplingParams.logits_processors)
  to
  [`SamplingParams`](/max/api/python/interfaces/#max.interfaces.SamplingParams)
  for updating logits in-place during each step of token generation.

- Added `generate()` to
  [`TextGenerationPipeline`](/max/api/python/pipelines/pipeline#max.pipelines.lib.pipeline.TextGenerationPipeline)
  and
  [`SpeculativeDecodingPipeline`](/max/api/python/pipelines#max.pipelines.SpeculativeDecodingPipeline),
  a convenience method for getting text generations. `generate_async()` is
  available for getting streamed outputs.

- Renamed the `target_num_new_tokens` configuration parameter to
  [`prefill_chunk_size`](/max/api/python/pipelines/config/#max.pipelines.lib.config.PipelineConfig.prefill_chunk_size)
  in
  [`PipelineConfig`](/max/api/python/pipelines/config/#max.pipelines.lib.config.PipelineConfig)
  and `TTSConfig` classes to better reflect its role in chunked prefill
  operations.

- Fixed [`ops.range`](/max/api/python/graph/ops#max.graph.ops.range) to respect
  the `dtype` parameter when using [`Dim`](/max/api/python/graph/dim) objects as
  inputs. Previously, the dtype was ignored and defaulted to int64.

- Made the `devices` argument in
  [`InferenceSession()`](/max/api/python/engine#max.engine.InferenceSession)
  required. To maintain the previous default behavior, use
  `InferenceSession(devices=[CPU()])`.

- Added an optional `logging` argument to
  [`InferenceSession()`](/max/api/python/engine#max.engine.InferenceSession).
  When set to `"op"`, this option enables operation launch output to stderr.

- Added [`max.nn.lora`](/max/api/python/nn/lora), providing
  Low-Rank Adaptation (LoRA) support for parameter-efficient fine-tuning of
  neural network models.

- Added [`max.nn.moe`](/max/api/python/nn/moe), implementing
  Mixture of Experts (MoE) layers for scalable model architectures.

- Added [`max.nn.sampling`](/max/api/python/nn/sampling),
  containing advanced sampling methods including MinP and rejection sampling
  techniques.

- Added [`max.nn.hooks`](/max/api/python/nn/hooks), providing
  debugging and inspection hooks for neural network layers.

- Added attention submodules
  [`max.nn.attention.mask_config`](/max/api/python/nn/attention/mask_config),
  [`max.nn.attention.multihead_attention`](/max/api/python/nn/attention/multihead_attention),
  and
  [`max.nn.attention.multi_latent_attention`](/max/api/python/nn/attention/multi_latent_attention)
  for comprehensive attention mechanism configuration and implementation.

- Moved some Mojo-related functionality to a new top-level `mojo` Python
namespace. Specifically, `max.mojo` (previously used for Mojo-Python interop),
some of `max.support`, and `max.entrypoints.mojo` now live under the `mojo`
namespace and are provided in the new [`mojo`
package](/mojo/manual/install#whats-included).

### MAX kernels {#25-6-kernels}

- Added a leaky ReLU activation function kernel.

- Added a specialized [RMS norm](/mojo/kernels/nn/normalization/rms_norm/)
  function kernel for the common case of `cols=128`, `bfloat16`.

### Mojo language {#25-6-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming changes, see the [Mojo
changelog](/mojo/changelog).

<!-- #########################
     ###### NEW RELEASE ######
     ######################### -->

## v25.5 (2025-08-05)

- [Highlights](#25-5-highlights)
- [Documentation](#25-5-docs)
- [MAX models](#25-5-models)
- [MAX framework](#25-5-max)
  - [Inference server](#25-5-max-serve)
  - [`max` CLI](#25-5-max-cli)
  - [Python API](#25-5-max-python)
- [Mojo language](#25-5-mojo)

### Highlights {#25-5-highlights}

- **OpenAI-compatible batch API**: The [`/v1/batches`
API](/max/api/serve#operation/createBatch) is now available with
[Mammoth](/mammoth/).

  We recently announced a [partnership with SF
  Compute](https://www.modular.com/blog/sf-compute) to make this API available
  through their dynamic GPU pricing marketplace. Their Large Scale Inference
  Batch API looks different from the `/v1/batches` API in Mammoth because it's
  a superset.

- **New `mojo` Conda package**: For Mojo-specific projects that run on CPUs and
GPUs, you can now install the bare essentials with the `mojo` Conda package
that's less than 900 MB on disk. For example, this now works:

  ```sh
  pixi add mojo
  ```

  The `mojo` Python package is not available for pip/uv yet.

  For a complete model-development and serving toolkit, you should still install
  the `modular` package (which includes `mojo` as a dependency).

- **Open-source graph APIs**: We've added the `max.graph` Python APIs to our
[GitHub
repo](https://github.com/modular/modular/tree/modular/v25.5.0/max/graph). We've
made great strides in recent months to simplify these APIs that help you build
high-performance models you can [serve with
MAX](/max/develop/serve-custom-model-architectures).

### Documentation {#25-5-docs}

- New [Serve custom model architectures
tutorial](/max/develop/serve-custom-model-architectures), with [example code
on
GitHub](https://github.com/modular/modular/tree/main/max/examples/custom-models).

- New guide for [using LoRA adapters with MAX](/max/serve/lora-adapters).

- Updated the [Deploy Llama 3 on GPU
tutorial](/max/tutorials/max-serve-local-to-cloud/) with instructions using
AMD MI300X (on Azure).

- Added [Pixi basics](/pixi), which is where we redirect all the now-removed
Magic docs (see our [announcement migrating Magic to
Pixi](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530)).

### MAX models {#25-5-models}

- Added support for
[Idefics3](https://github.com/modular/modular/tree/modular/v25.5.0/max/pipelines/architectures/idefics3)
model.

### MAX framework {#25-5-max}

- Removed all `torch` package dependencies.

  - Reduces the total installation size of `modular` (including
  dependencies) from 2.2 GB for CPUs and 6.5 GB for GPUs **down to 1.5 GB**, for
  all Python packages. Conda packages pull additional system dependencies so
  sizes may vary, but one example brings the size down from 9.8 GB to 2.0 GB.

  - `pip install` no longer requires the `--extra-index-url
  https://download.pytorch.org/whl/cpu` option (which was to avoid installing
  the GPU version of `torch` that has a lot of CUDA dependencies).

  - `uv pip install` no longer requires the `--index-strategy unsafe-best-match`
  option (which was to avoid package resolution issues with the above
  `--extra-index-url` option).

- Removed HuggingFace fallback for model pipelines not natively supported in
MAX (`PipelineEngine.HUGGINGFACE`), because it's almost never used and it
creates significant tech debt.

#### Inference server {#25-5-max-serve}

- Added the [`/health` endpoint](/max/api/serve/#operation/health) for service
readiness checks, used by tools like lm-eval to determine when the service is
ready to accept requests.

- [Prefix caching](/max/serve/prefix-caching) now uses a Mojo token hashing
operation. Previously we used the `hash()` method from the Python stdlib.
However, this resulted in noticeable CPU overhead and reduced GPU utilization.
In this release, we migrated the token hashing operation to an accelerated Mojo
implementation.

- Re-implemented the OpenAI API's `logprobs` and `echo` request
  parameters to eliminate an expensive device transfer.
  The `--enable-echo` flag, which previously incurred a significant performance
  penalty, is now 9-12x faster.

- Added support for `file://` URIs in image inputs for multimodal models. Local
  file access is controlled via the `MAX_SERVE_ALLOWED_IMAGE_ROOTS` environment
  variable, which specifies a list of allowed root directories. Files are read
  asynchronously using aiofiles for better performance under high load.

- Improved [function calling](/max/serve/function-calling) (tool use) to more
reliably extract JSON tool calling responses for Llama models in an
OpenAI-compatible format.

- Switched from XGrammar to
[llguidance](https://github.com/guidance-ai/llguidance) for generating
structured output (constrained decoding).

#### `max` CLI {#25-5-max-cli}

- Added `--vision-config-overrides` CLI option to override
  vision model configuration parameters. For example, to decrease InternVL's
  maximum dynamic patches from 12 to 6:

  ```bash
  max serve --model-path OpenGVLab/InternVL3-38B-Instruct \
    --vision-config-overrides '{"max_dynamic_patch": 6}'
  ```

- Removed `--ignore-eos` CLI argument. The full set of OpenAI chat and
  completion sampling parameters are now supported in the http requests. As
  such, the parameter can just be set via the http payload.

#### Python API {#25-5-max-python}

- Added the [`max.interfaces`](/max/api/python/interfaces) module. This module
should serve as a relatively import free module to hold all shared interfaces
across the MAX stack. Slowly we will be moving common interfaces to this
module. So far, we've moved the following from `max.pipelines.core`:

  - Moved `TextGenerationStatus`, `TextResponse`, `TextGenerationResponse`,
  `InputContext`, and `PipelineTask` into `max.interfaces`.

  - Moved all `TokenGeneratorRequest`-prefixed objects into `max.interfaces`
  and renamed with the `TextGenerationRequest` prefix.

  - Moved `TextGenerationStatus` to
  [`GenerationStatus`](/max/api/python/interfaces/#max.interfaces.GenerationStatus).

  - Moved `TextResponse` and `TextGenerationResponse` to
  [`TextGenerationOutput`](/max/api/python/interfaces/#max.interfaces.TextGenerationOutput).

  - Moved `EmbeddingsResponse` to
  [`EmbeddingsOutput`](/max/api/python/interfaces#max.interfaces.EmbeddingsOutput).

- Added [`ops.scatter_nd`](/max/api/python/graph/ops/#max.graph.ops.scatter_nd)
operation for scattering updates into a tensor at specified indices.

- Added [`ops.avg_pool2d`](/max/api/python/graph/ops/#max.graph.ops.avg_pool2d)
and [`ops.max_pool2d`](/max/api/python/graph/ops/#max.graph.ops.max_pool2d).

- Added [`max.torch.graph_op`](/max/api/python/torch#max.torch.graph_op)
interface to make it simple to embed larger MAX computations and models inside
PyTorch. These can use `max.nn` modules internally and may be used within
`torch.nn` modules, allowing the use of MAX subcomponents for access to our
high performance graph compiler and Mojo kernel library.

  ```python
  import torch
  import numpy as np
  import max
  from max.dtype import DType
  from max.graph import ops

  @max.torch.graph_op
  def max_grayscale(pic: max.graph.TensorValue):
      scaled = pic.cast(DType.float32) * np.array([0.21, 0.71, 0.07])
      grayscaled = ops.sum(scaled, axis=-1).cast(pic.dtype)
      # max reductions don't remove the dimension, need to squeeze
      return ops.squeeze(grayscaled, axis=-1)

  @torch.compile
  def grayscale(pic: torch.Tensor):
      output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
      max_grayscale(output, pic)  # Call as destination-passing style
      return output

  img = (torch.rand(64, 64, 3, device=device) * 255).to(torch.uint8)
  result = grayscale(img)
  ```

- Moved `AlgebraicDim`, `Dim`, `StaticDim`, and `SymbolicDim` out of `max.type`
and into [`max.graph.dim`](/max/api/python/graph/dim). You can still import
them directly from `max.graph`.

- Moved `Shape` out of `max.type` and into
[`max.graph.shape`](/max/api/python/graph/shape). You can still import it
directly from `max.graph`.

- Removed the ability to pass Python objects into models and have them returned
  as Mojo `PythonObject` types in the kernels.

- Removed `RandomWeights`.

- Removed `Model.execute_legacy()`. Instead use the
standard [`execute()`](/max/api/python/engine#max.engine.Model.execute) or
[`__call__()`](/max/api/python/engine#max.engine.Model.__call) methods.

- Removed TorchScript-related helper functions and APIs, including support for
`.pt` TorchScript files in custom extensions.

### Mojo language {#25-5-mojo}

For all the updates to the Mojo language, standard library, and tools,
including all GPU programming changes, see the [Mojo
changelog](/mojo/changelog).

<!-- #########################
     ###### NEW RELEASE ######
     ######################### -->

## v25.4 (2025-06-18)

- [Highlights](#25-4-highlights)
- [Documentation](#25-4-docs)
- [MAX models](#25-4-models)
- [MAX framework](#25-4-max)
  - [Inference server](#25-4-max-serve)
  - [`max` CLI](#25-4-max-cli)
  - [Python API](#25-4-max-python)
  - [Mojo API](#25-4-max-mojo)
  - [Custom ops](#25-4-custom-ops)
  - [GPU programming](#25-4-gpu-programming)
- [Mojo language](#25-4-mojo)

### ✨ Highlights {#25-4-highlights}

- **AMD GPUs are officially supported!**

  You can now deploy MAX with acceleration on AMD MI300X and MI325X GPUs, using
  the same code and container that works on NVIDIA GPUs. For the first time,
  you can build portable, high-performance GenAI deployments that run
  on any platform without vendor lock-in or platform-specific optimizations.

  For more details, including benchmarks, see our [Modular + AMD blog
  post](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus).

- **Now accepting GPU kernel contributions**

  Last month, we open-sourced the code for the CPU and GPU kernels that power
  the MAX framework, and now we're accepting contributions! For information
  about how to contribute and the sort of kernels most interesting to us,
  see the [MAX AI kernels contributing
  guide](https://github.com/modular/modular/blob/main/max/kernels/CONTRIBUTING.md).

- **Preview: Mojo interoperability from Python**

  This release includes an early version of a new Python-to-Mojo
  interoperability API. You can now write just the performance-critical parts
  your code in Mojo and call it from Python just like you're importing another
  Python library. Check out our docs to [call Mojo from
  Python](/mojo/manual/python/mojo-from-python).

### Documentation {#25-4-docs}

We've redesigned [builds.modular.com](https://builds.modular.com) and
[docs.modular.com](https://docs.modular.com) with a unified top navigation bar
that so you can more easily discover all the available docs and code resources.

New docs:

- [GPU Puzzles](https://builds.modular.com/puzzles/introduction.html): Several
new puzzles, including: 1D convolution op, softmax op, attention op,
embedding op, kernel fusion, custom backward pass, GPU functional programming
patterns, and warp fundamentals.

- [Using AI coding assistants guide](/max/coding-assistants): Learn how to use
large language models (LLMs) and coding assistants (such as Cursor and Claude
Code) to accelerate your development with Modular.

- [Build an MLP block as a graph module tutorial](/max/develop/build-an-mlp-block):
Learn how to create reusable `Module` components in your MAX graphs.

- [Write custom ops for PyTorch
tutorial](/max/develop/custom-kernels-pytorch) (Beta feature): Learn to write
high-performance GPU kernels for your PyTorch models with Mojo.

- [Profile MAX kernel
performance](https://github.com/modular/modular/blob/main/max/docs/kernel-profiling.md):
Learn how to set up Nsight Compute to profile your Mojo-based kernels on NVIDIA
GPUs.

Major updates:

- [Build custom ops for GPUs tutorial](/max/develop/build-custom-ops):
  Now includes how to write hardware-specific functions for CPUs and GPUs.

- [Optimize a matrix multiply custom op
tutorial](/max/develop/custom-ops-matmul): Migrated from a Recipe with
revisions to help you improve the performance of your GPU custom ops.

### MAX models {#25-4-models}

- Added the OLMo 2 model architecture
([`olmo2`](https://github.com/modular/modular/tree/modular/v25.4.0/max/pipelines/architectures/olmo2)).

  [Try OLMo 2 now](https://builds.modular.com/models/OLMo-2-1124/7B).

- Added Google's Gemma 3 multimodal model architecture
([`gemma3multimodal`](https://github.com/modular/modular/tree/modular/v25.4.0/max/pipelines/architectures/gemma3)).

  [Try Gemma3 now](https://builds.modular.com/models/gemma-3-it/1B).

- Added the Qwen 3 model architecture
([`qwen3`](https://github.com/modular/modular/tree/modular/v25.4.0/max/pipelines/architectures/qwen3)).

  [Try Qwen3 now](https://builds.modular.com/models/Qwen3/1.7B).

- Added the InternVL3 model architecture
([`internvl`](https://github.com/modular/modular/tree/modular/v25.4.0/max/pipelines/architectures/internvl)).
This is still a work in progress.

- GGUF quantized Llamas (q4_0, q4_k, and q6_k) are now supported with paged
  KVCache strategy.

### MAX framework {#25-4-max}

#### Inference server {#25-4-max-serve}

- Inflight batching no longer requires chunked prefill.

- Expanded token sampling logic, including top_k, min_p, min_new_tokens,
temperature.

- Extended sampling configuration to be per-request, e.g. different requests
  can ask for different sampling hyperparameters.

- Removed support for TorchScript and torch MLIR models.

#### `max` CLI {#25-4-max-cli}

- Added the `--use-subgraphs` flag to `max generate` to allow for the use of
  subgraphs in the model.

- Added the `--port` option to specify the port number with the `max serve`
command.

#### Python API {#25-4-max-python}

- Lots of new APIs in the [`max.nn`](/max/api/python/nn/) package.

- Added `max.mojo.importer` module to import Mojo code into Python. See the
docs for [calling Mojo from Python](/mojo/manual/python/mojo-from-python).

- Added
[`Graph.add_subgraph()`](/max/api/python/graph/Graph#max.graph.Graph.add_subgraph)
to allow for the addition of a subgraph to a graph.

- Added
[`Module.build_subgraph()`](/max/api/python/nn/module#max.nn.module.Module.build_subgraph)
to allow for the creation of a subgraph for a layer that inherits from
`Module`.

- Added the [`call`](/max/api/python/graph/ops#max.graph.ops.call) op
which allows for the execution of a subgraph.

- Added the [`fold`](/max/api/python/graph/ops#max.graph.ops.fold) op for
combining sliding blocks into a larger tensor.

- Added [`KernelLibrary`](/max/api/python/graph/KernelLibrary) as an argument
type for the [`Graph`](/max/api/python/graph/Graph) constructor.

- Added
[`QuantizationConfig`](/max/api/python/graph/quantization#max.graph.quantization.QuantizationConfig)
to specify quantization parameters for ops such as
[`qmatmul()`](/max/api/python/graph/ops#max.graph.ops.qmatmul).

- Added the `strict` argument to the
[`Module.load_state_dict()`](/max/api/python/nn/module#max.nn.module.Module.load_state_dict)
method. When `strict=True` (default), an error is raised if the `state_dict`
contains unused keys. When `strict=False`, extra keys are ignored. This helps
model developers identify missing implementations in their models.

- Added audio generator APIs for text-to-speech models (such as
[`AudioGenerator`](/max/api/python/pipelines/core#max.pipelines.core.AudioGenerator),
[`PipelineAudioTokenizer`](/max/api/python/pipelines/core#max.pipelines.core.PipelineAudioTokenizer),
[`TTSContext`](/max/api/python/pipelines/core#max.pipelines.core.TTSContext),
and others). This is still a work in progress.

- The
[`ops.masked_scatter()`](/max/api/python/graph/ops#max.graph.ops.masked_scatter)
function now requires naming the `out_dim` explicitly as it is data-dependent.
For example:

  ```python
  ops.masked_scatter(
      inputs_embeds, video_mask, video_embeds, out_dim="unmasked_inputs"
  )
  ```

- Deprecated the `CONTINUOUS` KVCache strategy
([`KVCacheStrategy`](/max/api/python/nn/kv_cache/cache_params/#max.nn.kv_cache.cache_params.KVCacheStrategy)).
Please use `PAGED` KVCache strategy instead.

- Removed the `Settings` argument from
[`LLM`](/max/api/python/entrypoints#max.entrypoints.llm.LLM) constructor. The
server is now automatically configured in the background without consuming an
HTTP port.

- Removed `Graph.unique_symbolic_dim()`.

- Removed `max_to_torch_type()` and `torch_to_max_type()` and replaced them with
[`DType.to_torch()`](/max/api/python/dtype#max.dtype.DType.to_torch) and
[`DType.from_torch()`](/max/api/python/dtype#max.dtype.DType.from_torch),
respectively. This aligns with the corresponding NumPy methods.

- Removed `stats_report` property and `reset_stats_report` method from
[`InferenceSession`](/max/api/python/engine#max.engine.InferenceSession). This
functionality was primarily used for internal PyTorch debugging and is no
longer needed.

- Removed the naive KVCache (`nn.kv_cache.naive_cache`).

- Removed `nn.attention` and `nn.naive_attention_with_rope`.

- Renamed `ops.select` to
[`ops.where`](/max/api/python/graph/ops#max.graph.ops.where). This matches the
name of the similar operation in torch and numpy.

#### Mojo API {#25-4-max-mojo}

- [`LayoutTensor`](/mojo/kernels/layout/layout_tensor/LayoutTensor/) now has a
`size` method to get the total number of elements.

- Following our [previous deprecation](#25-3-engine-mojo-api) of the Mojo
`max.driver`, `max.graph` and `max.engine` APIs, we've removed them from the
package and API docs.

  As a result, we've also removed Mojo `max.tensor` APIs (including
  `Tensor`, `TensorShape`, and `TensorSpec`). You can replace any use with
  [`LayoutTensor`](/mojo/kernels/layout/layout_tensor/LayoutTensor/).

#### Custom ops {#25-4-custom-ops}

- Improved error messages when custom op parameters are provided with values that
  don't have the proper type.

- The [`ops.custom()`](/max/api/python/graph/ops#max.graph.ops.custom) function
now requires a `device` argument to specify where the operation should execute.
This avoids the need for custom ops to infer their execution device, which can
be error-prone.

- Added the [`max.torch`](/max/api/python/torch) module with the
`CustomOpLibrary` class for using custom Mojo kernels from PyTorch. For
example, with a custom `grayscale` operation written in Mojo:

  ```mojo
  @register("grayscale")
  struct Grayscale:
      @staticmethod
      fn execute[
          # The kind of device this is running on: "cpu" or "gpu"
          target: StaticString,
      ](
          img_out: OutputTensor[dtype = DType.uint8, rank=2],
          img_in: InputTensor[dtype = DType.uint8, rank=3],
          ctx: DeviceContextPtr,
      ) raises:
          ...
  ```

  You can load it with PyTorch like so:

  ```python
  from max.torch import CustomOpLibrary

  op_library = CustomOpLibrary("path/to/custom.mojopkg")

  @torch.compile(backend=backend)
  def grayscale(pic):
      result = pic.new_empty(pic.shape[:-1])
      op_library.grayscale(result, pic)
      return result

  img = (torch.rand(64, 64, 3) * 255).to(torch.uint8)
  result = grayscale(img)
  ```

  See our [tutorial to write custom ops for
  PyTorch](/max/develop/custom-kernels-pytorch), and out [PyTorch custom
  operation
  examples](https://github.com/modular/modular/tree/main/max/examples/pytorch_custom_ops),
  which range from a very basic "hello world" to the replacement of a layer in
  a full model.

#### GPU programming {#25-4-gpu-programming}

- Full support for AMD CDNA3 datacenter GPUs is now available! Specifically,
MI300X and MI325X.

- Added initial support for programming on AMD RDNA3 consumer GPUs. Basic
tuning parameters have been specified for AMD Radeon 780m integrated GPUs. (AMD
RDNA3 support is for GPU programming only; AI models are still missing some GPU
kernels for this architecture.) For details, see the [GPU
requirements](/max/packages#gpu-compatibility).

- Now accepting CPU and GPU kernel contributions. See the [MAX AI kernels
contributing
guide](https://github.com/modular/modular/blob/main/max/kernels/CONTRIBUTING.md).

### Mojo language {#25-4-mojo}

For all the updates to the Mojo language, standard library, and tools, see the
[Mojo changelog](/mojo/changelog).

<!-- #########################
     ###### NEW RELEASE ######
     ######################### -->

## v25.3 (2025-05-06)

- [Highlights](#25-3-highlights)
- [Documentation](#25-3-docs)
- [`max` CLI](#25-3-max-cli)
- [MAX models](#25-3-models)
- [MAX Serve](#25-3-serve)
- [MAX Engine & Graph](#25-3-engine)
  - [Python API](#25-3-engine-mojo-api)
  - [Mojo API](#25-3-engine-mojo-api)
  - [Custom ops](#25-3-custom-ops)
- [Kernels](#25-3-kernels)
- [GPU programming](#25-3-gpu-programming)
- [Mojo language](#25-3-mojo)

### ✨ Highlights {#25-3-highlights}

- You can now **install Modular APIs and tools with pip**:

  ```sh
  pip install modular \
    --index-url https://download.pytorch.org/whl/cpu
  ```

  This installs the `max` CLI, `max` Python library, `mojo` CLI, and Mojo
  libraries. However, the Mojo LSP and debugger are currently not included.

  We use the `--index-url` argument to ensure that `torch` installs its CPU
  dependencies only, thus avoiding a lot of unnecessary GPU packages. This is a
  temporary workaround until we can remove our dependency on `torch`.

- We **open-sourced the MAX AI kernels** and the rest of the **Mojo standard
library**!

  The [MAX AI kernels library](/mojo/lib#max-ai-kernels-library) is a new Mojo
  API for writing high-performance and portable programs across CPU and GPU, but
  it's also [the source code for our CPU/GPU
  kernels](https://github.com/modular/modular/tree/main/max/kernels/src). You
  can now see the Mojo code we use in MAX to power GenAI workloads on CPUs and
  GPUs.

  Just like the Mojo standard library, these kernels are open source under the
  Apache 2.0 License with LLVM exceptions. Plus, the rest of the Mojo standard
  library is also [now open source in
  GitHub](https://github.com/modular/modular/tree/main/mojo/std/src).

- **Learn to program GPUs** with [Mojo GPU Puzzles](https://builds.modular.com/puzzles)!

  This is a brand new site that offers a hands-on guide to mastering GPU
  programming with Mojo. Starting from basic concepts, you'll learn
  step-by-step how to program for GPUs by solving increasingly challenging
  puzzles.

### Documentation {#25-3-docs}

We've restructured the documentation to unify MAX and Mojo documentation
under the Modular Platform. We believe this improves content discovery with a
simplified navigation and helps unify the platform story as a whole.

We've also added the following new docs:

- [REST API reference](/max/api/serve): Although it's not a new API (our
serving library has supported OpenAI APIs for the last few versions), this
now shows precisely which endpoints and body parameters we support.

- [Speculative decoding](/max/serve/speculative-decoding): An introduction to
using speculative decoding to reduce latency for LLMs. This feature is still in
development.

- [Offline inference](/max/serve/offline-inference): An introduction to our
Python API for running inference with an LLM locally (without sending requests
to a serving endpoint).

- [Introduction to layouts](/mojo/manual/layout/layouts): A guide to working
with dense multidimensional arrays on CPUs and GPUs, using new Mojo `layout`
types that abstract-away complex memory layout patterns.

### `max` CLI {#25-3-max-cli}

- Renamed the `max-pipelines` CLI tool to `max`. We recommend re-installing
  it as shown in the [`max` CLI docs](/max/cli/).

- Remove previously deprecated `--use-gpu`, `--serialized_model_path`,
`--save_to_serialized_model_path`, `--max_cache_batch_size` and
`--huggingface-repo-id` options.

- Move `InputContext`, `TextContext`, and `TextAndVisionContext` from
`max.pipelines` to `max.pipelines.context`.

### MAX models {#25-3-models}

- Added `Llama4ForConditionalGeneration` support,
  featuring new MoE layers. Currently, it is limited to text inputs.
  Run the model by calling:

  ```sh
  max generate --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --devices 0,1,2,3
  ```

- Added support for running text generations using the Mistral 3 24B model.
  Run the model with:

  ```sh
  max generate --model-path mistralai/Mistral-Small-3.1-24B-Instruct-2503 --devices 0
  ```

- Fixed empty textual outputs for certain Mistral models
  ([MAX issue 4193](https://github.com/modular/modular/issues/4193)).

- Added support for loading a custom pipeline architecture by module. Using
`--custom-architectures=folder/path/to/import:my_module` will lead to loading
architectures from the file. The architectures must be exposed via an
`ARCHITECTURES` variable in the file. Once loaded, a model can be run using the
new architectures. The flag can be specified multiple times to load more
modules.

### MAX Serve {#25-3-serve}

- Moved from radix trie to hash based prefix caching implementation which has
smaller CPU overheads. This improves performance particularly in workloads with
high cache reuse rates.

- Added experimental support for offloading KVCache to host memory via the
`--enable-kvcache-swapping-to-host` and `--host-kvcache-swap-space-gb` flags.
This allows for superior KVCache reuse through prefix caching in workloads
where the reusable KVCache amount exceeds GPU VRAM.

- Fixed the `usage.prompt_tokens` field in the OpenAI API Usage Info response.
Previously this field was always set to Null, but now it correctly
contains the number of prompt tokens in the request.

- Switched from Python Multiprocessing Queue to ZeroMQ. This reduces latencies
between frontend server process and model worker process related to networking.

- Stray model workers on Linux now terminate more reliably when the parent
  process is killed.

### MAX Engine & Graph {#25-3-engine}

#### Python API {#25-3-engine-python-api}

- We now raise an error if there's a mismatch between the expected device of a
weight on a graph and the device of the actual tensor data specified in
[`InferenceSession.load()`](/max/api/python/engine#max.engine.InferenceSession.load).

- Removed `output_device` argument from
[`Model.execute()`](/max/api/python/engine#max.engine.Model.execute).

- Removed the `copy_inputs_to_device` argument in
[`Model.execute`](/max/api/python/engine#max.engine.Model.execute) to improve
predictability of the API. Now `execute()` raises a `TypeError` if arguments
are passed whose devices don't match the model.

- Swapped the order of the `dtype` and `shape` fields of
[`driver.Tensor`](/max/api/python/driver#max.driver.Tensor).
Previously, the arguments are ordered as `(shape, dtype)`. They are now swapped
to `(dtype, shape)` to be in line with other tensor-like types.

- Replaced some instances of
[`Tensor.zeros`](/max/api/python/driver#max.driver.Tensor.zeros)
with `Tensor.__init__` when the engine did not depend on the tensor being zero
initialized. This elides the unnecessary memset to provide a minor performance
improvement.

- Added a new experimental
[`Tensor.inplace_copy_from()`](/max/api/python/driver#max.driver.Tensor.inplace_copy_from).
This allows users to copy the contents of one `Tensor` into another.

- Made the default behavior of [`Weight`](/max/api/python/graph/Weight) as
expecting the initial allocation on host. A transfer is then inserted to the
target device and this value is returned when weights generate an MLIR value.
This is done due to current conservative ownership around external weights.

- Added the [`irfft`](/max/api/python/graph/ops/#max.graph.ops.irfft) op, which
computes the inverse real fast fourier transform (FFT).

- Added the [`argmax`](/max/api/python/graph/ops#max.graph.ops.argmax) op,
which returns the index of the maximum value in an array or sequence.

- Added the [`GroupNorm`](/max/api/python/nn/norm/group_norm) layer.

- Switched layer names so that `max.nn` layers that are implemented with the
deprecated `Layer` class are marked as "V1", and layers that are implemented
with the new [`max.nn.Module`](/max/api/python/nn/module#max.nn.module.Module)
are the default. That is, `max.nn.LinearV2` is now
[`max.nn.Linear`](/max/api/python/nn/Linear), and the
previous `max.nn.Linear` is now
`max.nn.LinearV1`.

- DeviceRefs in types/layers are in general expected to be explicit rather than
  implicit.

#### Mojo API {#25-3-engine-mojo-api}

- Removed some functionality from
[`tensor.Tensor`](/mojo/kernels/extensibility/tensor/tensor/Tensor):

  - Serializing `Tensor` to disk (`Tensor.tofile(path)` and `Tensor.save(path)`).
  - Reading the serialized data back from disk (`Tensor.load(path)` and
    `Tensor.fromfile(path)`.
  - `rand` and `randn` methods have been removed.  Use the ones in the Mojo
    standard library if you still need access for constructing a new `Tensor`
    with random elements based on a particular `TensorShape`.

- **Deprecated the Mojo Driver, Graph, and Engine APIs**

  These APIs are not currently used internally. Instead, we build graphs using
  the Python APIs, and our engineering efforts have been focused on making that
  experience as robust and user-friendly as possible. As a result, the Mojo
  versions of these APIs have not kept pace with new features and language
  improvements. These APIs will be open sourced for the community before being
  removed.

#### Custom ops API {#25-3-custom-ops}

- You can now pass Mojo source package paths as
[`Graph`](/max/api/python/graph/Graph) custom extensions. The Mojo code will be
compiled automatically, no need to run `mojo package` manually as a prior step.
Previously, only pre-compiled `.mojopkg` paths were accepted, requiring the
Mojo code to be built as a prerequisite step before running a `Graph` with a
custom op.

  Given a project structure like:

  ```text
  project
  |-- main.py
  \-- kernels
      |-- __init__.mojo
      \-- my_custom_op.mojo
  ```

  You can construct a `Graph` in `main.py` using Mojo custom op kernels simply
  using:

  ```python
  g = Graph(
    ...,
    custom_extensions = [Path(__file__).parent / "kernels"]
  )
  ```

  A change to your Mojo source code defining a custom op will be reflected
  immediately the next time the `Graph` is constructed.

- New [image_pipeline example](https://github.com/modular/modular/tree/main/max/examples/custom_ops)
  that demonstrates sequencing custom ops together which modify an image,
  leaving data on the GPU for each op, before writing it back to CPU and disk.

### Kernels {#25-3-kernels}

- More compute overlap is now enabled for Hopper GPUs. This allows finer-grained
  scheduling of kernel operations by analyzing producer-consumer patterns within
  a compute kernel. As a result, there is more kernel compute overlap, especially
  for compute-heavy kernels with data-dependent execution paths.

### GPU programming {#25-3-gpu-programming}

- CUDA driver requirement reduced to version 12.4 and the NVIDIA driver to be
  version 550. Requiring these earlier driver versions allows MAX to be more
  easily deployed on AWS and GCP, since these are the default versions used by
  those cloud providers.

- Added support for programming NVIDIA Jetson Orin GPUs (`sm_87`).

Also see the [Mojo changelog of GPU changes](/mojo/changelog#gpu-changes).

### Mojo language {#25-3-mojo}

- We recently open-sourced the rest of the Mojo standard library, including the
`algorithm`, `benchmark`, `buffer`, `compile`, `complex`, `gpu`, and `layout`
packages. [See it all in
GitHub](https://github.com/modular/modular/tree/main/mojo/std/src).

- We've also open sourced [all our MAX AI
kernels](https://github.com/modular/modular/tree/main/max/kernels/src). This
new library includes `kv_cache`, `layout`, `linalg`, `nn`, `nvml`, and
`quantization`.

For all the updates to the Mojo language, standard library, and tools, see the
[Mojo changelog](/mojo/changelog).

<!-- #########################
     ###### NEW RELEASE ######
     ######################### -->

## v25.2 (2025-03-25)

- [Highlights](#25-2-highlights)
- [MAX Serve](#25-2-serve)
- [MAX models](#25-2-models)
  - [`max-pipelines` CLI](#25-2-pipelines-cli)
- [MAX Engine](#25-2-engine)
  - [Driver APIs](#25-2-driver)
  - [Graph APIs](#25-2-graph)
  - [Custom ops](#25-2-custom-ops)
  - [Hopper Kernels](#25-2-hopper-kernels)
- [GPU programming](#25-2-gpu-programming)
- [Mojo](#25-2-mojo)
- [Documentation](#25-2-documentation)

### ✨ Highlights {#25-2-highlights}

- **Support for NVIDIA Hopper GPUs**

  MAX has been optimized to run on Hopper GPUs. For more information on MAX and
  NVIDIA's hardware, see the [MAX
  container](/max/container#recommended-cloud-instances) documentation.

- **Multi-GPU support**

  MAX uses tensor parallelism to distribute work across multiple GPUs so you can
  run LLMs like
  [`Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
  even with long context window.

- **Expanded library of MAX models**

  We're rapidly growing our library of base model architectures that MAX can
  accelerate with MAX Serve (including `Phi3ForCausalLM`, `OlmoForCausalLM`,
  and `GraniteForCausalLM`). We also now support `GTPQ` for the Llama models.
  For more information, check out our [MAX model
  repository](https://builds.modular.com/?category=models).

- **Advanced E2E optimizations for long context window**

  In flight batching, chunked prefill, and copy-on-write optimize the execution
  for prefix heavy and long context window scenario.

- **GPU programming with Mojo**

  Lots of new APIs are now available to enable both low-level GPU programming and
  abstracted programming patterns that simplify the code required to write GPU
  kernels for your AI models.

### MAX Serve {#25-2-serve}

- Extended MAX Serve batch scheduling to account for the prefix cache. The
scheduler can now create larger batches when many prompt tokens are already
cached, improving throughput up to 10% in some benchmarks.

- Added support for in-flight batching, allowing token generation requests to be
scheduled alongside context encoding requests to reduce inter-token latency. This
behavior can be controlled by CLI argument `--enable-in-flight-batch`.

- Added support for copy-on-write on KV blocks when using PagedAttention with
Prefix Caching. This improves the prefix cache hit rate and prefill performance
in some scenarios.

- MAX Serve now supports `transformers` v.4.49.0, with a patch
to avoid graph breaks when using `torch.compile()` on Llama models.

- Added support for recording HTTP traffic out to a file for diagnostics or later
replay.

### MAX models {#25-2-models}

- Added support for executing `LlamaForCausalLM` architecture models on multiple
GPUs. The model uses tensor parallelism automatically when passing multiple
device IDs to the `--devices` CLI argument. Try running
[`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
on 4 GPUs with the following example:

  ```sh
  max-pipelines generate --model-path=meta-llama/Llama-3.3-70B-Instruct \
    --quantization-encoding bfloat16 \
    --devices gpu:0,1,2,3 \
    --prompt="Design a
      self-sustaining colony on Neptune's moon Triton with a myth/science
      fusion name, three quantum tech breakthroughs, one ethical debate, a
      neon-lit cultural ritual, and a hidden flaw—presented in bullet points."
  ```

- Added support for the `Phi3ForCausalLM` model architecture (such as
[`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4)). For example:

  ```sh
  max-pipelines generate \
    --model-path microsoft/phi-4 \
    --prompt "Write bubble sort in mojo"
  ```

- Added support for the `OlmoForCausalLM` model architecture (such as
[`allenai/OLMo-1B-0724-hf`](https://huggingface.co/allenai/OLMo-1B-0724-hf)). For
example:

  ```sh
  max-pipelines generate \
    --model-path allenai/OLMo-1B-0724-hf \
    --prompt "Write bubble sort in mojo"
  ```

- Added support for the `GraniteForCausalLM` model architecture (such as
[`ibm-granite/granite-3.1-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)).
For example:

  ```sh
  max-pipelines generate \
    --model-path ibm-granite/granite-3.1-8b-instruct \
    --prompt "Write bubble sort in mojo"
  ```

- Added support for:

  - [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
  - [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4)
  - [`LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
  - [`LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct`](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)

- We now support GPTQ quantization for models that run on the GPU. This is
handled transparently when the model weights are specified. For example, this
runs Llama 3.1 8B using int4-quantized GPTQ weights:

  ```sh
  max-pipelines generate \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --prompt "Why is the sky blue?" \
    --max-batch-size 1 \
    --max-length 10000
  ```

  This reduces the total memory consumption of this model from ~16 GB to ~5 GB,
  allowing the model to fit in the RAM smaller GPUs.

- Model weights are now downloaded in parallel.

- Added constraints on whitespace during [Structured
Output](/max/serve/structured-output). This reduces tokens counts and improves
model adherence.

- Added jump ahead decoding during Structured Output. This auto-completes tokens
when a singular path forward is identified, improving single completion times by
up to ~20% for long prompts.

- In the event of an unhandled exception, we now use the standard Python
traceback format instead of using pretty-printed Rich tracebacks.

- We now need to explicitly import `LLM` from
[`max.entrypoints.llm`](/max/api/python/entrypoints) rather than the previous
`max.entrypoints` import.

- The `max.pipelines.dataprocessing.tokenizer` and
`max.pipelines.dataprocessing.gguf_utils` modules have been removed.

- The previously deprecated `PipelineConfig.architecture` field and its
corresponding `--architecture` CLI argument have been removed.

### `max-pipelines` CLI {#25-2-pipelines-cli}

- The `--devices` CLI argument now supports a comma-separated list of GPU IDs
prefixed with `gpu:` like `--devices=gpu:0,1,2,3`. We no longer support the
previous `--devices=gpu-<N>` format.

  ```sh
  max-pipelines generate --model-path=meta-llama/Llama-3.3-70B-Instruct \
    --quantization-encoding bfloat16 \
    --devices gpu:0,1,2,3 \
    --prompt="Design a self-sustaining colony on Neptune's moon Triton with a myth/science fusion name, three quantum tech breakthroughs, one ethical debate, a neon-lit cultural ritual, and a hidden flaw—presented in bullet points."
  ```

- Removed `--huggingface-repo-id`
[PipelineConfig](/max/api/python/pipelines/config/#max.pipelines.config.PipelineConfig)
option and CLI argument in favor of `--model-path`.

- We consolidated `--model-path` and `-weight-path`. Valid `--weight-path` values
now override `--model-path`, which handles both local and remote (Hugging Face)
cases. If we cannot derive the weights from the `--weight-path`, we now fall back
to the `--model-path`, which you must set explicitly.

- Added `--huggingface-revision` option, to allow selecting a non-default branch
or a specific commit in a Hugging Face model repository.

### MAX Engine {#25-2-engine}

- The MAX graph compiler now has kernel caching. This is a significant
improvement to our compilation pipeline. Here are some of the highlights:

- Up to 28% faster compilation times when making iterative changes to models
- Improved caching between different but similar models (up to 27% faster)
- Lays foundation for future caching optimizations

What does this mean for you? Faster development cycles! When you're working on
model pipelines and making changes to the graph, the graph compiler will now
intelligently reuse kernels that haven't changed, significantly reducing
compilation times.

The improvements are particularly noticeable during iterative development, with
compilation times dropping from ~80s to ~57s in some cases of compiling
Llama3.1-8B for 4 GPUs. Even when compiling different models from the same family
(like Llama/Granite variants), you'll see significant speedups on subsequent
compilations.

### Driver APIs {#25-2-driver}

- Added `Accelerator.can_access(other: Device) -> bool` method to check if one
  device can directly access memory of another device.

- Fixed a bug in `max.driver.tensor.load_max_tensor()` for `bfloat16` dtype,
  which would cause an error about mmap size being too large.

- `max.driver.Tensor.item()` now works on any single-element tensor (previously
  restricted to rank-0 tensors).

- Added
[`Device.synchronize()`](/max/api/python/driver#max.driver.Device.synchronize),
which ensures all operations on the device complete before returning.

- Removed `MojoCallContextPtr` in favor of `DeviceContextPtr`.
`MojoCallContextPtr` only contained a `DeviceContextPtr`, so this change
directly exposes the `DeviceContextPtr`. Custom ops using `MojoCallContextPtr`
now directly take a `DeviceContextPtr` argument:

  ```mojo
      @staticmethod
      fn execute[
          type: DType, rank: Int
      ](
          output: OutputTensor[type=type, rank=rank],
          input: InputTensor[type=type, rank=rank],
          ctx: MojoCallContextPtr,
      ):
  ```

  becomes

  ```mojo
      @staticmethod
      fn execute[
          type: DType, rank: Int
      ](
          output: OutputTensor[type=type, rank=rank],
          input: InputTensor[type=type, rank=rank],
          ctx: DeviceContextPtr,
      ):
  ```

- You can now skip compiling a GPU kernel first before enqueueing it, and pass
a function directly to `ctx.enqueue_function[func](...)`:

  ```mojo
  fn func():
      print("Hello from GPU")

  @register("custom_op")
  struct CustomOp:

      @staticmethod
      fn execute(ctx: DeviceContextPtr) raises:
          var dev_ctx = ctx.get_device_context()
          dev_ctx.enqueue_function[func](grid_dim=1, block_dim=1)
  ```

  However, if you're reusing the same function and parameters multiple times, this
  incurs some overhead of around 50-500 nanoseconds per enqueue. So you can still
  compile the function first and pass it to `ctx.enqueue_function` in this scenario:

  ```mojo
  var compiled_func = ctx.compile_function[func]()
  # Multiple kernel launches with the same function/parameters
  ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
  ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
  ```

- Changed `Accelerator` and `CPU` from factory methods that created `Device`
objects in Python (which were accelerators and CPUs in the C++ implementation) to
actual Python types. This change elevates the `Accelerator` and `CPU` type
concepts to Python, making them types rather than methods.

  This allows type annotations in Python. For example, a list of accelerators
  used to be defined like this:

  ```python
  graph_devices: list[DeviceRef]
  ```

  Now it can be defined like this:

  ```python
  graph_devices: list[Accelerator]
  ```

- Elementwise operations (e.g. `__add__`) have been removed from `Tensor`
  (that is, `tensor_internal.Tensor`). This `Tensor` type is being phased out; please
  reduce usage in favor of `LayoutTensor`.

### Graph APIs {#25-2-graph}

- The `nn` package is now [`max.nn`](/max/api/python/nn/).

- Added [`ops.chunk`](/max/api/python/graph#max.graphs.ops.chunk)) to support
chunking tensors along an axis.

- Added support for while loops with [`ops.while_loop`](/max/api/python/graph#max.graphs.ops.while_loop).

- Added support for conditional execution with [`ops.cond`](/max/api/python/graph#max.graph.ops.cond).

- Added axis reduction overloads for
[`ops.min`](/max/api/python/graph/ops#max.graph.ops.min) and
[`ops.max`](/max/api/python/graph/ops#max.graph.ops.max). For example;
`ops.min(tensor, axis=-1)`.

- The [`gelu()`](/max/api/python/graph/ops#max.graph.ops.gelu) function now accepts
an `approximate` keyword. The keyword controls the `gelu` approximation with
`none`, `tanh`, and `fast` approximations accepted.

- Removed the `roundeven()` operation from the Python API. The
[`round()`](/max/api/python/graph/ops#max.graph.ops.round) operation now has the
same behavior as `roundeven()`, so there is no need for both to exist.

- Added helpers to create analogous tensors from buffer types and vice versa.

- Added `max.nn.Module`, a base class for writing layers and constructing
networks of layers (e.g. using `max.nn.Sequential`). Currently, this class
supports graph building by ensuring that all weight names are unique and
systematically generated. This class also supports managing the weight values
with the `module.state_dict()` and `module.load_state_dict()` methods. More
functionality and documentation will be added in future releases.

### Custom ops {#25-2-custom-ops}

- Changes have been made to the way that custom ops are registered: rather
than using the `num_dps_outputs` attribute on `@compiler.register` to specify the
number of outputs, that number is now inferred from the signature of the custom
operation. Inputs to the operation now use the `InputTensor` type and outputs
from the operation use `OutputTensor`, instead of the previous
`ManagedTensorSlice` for both. This eliminates the need for a manual
`num_dps_outputs` attribute, and makes it safer to work with these inputs and
outputs by preventing accidental writes to input tensors. The new interface looks
something like the following:

    ```mojo
    @compiler.register("add_one_custom")
    struct AddOneCustom:
        @staticmethod
        fn execute[
            target: StringLiteral,
        ](
            out: OutputTensor,
            x: InputTensor[type = out.type, rank = out.rank],
            ctx: DeviceContextPtr,
        ) raises:
            @parameter
            @always_inline
            fn elementwise_add_one[
                width: Int
            ](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
                return x.load[width](idx) + 1

            foreach[elementwise_add_one, target=target](out, ctx)
    ```

- The `foreach` function now `raises` to be able to handle errors within an
elementwise calculation.

### Hopper kernels {#25-2-hopper-kernels}

State-of-the-Art Kernels in Mojo for H100/H200 GPUs

- **Hopper Architecture Matrix Multiplication Kernels**: The implementation
achieved performance comparable to NVIDIA's highly optimized cuBLAS library.
These kernels take full advantage of the Tensor Cores in Hopper architecture GPUs
to accelerate the fundamental matrix multiplication operations that underpin deep
learning workloads.

- **Multi-GPU AllReduce Implementation**: The AllReduce operation is critical for
distributed inference across multiple GPUs, as it efficiently aggregates
gradients. The Mojo implementation surpassed NVIDIA's NCCL library in performance
benchmarks. This improvement reduces communication overhead during distributed
inference.

- **MAX Attention Kernel with Flash Attention 3:** This implementation
incorporates the latest Flash Attention 3 algorithm and extends it, which
significantly accelerates the computation of attention mechanisms in transformer
models. The MAX attention kernel optimizes memory access patterns and
computational steps, reducing both the memory footprint and execution time of
attention operations. This is particularly important for LLMs where attention
calculations represent a substantial portion of the computational workload.

### GPU programming {#25-2-gpu-programming}

- Added the Mojo `max.driver` API to enable dispatching
GPU functions from Mojo.

Check out [examples for GPU programming in
Mojo](https://github.com/modular/modular/tree/main/mojo/examples/gpu-functions),
which use this new API.

### Mojo {#25-2-mojo}

Mojo is a crucial component of the MAX stack that enables all of MAX's
performance-oriented code across hardware. For all the updates to the Mojo
language, standard library, and tools, see the [Mojo
changelog](/mojo/changelog).

### Documentation {#25-2-documentation}

New examples for writing custom ops:

- [`fused_attention`](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/fused_attention.mojo)
  demonstrates complex GPU programming using MAX abstractions for a
  practical use in AI model development.

- [`matrix_multiplication`](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/matrix_multiplication.mojo)
  includes a series of progressive optimizations for matrix multiplications
  on GPUs.

- [`histogram`](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/histogram.mojo)
shows how to implement the histogram pattern as a custom op.

- New [examples for GPU programming in
Mojo](https://github.com/modular/modular/tree/main/mojo/examples/gpu-functions)
using the new MAX Driver API

    These use a Mojo programming model that should look familiar to CUDA C
    programmers, showing how to define and dispatch GPU functions within a
    single Mojo file. These examples recreate the first three samples from
    the popular textbook ["Programming Massively Parallel
    Processors"](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311),
    showing how basic concepts translate from CUDA into Mojo. Additionally, a
    Mandelbrot set calculation example that parallels a similar one in the
    existing custom ops examples.

- New [MAX containers](https://docs.modular.com/max/container/) available. For
more information on the base and full MAX containers, see [Container
contents](https://docs.modular.com/max/container/#container-contents).

## v25.1.1 (2025-02-19)

Fix performance issues in autoregressive models with paged attention
by setting sensible default values for `--max-num-steps` that are
platform-specific.

## v25.1 (2025-02-13)

- [Highlights](#25-1-highlights)
- [Documentation](#25-1-docs)
- [MAX Serve](#25-1-serve)
- [MAX models](#25-1-max-models)
- [MAX Engine](#25-1-engine)
  - [Graph APIs](#25-1-graph)
  - [Pipeline APIs](#25-1-pipelines)
  - [GPU programming](#25-1-gpus)
- [Mojo](#25-1-mojo)

### ✨ Highlights {#25-1-highlights}

- **Custom ops for GPUs**

  Our new custom op API allows you to extend MAX Engine with new graph
  operations written in Mojo that execute on either CPU or GPU, providing full
  composability and extensibility for your models. See more in the section
  about [GPU programming](#25-1-gpus).

- **Enhanced support for agentic workflows**

  MAX Serve now supports function calling, which allows you to instruct your
  model to interact with other systems, such as retrieve data and execute
  external tasks. [Learn more about function calling and tool
  use](/max/serve/function-calling).

  MAX Serve now supports structured output (also known as constrained decoding)
  for MAX models on GPU. This allows you to enforce the output format from a
  model using an input schema that defines the output structure. [Learn more about
  structured output](/max/serve/structured-output).

- **Extended model architecture support**

  - MAX Serve now supports multimodal models that take both text and image
  inputs. For example, see [how to deploy Llama 3.2
  Vision](/max/tutorials/deploy-llama-vision).

  - MAX Serve now supports text embedding models. Learn how to [deploy a text
  embedding model](/max/tutorials/run-embeddings-with-max-serve).

- **New `max-pipelines` CLI tool**

  Instead of cloning our GitHub repo to access our latest GenAI models, you can
  instead install the `max-pipelines` CLI tool and quickly run an inference or
  deploy an endpoint.

### Documentation {#25-1-docs}

New tutorials:

- [Build custom ops for GPUs](/max/develop/build-custom-ops)

- [Serverless GPU inference on Google Cloud
Run](/max/tutorials/deploy-serverless-cloud-run)

- [Generate image descriptions with Llama 3.2
Vision](/max/tutorials/deploy-llama-vision)

- [Deploy a text embedding model](/max/tutorials/run-embeddings-with-max-serve)

Other docs:

- [Function calling and tool use](/max/serve/function-calling)

- [Structured output](/max/serve/structured-output)

- [Prefix caching with PagedAttention](/max/serve/prefix-caching)

- `max-pipelines` CLI

### MAX Serve {#25-1-serve}

- The `/v1/completions` REST endpoint now supports:

  - Pre-tokenized prompts.

  - Image inputs for multimodal models such as `Llama-3.2-11B-Vision-Instruct`.
  For an example, see [how to generate image
  descriptions with Llama 3.2 Vision](/max/tutorials/deploy-llama-vision).

    **Known issue:** You might receive faulty results because some parts of the
    text prompt get ignored for certain input combinations. We've identified
    the problem and will have a fix in a subsequent nightly
    release.

  - Function calling and tool use, which allows you to instruct your
  model to interact with other systems, such as retrieve data and execute
  external tasks. [Learn more about function calling and tool
  use](/max/serve/function-calling).

  - Structured output (also known as constrained decoding), which allows you to
  enforce the output format from a model using a JSON schema and the
  `response_format` field. To enable constrained decoding pass
  `--enable-structured-output` when running the server. However, this feature
  currently works for MAX models on GPU only (support for PyTorch models and
  CPU is in progress). [Learn more about structured
  output](/max/serve/structured-output).

- Added support for the `/v1/embeddings` API endpoint, allowing you to generate
vector representations using embedding models. See how to [deploy a text
embedding model](/max/tutorials/run-embeddings-with-max-serve).

- Max Serve can evict requests when the number of available pages in the
  PagedAttention KVCache is limited. Before, the KV manager would throw an OOM
  error when a batch that cannot fit in the cache was scheduled.

### MAX models {#25-1-max-models}

- Added the `max-pipelines` CLI tool that simplifies the
process to run inference with GenAI models (specified with a Hugging Face repo
ID) and deploy them to a local endpoint with MAX Serve.

  Previously, running or serving these models required cloning the
  [modular/max](https://github.com/modular/max) GitHub repo and then running
  commands such as `magic run llama3`.

  These model-specific commands like `llama3` and `replit` commands have been
  removed. They're now standardized and subsumed by flags like
  `--model-path` in the `max-pipelines` tool. Arguments such as
  `--max-length` and `--weight-path` are also still supported by
  `max-pipelines`.

  To view a list of supported model architectures from Hugging Face, run
  `max-pipelines list`.

- Added support for PagedAttention, which improves memory efficiency by
partitioning the KV cache into smaller blocks, reducing fragmentation and
enabling larger inference batches. You can enable it with
`--cache-strategy=paged` and `--kv-cache-page-size` with a value that's a
multiple of 128.

- Added support for prefix caching in all cases where PagedAttention is
supported. This allows for more efficient usage of KVCache and improved prefill
performance for workloads with common prefixes. You can enable it by setting
`--enable-prefix-caching`. For more information, see [Prefix caching with
PagedAttention](/max/serve/prefix-caching).

- Batch size and max length are now inferred from available memory and the HF
  Models' default values for max length, respectively. If a configuration leads
  to an OOM, then we provide recommendations (to the best of our ability) to the
  user to fit the model into memory.

- Added support for heterogeneous KV caches for multi-modal models, such as
Llama Vision, which cache different KV states for self and cross attention
layers.

- Added support for embedding models, starting with MPNet. For example:

  ```shell
  max-pipelines generate \
    --model-path=sentence-transformers/all-mpnet-base-v2 \
    --prompt="Encode this sentence."
  ```

  Also see [how to deploy a text
  embedding model](/max/tutorials/run-embeddings-with-max-serve).

- Added support for image and text multimodal models:

  - `max-pipelines generate` now accepts image input with `--image_url`.

  - Added an experimental Pixtral pipeline you can run as follows:

    ```shell
    max-pipelines generate \
      --model-path=mistral-community/pixtral-12b \
      --prompt="What is in this image? [IMG]" \
      --image_url=http://picsum.photos/1024/1024
    ```

    The pipeline is automatically used for all models implementing the
    `LlavaForConditionalGeneration` architecture.

    The implementation currently has a limit of one image. We plan support an
    arbitrary number of images of mixed sizes soon.

  - Added an experimental Llama Vision pipeline you can run as follows:

    ```shell
    max-pipelines generate \
      --model-path=meta-llama/Llama-3.2-11B-Vision-Instruct \
      --prompt="<|image|><|begin_of_text|>What is in this image?" \
      --image_url=http://picsum.photos/1024/1024
    ```

    The pipeline is automatically used for all models implementing the
    `MllamaForConditionalGeneration` architecture.

    Note: This model is gated and requires that you set the
    [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken)
    environment variable. See
    [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).

  - See [how to generate image
    descriptions with Llama 3.2 Vision](/max/tutorials/deploy-llama-vision).

- Added support for the `Qwen2ForCausalLM` model architecture (such as
`Qwen/Qwen2.5-7B-Instruct`). For example:

  ```shell
  max-pipelines generate \
    --model-path=Qwen/Qwen2.5-7B-Instruct \
    --prompt="Write bubble sort in python" \
    --quantization-encoding bfloat16
  ```

- Added support for offline batched inference for text-based LLMs, allowing you
to load a model and run inference with a batch of inputs directly from Python,
instead of relying on an HTTP interface. For an example, see
[`examples/offline-inference/basic.py`](https://github.com/modular/modular/blob/main/examples/offline-inference/basic.py).

- The `--max-cache-batch-size` flag has been deprecated in favor of
  `--max-batch-size`. Using `--max-cache-batch-size` now emits a deprecation
  warning and will stop working in a future release.

- The `--use-gpu` flag has been deprecated in favor of `--devices=cpu`,
`--devices=gpu`, or `--devices=gpu-0,gpu-1,...`. If the device isn't specified,
the model runs on the first available GPU, or CPU if no GPUs are available.

### MAX Engine {#25-1-engine}

- Improved internal kernel compilation speed 1.5 - 4X across different models.

  We've revamped our GPU compilation process so that all kernels in a program
  are compiled together into a single LLVM module, then split into separate
  kernels afterward. This ensures shared code between kernel entry points is
  only compiled once. For example, we observe a 3.7x speed up for Llama3.1-8b
  GPU startup time.

- Improved initial model execution speed on NVIDIA GPUs.

  Instead of compiling to PTX and performing just-in-time compilation during
  runtime, we now generate CUBIN binaries directly. While this increases
  initial compilation time, it significantly improves execution speed.

- The kernels have been further tuned for performance on NVIDIA A100 GPUs.

#### Graph APIs {#25-1-graph}

- You can now write custom operations (ops) in Mojo, and add them to a graph
  constructed in Python, using
  [`custom()`](/max/api/python/graph/ops#max.graph.ops.custom) and
  [`inplace_custom()`](/max/api/python/max/graph/ops#max.graph.ops.inplace_custom).

  For more detail, see the section below about [GPU programming](#25-1-gpus).

- Cached compiled MAX graphs that make use of custom operations now get
invalidated when the implementation of the custom operations change.

- [`Graph.add_weight()`](/max/api/python/graph/Graph#max.graph.Graph.add_weight)
now takes an explicit `device` argument. This enables explicitly passing
GPU-resident weights to
[`session.load()`](/max/api/python/engine#max.engine.InferenceSession.load) via
the weights registry to initialize the model.

- [`max.graph.Weight`](/max/api/python/graph/Weight) now inherits
from `TensorValue`, allowing you to call `weight.cast()` or `weight.T`. As such,
the [`TensorValue`](/max/api/python/graph/TensorValue#max.graph.TensorValue) no
longer accepts `Weight` for the `value` argument.

#### Pipeline APIs {#25-1-pipelines}

- [`TextTokenizer.new_context()`](/max/api/python/pipelines/tokenizer#max.pipelines.tokenizer.TextTokenizer.new_context)
now supports tool definitions passed through its `request` argument (via
`TokenGeneratorRequest.tools`).

  It also now supports JSON schemas passed through its `request` argument (via
  [`TokenGeneratorRequest.response_format`](/max/api/python/pipelines/interfaces/#max.pipelines.interfaces.TokenGeneratorRequest.response_format)).

- Removed the default `num_steps` value for
[`TokenGenerator.next_token()`](/max/api/python/pipelines/interfaces/#max.pipelines.interfaces.TokenGenerator.next_token),
ensuring users pass a value, reducing the potential for silent errors.

- [`KVCacheStrategy`](/max/api/python/pipelines/kv_cache/cache_params#max.pipelines.kv_cache.cache_params.KVCacheStrategy)
now defaults to `MODEL_DEFAULT`.

  As opposed to the previous setting which always used the "continuous" caching
  strategy, KV caching strategy is now defaulted on an architecture-specific
  basis to ensure the most optimized caching strategy is used.

- The
[`Linear`](/max/api/python/nn/Linear)
layer now has a `create()` class method that automatically creates
specializations of `Linear` for non-quantized, k-quant, or GPTQ layers.

- Added
[`nn.Conv1D`](/max/api/python/nn/conv#max.nn.conv.Conv1D)
for audio models like Whisper.

#### GPU programming {#25-1-gpus}

This release includes all new APIs to program on GPUs. The way to write code
for GPUs is to create custom operations with GPU functions that you can load
into a MAX graph. This foundational API includes a few key components:

- Mojo APIs to write custom op functions:

  - The [`@compiler.register`](/max/api/mojo-decorators/compiler-register)
  decorator is applied to a Mojo struct that implements a custom op in an
  `execute()` function—for either CPU or GPU—and a `shape()` function that
  defines the custom op's output tensor.

  - The [`max.tensor`](/mojo/kernels/extensibility/tensor/) package adds
  essential Mojo APIs for writing custom ops, such as:

    - The [`foreach()`](/mojo/kernels/extensibility/tensor/managed_tensor_slice/foreach)
    function, which efficiently executes an element-wise computation in parallel
    on either a GPU or CPU.

    - The
    [`ManagedTensorSlice`](/mojo/kernels/extensibility/tensor/managed_tensor_slice/ManagedTensorSlice)
    type defines the input and output tensors for the custom op.

- Python APIs to load custom ops into a model:

  - The [`custom()`](/max/api/python/graph/ops#max.graph.ops.custom) and
  `inplace_custom()`
  functions allow you to add the previously-defined Mojo custom op to a MAX
  graph written in Python.

  - The [`InferenceSession`](/max/api/python/engine#max.engine.InferenceSession)
  constructor accepts the custom op implementation as a [Mojo
  package](/mojo/manual/packages#mojo-packages) in the `custom_extensions`
  argument.

For more detail, see the [tutorial to build custom ops for
GPUs](/max/develop/build-custom-ops), or check out this [simple example of
a custom
op](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/add_custom.mojo).

Additionally, we've added a new [`gpu` package](/mojo/std/gpu/) to the Mojo
standard library that provides low-level programming constructs for working
with GPUs. These APIs let you do things that you can't currently do with the
high-level `foreach()` abstraction above. The Mojo `gpu` APIs allow you to
manually manage interaction between the CPU host and GPU device, manage memory
between devices, synchronize threads, and more. For some examples, see
[`vector_addition.mojo`](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/vector_addition.mojo)
and
[`top_k.mojo`](https://github.com/modular/modular/blob/main/examples/custom_ops/kernels/top_k.mojo).

### Mojo {#25-1-mojo}

Mojo is a crucial component of the MAX stack that enables all of MAX's
performance-oriented code across hardware. For all the updates to the Mojo
language, standard library, and tools, see the [Mojo
changelog](/mojo/changelog).

## v24.6 (2024-12-17)

This is a huge update that offers a first look at our serving library for
MAX on GPUs!

- [Highlights](#24-6-highlights)
- [Documentation](#24-6-docs)
- [MAX Serve](#24-6-serve)
- [MAX models](#24-6-models)
- [MAX Engine](#24-6-engine)
  - [Driver APIs](#24-6-driver-api)
  - [Graph compiler](#24-6-graph-compiler)
  - [Graph APIs](#24-6-graph-api)
  - [Custom op registration](#24-6-custom-ops)
  - [Numeric kernels](#24-6-kernels)
- [Mojo](#24-6-mojo)

Also check out our [blog post introducing MAX
24.6](https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform).

### ✨ Highlights {#24-6-highlights}

- **MAX Engine on GPUs preview**

  We're excited to share a preview of MAX Engine on GPUs. We've created a few
  tutorials that demonstrate MAX's ability to run GenAI models with our
  next-generation MAX graph compiler on NVIDIA GPU architectures (including
  A100, A10, L4, and L40 GPUs). You can experience it today by [deploying
  Llama 3 on an A100 GPU](/max/tutorials/max-serve-local-to-cloud).

- **MAX Serve preview**

  This release also includes an all-new serving interface called MAX
  Serve. It's a Python-based serving layer that supports both
  native MAX models when you want a high-performance deployment, and
  off-the-shelf PyTorch LLMs from Hugging Face when you want to explore and
  experiment—all with GPU support. It provides an OpenAI-compatible REST
  endpoint for inference requests, and a Prometheus-compatible metrics
  endpoint. You can use a `magic` command to start a local server , or use our
  ready-to-deploy MAX container to start an endpoint in the cloud. Try it now
  [with an LLM from Hugging Face](/max/tutorials/max-serve-local-to-cloud).

- **Upgraded MAX models**

  As we continue to build our Python-based MAX Graph API that allows you to
  build high-performance GenAI models, we've made a ton of performance
  improvements to the existing models and added a few new models to our GitHub
  repo. All the Python-based MAX models now support GPUs and broad model
  architectures. For example,
  [`llama3`](https://github.com/modular/modular/tree/main/max/pipelines/architectures/llama3)
  adds compatibility for the LlamaForCausalLM family, which includes over
  20,000 model variants and weights on Hugging Face.

### Documentation {#24-6-docs}

New tutorials:

- [Deploy Llama 3 on GPU with MAX
Serve](/max/tutorials/max-serve-local-to-cloud)

- [Deploy Llama 3.1 on GPU-powered Kubernetes
  clusters](/max/tutorials/deploy-max-serve-on-kubernetes)

- [Get started with MAX Graph in
  Python](/max/tutorials/get-started-with-max-graph-in-python)

Other new docs:

- [MAX container](/max/container)

- [Benchmark MAX
  Serve](https://github.com/modular/modular/tree/main/benchmark)

Also, our documentation is now available for **MAX nightly builds**! If you're
building with a nightly
release, you can
switch to see the nightly docs using a toggle to the right of the search bar.

### MAX Serve {#24-6-serve}

This release includes a preview of our Python-based serving library called MAX
Serve. It simplifies the process to deploy your own inference
server with consistent and reliable performance.

MAX Serve currently includes the following features:

- Deploys locally and to the cloud with our [MAX container
image](/max/container), or with the `magic` CLI.

- An OpenAI-compatible server with streaming `/chat/completion` and
`/completion` endpoints for LLM inference requests.

- Prometheus-compatible [metrics endpoint](/max/container#metrics) with LLM
KPIs (TTFT and ITL) for monitoring and evaluating performance.

- Supports most `TextGeneration` Hugging Face Hub models.

- Multiprocess HTTP/model worker architecture to maximize CPU core utilization
by distributing multiple incoming requests across multiple processes, ensuring
both high throughput and responsiveness.

- Continuous heterogeneous batching to combine multiple incoming requests into
a single inference (no waiting to fill a batch size) and improve total
throughput.

There's much more still in the works for MAX Serve, but you can try it today
with our tutorials to [Deploy Llama 3 on GPU with MAX
Serve](/max/tutorials/max-serve-local-to-cloud).

**Known issues:**

- While this release is enough to support typical chatbot applications,
  this release does not yet support the function-calling portion of the
  OpenAI API specification needed to enable robust agentic workflows.

- Sampling is still limited and doesn't currently respect temperature or
  other sampling-related API request input.

- Structured generation is not supported.

- Support for multi-modal models is still nascent.

### MAX models {#24-6-models}

All of our Python-based GenAI [models on
GitHub](https://github.com/modular/modular/tree/main/max/pipelines/architectures)
now support GPUs!

As we add more models, we're also building a robust set of libraries and
infrastructure that make it easier to build and deploy a growing library of
LLMs. Some of which is available in a new
[`max.pipelines`](/max/api/python/pipelines/) package and some of it is
alongside the [models on
GitHub](https://github.com/modular/modular/tree/main/max/pipelines/architectures).
Here are just some of the highlights:

- Deep integration with the Hugging Face ecosystem for a quick-to-deploy
experience, such as using HF Model Hub tools to fetch config files, support for
weights in [safetensor](https://github.com/huggingface/safetensors) format,
support for HF tokenizers, and more. (We also support GGUF weight formats.)

- Expanded set of model abstractions for use by different LLM architectures:

  - Attention layers (including highly optimized implementations with
  configurable masking, like
  [`AttentionWithRope`](https://github.com/modular/modular/tree/main/max/nn/attention/attention_with_rope.py)).
  The optimized attention layers include variants that accept an attention
  mask. More memory-efficient variants that don't take a mask instead take a
  "mask functor" argument to the kernel, which implements masking without
  materializing a mask by computing a mask value from input coordinates on the
  fly.

  - Transformers such as [`Transformer` and
  `TransformerBlock`](https://github.com/modular/modular/tree/main/max/nn/transformer/transformer.py).
  These include an initial implementation of ragged tensors—tensors for which
  each dimension can have a different size, avoiding the use of padding tokens
  by flattening a batch of sequences of differing lengths.

  - Common layers such as
  [`RMSNorm`](https://github.com/modular/modular/tree/main/max/nn/norm/rms_norm.py)
  ,
  [`Embedding`](https://github.com/modular/modular/tree/main/max/nn/embedding.py),
  and
  [`Sequential`](https://github.com/modular/modular/tree/main/max/nn/sequential.py).

  - KV cache management helpers, like
  [`ContinuousBatchingKVCacheManager`](/max/api/python/pipelines/kv_cache/continuous_batching_cache#max.pipelines.kv_cache.continuous_batching_cache.ContinuousBatchingKVCacheManager).

  - Low-level wrappers over optimized kernels like
  [`fused_qk_ragged_rope`](https://github.com/modular/modular/tree/main/max/nn/kernels.py).
  These are custom fused kernels that update the KV cache in place. Although
  they are custom, they reuse the underlying kernel implementation by passing
  in lambda functions used to retrieve inputs and write to outputs in place.

- Added generalized interfaces for text generation such as
[`TokenGenerator`](/max/api/python/pipelines/interfaces#max.pipelines.interfaces.TokenGenerator)
and
[`PipelineModel`](/max/api/python/pipelines/pipeline#max.pipelines.pipeline.PipelineModel),
which provide modularity within the models and serving infrastructure. Also
added a plug-in mechanism
([`PipelineRegistry`](/max/api/python/pipelines/registry#max.pipelines.registry.PipelineRegistry))
to more quickly define new models, tokenizers, and other reusable components.
For example, anything that conforms to
[`TokenGenerator`](/max/api/python/pipelines/interfaces#max.pipelines.interfaces.TokenGenerator)
can be served using the LLM infrastructure within MAX Serve. We then used this
interface to create the following:

  - An optimized
  [`TextGenerationPipeline`](/max/api/python/pipelines/pipeline#max.pipelines.pipeline.TextGenerationPipeline)
  that can be combined with any compatible graph and has powerful performance
  features like graph-based multi-step scheduling, sampling, KV cache
  management, ragged tensor support, and more.

  - A generic
  [`HFTextGenerationPipeline`](/max/api/python/pipelines/hf_pipeline#max.pipelines.hf_pipeline.HFTextGenerationPipeline)
  that can run any Hugging Face model for which we don't yet have an optimized
  implementation in eager mode.

- Models now accept weights via a weights registry, which is passed to the
[`session.load()`](/max/api/python/engine#max.engine.InferenceSession.load)
method's `weights_registry` argument. The decoupling of weights and model
architecture allows implementing all of the different fine-tunes for a given
model with the same graph. Furthermore, because the underlying design is
decoupled, we can later expose the ability to compile a model once and swap
weights out on the fly, without re-compiling the model.

- Added generic implementations of common kernels, which allow you to plug-in
different batching strategies (ragged or padded), KV cache management
approaches (continuous batching), masking (causal, sliding window, etc.), and
position encoding (RoPE or ALIBI) without having to re-write any kernel code.
(More about this in a future release.)

- Multi-step scheduling to run multiple token-generation steps on GPU before
synchronizing to the CPU.

**Updated models:**

- Significant performance upgrades for [Llama
3](https://github.com/modular/modular/tree/main/max/pipelines/architectures/llama3),
and expanded compatibility with the `LlamaForCausalLM` models family. For
example, it also supports Llama 3.2 1B and 3B text models.

**New models:**

- [Mistral
NeMo](https://github.com/modular/modular/tree/main/max/pipelines/architectures/mistral)
(and other `MistralForCausalLM` models)

- [Replit Code V1.5
3B](https://github.com/modular/modular/tree/main/max/pipelines/architectures/replit)

**Known issues:**

- The Q4 quantized models currently work on CPU only.

- Using a large setting for `top-k` with the Llama 3.1 model may lead to
segmentation faults for certain workloads when run on NVIDIA GPUs. This should
be resolved in the latest nightly MAX builds.

- The models currently use a smaller default context window than the
`max_seq_len` specified in the Hugging Face configuration files for a given
model. This can be manually adjusted by setting the `--max-length` parameter to
the desired context length when serving a model.

- Some variants of the supported core models (like `LlamaForCausalLM` with
different number of heads, head sizes, etc.) might not be fully optimized yet.
We plan to fully generalize our implementations in a future release.

### MAX Engine {#24-6-engine}

MAX Engine includes a lot of the
core infrastructure that enables MAX to accelerate AI models on any hardware,
such as the graph compiler, runtime, kernels, and the APIs to interact with it
all, and it all works without external dependencies such as PyTorch or CUDA.

This release includes a bunch of performance upgrades to our graph compiler and
runtime. We've added support for NVIDIA GPU architectures (including A100, A10,
L4, and L40 GPUs), and built out new infrastructure so we can quickly add
support for other GPU hardware.

**Engine API changes:**

- [`InferenceSession`](/max/api/python/engine#max.engine.InferenceSession)
now accepts a `custom_extensions` constructor argument, same as `load()`, to
specify model extension libraries.

- The [`Model`](/max/api/python/engine#max.engine.Model) object is now callable
to run an inference.

**Breaking changes**:

- `Model.execute()` signature changed to support GPUs.

  - The [`execute()`](/max/api/python/engine#max.engine.Model.execute) function
  currently doesn't accept keyword arguments. Instead you can pass tensors as a
  [`driver.Tensor`](/max/api/python/driver#max.driver.Tensor), `int`, `float`,
  `bool`,
  [`np.generic`](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.generic),
  or [`DLPackArray`](/max/api/python/driver#max.driver.DLPackArray)
  ([DLPack](https://github.com/dmlc/dlpack)). Note that both PyTorch and NumPy
  arrays implement the DLPack protocol, which means you can also pass either of
  those types to `execute()`.

  - [`execute_legacy()`](/max/api/python/engine#max.engine.Model.execute_legacy)
  preserves the semantics of `execute()` with support for keyword arguments to
  help with migration, but will be removed in a future release.
  `execute_legacy()` doesn't support GPUs.

  - Calling `execute()` with positional arguments still works the same.

#### Driver APIs {#24-6-driver-api}

MAX Driver (the [`max.driver`](/max/api/python/driver) module) is a new
component of MAX Engine that's still a work in progress. It provides primitives
for working with heterogeneous hardware systems (GPUs and CPUs), such as to
allocate on-device memory, transfer data between host and device, query device
stats, and more. It's a foundation on which other components of MAX Engine
operate (for example, `InferenceEngine` now uses
[`driver.Tensor`](/max/api/python/driver#max.driver.Tensor) to handle model
inputs and outputs).

**Driver API changes:**

- Added `CUDA()` device to open an NVIDIA GPU.

- Added support for fp16 and bfloat16 dtypes.

- Expanded functionality for `max.driver.Device`, with new class methods and
properties. We are still working on building this out to support more
accelerator features.

- [`driver.Tensor`](/max/api/python/driver#max.driver.Tensor) (and the
`InferenceSession.load()` argument `weights_registry` ) now supports zero-copy
interoperability with NumPy arrays and PyTorch tensors, using
[DLPack](https://github.com/dmlc/dlpack) /
[`DLPackArray`](/max/api/python/driver#max.driver.DLPackArray).

- [`driver.Tensor`](/max/api/python/driver#max.driver.Tensor) has new methods,
such as `from_dlpack()`, `element_size()` , `to()`, `to_numpy()`, `view()`,
`zeros()`, and more.

MAX Driver APIs are still changing rapidly and not yet ready for general use.
We'll publish more documentation in a future release.

**Known issues:**

- MAX Driver is currently limited to managing just one NVIDIA GPU at a time (it
does not yet support multi-GPU). It also does not yet support remote devices.

- DLPack support is not complete. For example, streams are not yet supported.

#### Graph compiler {#24-6-graph-compiler}

When you load a model into MAX Engine, the graph compiler is the component that
inspects and optimizes all graph operations (ops) to deliver the best run time
performance on each device.

This release includes various graph compiler improvements:

- Major extensions to support NVIDIA GPUs (and other devices in the future),
including async copies and caching of JIT'd kernels.

- The runtime now performs scheduling to enable GPU compute overlap with the
CPU.

- New transformations to the Mojo kernels to enable a number of optimizations,
including specialization on tensor dimensions, specialization on target
hardware, specialization on non-tensor dimension input to kernels, automatic
kernel fusion between operators, and more.

- New algebraic simplifications and algorithms for ops such as horizontal
fusion of matrix multiplications.

- New CPU-side primitives for device management that are automatically
transformed and optimized to reduce overhead (MAX does not need to use things
like CUDA Graphs).

- Updated memory planning to preallocate device memory (hoist computation from
inference runtime to initialization time) and reduce per-inference overhead.

#### Graph APIs {#24-6-graph-api}

The graph compiler is also exposed through the MAX Graph APIs (the
[`max.graph`](/max/api/python/graph/) package), which allow you to build
high-performance GenAI models in Python.

**Graph API changes:**

- Python stack traces from model execution failures now include a trace to the
original op-creation, allowing for easier debugging during development.

- The [`max.graph`](/max/api/python/graph/) APIs now include preliminary
support for symbolic algebraic expressions using
[`AlgebraicDim`](/max/api/python/graph/type#max.graph.type.AlgebraicDim),
enabling more powerful support for checked dynamic shapes. This allows
`-Dim("x") - 4`. Furthermore, the algebraic expressions simplify to a canonical
form, so that for example `-Dim("x") - 4 == -(Dim("x") + 4)` holds.

- More advanced dtype promotion now allows
[`TensorValue`](/max/api/python/graph/TensorValue) math operators to just work
when used with NumPy arrays and python primitives.

- [`TensorValue`](/max/api/python/graph/TensorValue) has new methods, such as
`broadcast_to()`, `cast()`, `flatten()`, `permute()`, and more.

- Added [`BufferValue`](/max/api/python/graph/BufferValue), which allows for
device-resident tensors that are read and mutated within the graph.

- [`DType`](/max/api/python/dtype#max.dtype.DType) has new methods/properties,
  `align`, `size_in_bytes`, and `is_float()`.

- [`Value`](/max/api/python/graph/Value) constructor accepts more types for
`value`.

- [`TensorValue`](/max/api/python/graph/TensorValue) constructor accepts more
types for `value`.

- [`TensorValue.rebind()`](/max/api/python/graph/TensorValue#max.graph.TensorValue.rebind)
accepts a new `message` argument.

**Breaking changes:**

- [`Graph.add_weight()`](/max/api/python/graph/Graph#max.graph.Graph.add_weight)
now accepts [`Weight`](/max/api/python/graph/Weight#max.graph.Weight) and
returns [`TensorValue`](/max/api/python/graph/TensorValue).
[`Weight`](/max/api/python/graph/Weight#max.graph.Weight) is essentially a
named placeholder for a tensor that knows its name, dtype, shape, and
optionally device and quantization encoding. `Graph.add_weight()` stages an op
in the graph that is populated by a named weight in the weights registry passed
to `session.load`.

- The [`Weight`](/max/api/python/graph/Weight#max.graph.Weight) constructor
arguments changed; added `align` , `dtype` , and `shape`; removed `assign` ,
`filepath`, `offset`, and `value`.

- The `ops.scalar()` method was removed along with the `is_static()` and
`is_symbolic()` methods from all `graph.type` objects.

  - Instead of `ops.scalar()`, use
  [`ops.constant()`](/max/api/python/graph/ops#max.graph.ops.constant).

  - Instead of `is_static()` and `is_symbolic()`, use
    `isinstance(dim, SymbolicDim)` and `isinstance(dim, StaticDim)`.

The MAX Graph APIs are not ready for general use but you can [experiment with
it now by following this
tutorial](/max/tutorials/get-started-with-max-graph-in-python). We'll add more
documentation when we finish some API redesigns.

#### Custom op registration {#24-6-custom-ops}

Although the APIs to write custom operators (ops) isn't ready for general use,
this release includes a significant redesign that lays the groundwork. You
might notice some associated APIs in this release and more APIs in the
nightlies, so here's a little about the work in progress:

- The custom op APIs will allow you to extend MAX Engine with new ops written
in Mojo, providing full composability and extensibility for your models. It's
the exact same API we use to write MAX Engine's built-in ops such as `matmul`.
That means your custom ops can benefit from all our compiler optimization
features such as kernel fusion—your ops are treated the same as all the ops
included "in the box."

- The new API requires far less adornment at the definition site to enable the
MAX model compiler to optimize custom ops along with the rest of the graph
(compared to our previous version that used `NDBuffer`).

- Custom ops support "destination passing style" for tensors.

- The design composes on top of Mojo's powerful meta programming, as well as
the kernel libraries abstractions for composable kernels.

We'll publish more documentation when the custom op API is ready for general
use. Check out the MAX repo's `nightly` branch to see the latest [custom op
examples](https://github.com/modular/modular/tree/main/max/examples/custom_ops).

**Known issues:**

- Custom ops don't have type or lifetime checking. They also don't reason about
mutability. Expect lots of sharp corners and segfaults if you hold them wrong
while we improve this!

#### Numeric kernels {#24-6-kernels}

The GPU kernels for MAX Engine are built from the ground up in Mojo with no
dependencies on external vendor code or libraries. This release includes the
following kernel improvements:

- AttenGen: a novel way to express attention pattern that's able to express
different attention masks, score functions, as well as caching strategies.

- State-of-the-art matrix multiplication algorithms with optimizations such as
the following:

  - Pipelining and double-buffering to overlap data transfer and computation
  and to hide memory access latency (for both global and shared memory).

  - Thread swizzling to avoid shared memory bank conflicts associated with
  tensor core layouts.

  - Block swizzling to increase L2 cache locality.

- SplitK/StreamK GEMM algorithms: divides the computation along the shared K
dimension into smaller matrices which can then be executed independently on
streaming multiprocessors (such as CUDA cores). These algorithms are ideal for
matrices with large K dimension but small M dimension.

- Large context length MHA: uses SplitK/StreamK to implement the attention
mechanism and eliminate the need of a huge score matrix, which drastically
reduces memory usage/traffic to enable large context length.

- DualGemm: accelerates the multi-layer perceptron (MLP) layers where the
left-hand side (LHS) is shared between two matrix multiplications.

**Known issues:**

- The MAX kernels are optimized for bfloat16 on GPUs.

- Convolution on GPU is not performance optimized yet.

- Although v24.6 technically runs on H100, it doesn't include
performance-optimized kernels for that device yet and it isn't recommended.

### Mojo {#24-6-mojo}

Mojo is a crucial component of the MAX stack that enables all of MAX's
performance-oriented code across hardware. For all the updates to the Mojo
language, standard library, and tools, see the [Mojo
changelog](/mojo/changelog#v246-2024-12-17).

## v24.5 (2024-09-13)

### ✨ Highlights

- Mojo and MAX are magical! We've created a new package and virtual environment
  manager, `magic`, for MAX and Mojo.

- New [Llama3.1
  pipeline](https://github.com/modular/modular/tree/main/max/pipelines/architectures)
  built with the new MAX Graph Python API.

- We have not one, but two new Python APIs that we're introducing in this
  release:
  - [MAX Graph Python API](#max-graph-python-api)
  - [MAX Driver Python API](#max-driver-python-api)

### ⭐️ New

- Added `repeat_interleave` graph op.

- Added caching for MAX graph models.
  This means that graph compilation is cached and the executable model is
  retrieved from cache on the 2nd and subsequent runs.
  Note that the model cache is architecture specific and isn't portable across
  different targets.

- Support for Python 3.12.

#### MAX Graph Python API

This Python API
will ultimately provide the same low-level programming interface for
high-performance inference graphs as the Mojo API. As with the Mojo API, it's an
API for graph-building only, and it does not implement support for training.

You can take a look at how the API works in the
[MAX Graph Python API reference](/max/api/python/graph/).

#### MAX Driver Python API

The MAX Driver API allows you to interact with devices (such as CPUs and GPUs)
and allocate memory directly onto them. With this API, you interact with
this memory as tensors.

Note that this API is still under development, with support for non-host
devices, such as GPUs, planned for a future release.

To learn more, check out the
[MAX Driver Python APIreference](/max/api/python/driver).

#### MAX C API

New APIs for adding torch metadata libraries:

- `M_setTorchMetadataLibraryPath`
- `M_setTorchMetadataLibraryPtr`

### 🦋 Changed

#### MAX Engine performance

- Compared to v24.4, MAX Engine v24.5 generates tokens for Llama an average of
  15%-48% faster.

#### MAX C API

Simplified the API for adding torch library paths, which now only takes one path
per API call, but can be called multiple times to add paths to the config:

- `M_setTorchLibraries` -> `M_setTorchLibraryPath`

### ⚠️ Deprecated

- The `max` command line tool is no longer supported and will be removed
  in a future release.

### ❌ Removed

- Dropped support for Ubuntu 20.04. If you're using Ubuntu, we currently
  support Ubuntu 22.04 LTS only.
- Dropped support for Python 3.8.
- Removed built-in PyTorch libraries from the max package. See the
  [FAQ](/max/faq) for information on supported torch versions.

## v24.4 (2024-06-07)

### 🔥 Legendary

- MAX is now available on macOS! [Try it now](/max).

- New quantization APIs for MAX Graph. You can now build high-performance
graphs in Mojo that use the latest quantization techniques, enabling even
faster performance and more system compatibility for large models.

  Learn more in the guide to [quantize your graph weights](/max/graph/quantize).

### ⭐️ New

#### MAX Mojo APIs

- Added AI pipeline examples in the `max` repo, with Mojo implementations for
  common transformer layers, including quantization support.

  - New Llama3 pipeline built with MAX Graph.

  - New Replit Code pipeline built with MAX Graph.

  - New TinyStories pipeline (based on TinyLlama) that offers a simple demo of
  the MAX Graph quantization API.

- Added `max.graph.checkpoint` package
to save and load model weights.

  All weights are stored in a
  `TensorDict`.
  You can save and load a `TensorDict` to disk with
  `save()` and
  `load()` functions.

- Added MAX Graph quantization APIs:

  - Added quantization encodings
  `BFloat16Encoding`,
  `Q4_0Encoding`,
  `Q4_KEncoding`,
  and
  `Q6_KEncoding`.
  - Added the
  `QuantizationEncoding`
  trait so you can build custom quantization encodings.
  - Added `Graph.quantize()`
    to create a quantized tensor node.
  - Added `qmatmul()` to
  perform matrix-multiplication with a float32 and a quantized matrix.

- Added some MAX Graph ops:

  - `avg_pool()`
  - `max_pool()`
  - `conv2d()`
  - `conv3d()`
  - `layer_norm()`
  - `tile()`
  - `select()`

- Added a `layer()` context
manager and
`current_layer()`
function to aid in debugging during graph construction. For example:

  ```mojo
  with graph.layer("foo"):
      with graph.layer("bar"):
          print(graph.current_layer())  # prints "foo.bar"
          x = graph.constant[DType.int64](1)
          graph.output(x)
  ```

  This adds a path `foo.bar` to the added nodes, which will
  be reported during errors.

- Added
  `format_system_stack()`
  function to format the stack trace, which we use to print better error
  messages from `error()`.

- Added
`TensorMap.keys()` to
get all the tensor key names.

#### MAX C API

Miscellaneous new APIs:

- `M_cloneCompileConfig()`
- `M_copyAsyncTensorMap()`
- `M_tensorMapKeys()` and `M_deleteTensorMapKeys()`
- `M_setTorchLibraries()`

### 🦋 Changed

#### MAX Mojo API

- `EngineNumpyView.data()`
and `EngineTensorView.data()`
functions that return a type-erased pointer were renamed to `unsafe_ptr()`.

- `TensorMap` now conforms
to `CollectionElement` trait to be copyable and movable.

- `custom_nv()` was removed, and its functionality moved into
  `custom()` as a function
  overload, so it can now output a list of tensor symbols.

## v24.3 (2024-05-02)

### 🔥 Legendary

- You can now write custom ops for your models with Mojo!

  Learn more about [MAX extensibility](/max/develop/custom-ops).

### 🦋 Changed

- Added support for named dynamic dimensions. This means you can specify when two
or more dimensions in your model's input are dynamic but their sizes at run
time must match each other. By specifying each of these dimension sizes with a
name (instead of using `None` to indicate a dynamic size), the MAX Engine
compiler can perform additional optimizations. See the notes below for the
corresponding API changes that support named dimensions.

- Simplified all the APIs to load input specs for models, making them more
consistent.

#### MAX Engine performance

- Compared to v24.2, MAX Engine v24.3 shows an average speedup of 10% on PyTorch
  models, and an average 20% speedup on dynamically quantized ONNX transformers.

#### MAX Graph API

The `max.graph` APIs are still changing
rapidly, but starting to stabilize.

- `AnyMoType` renamed to `Type`,
  `MOTensor` renamed to
  `TensorType`, and `MOList`
  renamed to `ListType`.

- Removed `ElementType` in favor of using `DType`.

- Removed `TypeTuple` in favor of using `List[Type]`.

- Removed the `Module` type so you can now start building a graph by directly
  instantiating a `Graph`.

- Some new ops in `max.ops`, including
  support for custom ops.

  See how to [create a custom op in MAX
  Graph](/max/develop/build-custom-ops).

#### MAX Engine Python API

- Redesigned
[`InferenceSession.load()`](/max/api/python/engine#max.engine.InferenceSession.load)
to replace the confusing `options` argument with a `custom_ops_path` argument.

  As a result, `CommonLoadOptions`, `TorchLoadOptions`, and
  `TensorFlowLoadOptions` have all been removed.

- [`TorchInputSpec`](/max/api/python/engine#max.engine.TorchInputSpec)
  now supports named dynamic dimensions (previously, dynamic dimension sizes
  could be specified only as `None`). This lets you tell MAX which dynamic
  dimensions are required to have the same size, which helps MAX better optimize
  your model.

#### MAX Engine Mojo API

- `InferenceSession.load_model()` was renamed to
`load()`.

- Redesigned
`InferenceSession.load()`
to replace the confusing `config` argument with a `custom_ops_path` argument
for use when [loading a custom op](/max/develop/build-custom-ops), and an
`input_specs` argument for use when loading TorchScript models.

  Doing so removed `LoadOptions` and introduced the new
  `InputSpec` type to define
  the input shape/type of a model (instead of `LoadOptions`).

- New `ShapeElement`
  type to allow for named dynamic dimensions (in `InputSpec`).

- `max.engine.engine` module was renamed to
`max.engine.info`.

#### MAX Engine C API

- [`M_newTorchInputSpec()`](/max/api/c/pytorch/config#m_newtorchinputspec)
  now supports named dynamic dimensions (via new `dimNames` argument).

### ❌ Removed

- Removed TensorFlow support in the MAX SDK, so you can no longer load a
TensorFlow SavedModel for inference. However, TensorFlow is still available for
enterprise customers.

  We removed TensorFlow because industry-wide TensorFlow usage has declined
  significantly, especially for the latest AI innovations. Removing TensorFlow
  also cuts our package size by over 50% and accelerates the development of
  other customer-requested features. If you have a production use-case for a
  TensorFlow model, please [contact
  us](https://www.modular.com/request-demo).

- Removed the Python `CommonLoadOptions`, `TorchLoadOptions`, and
  `TensorFlowLoadOptions` classes. See note above about
  `InferenceSession.load()` changes.

- Removed the Mojo `LoadOptions` type. See the note above about
  `InferenceSession.load()` changes.

## v24.2.1 (2024-04-11)

- You can now import more MAX Graph functions from `max.graph.ops` instead of
  using `max.graph.ops.elementwise`. For example:

  ```mojo
  from max.graph import ops

  var relu = ops.relu(matmul)
  ```

## v24.2 (2024-03-28)

- MAX Engine now supports TorchScript models with dynamic input shapes.

  No matter what the input shapes are, you still need to [specify the input
  specs](/max/model-formats#specify-torchscript-input-specs) for all
  TorchScript models.

- The Mojo standard library is now open source!

  Read more about it in [this blog
  post](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source).

- And, of course, lots of Mojo updates, including implicit traits, support for
  keyword arguments in Python calls, a new `List` type (previously
  `DynamicVector`), some refactoring that might break your code, and much more.

  For details, see the [Mojo changelog](/mojo/changelog#v242-2024-03-28).

## v24.1.1 (2024-03-18)

This is a minor release that improves error reports.

## v24.1 (2024-02-29)

The first release of the MAX platform is here! 🚀

This is a **preview version** of the MAX platform. That means it
is not ready for production deployment and designed only for local development
and evaluation.

Because this is a preview, some API libraries are still in development and
subject to change, and some features that we previously announced are not quite
ready yet. But there is a lot that you can do in this release!

This release includes our flagship developer tools, currently for **Linux
only**:

- **MAX Engine**: Our state-of-the-art graph compiler and runtime library that
executes models from PyTorch and ONNX, with incredible inference
speed on a wide range of hardware.

  - API libraries in Python, C, and Mojo to run inference with your existing
    models. [See the API references](/max/api).

  - The `max benchmark` tool, which runs MLPerf
    benchmarks on any compatible model without writing any code.

  - The `max visualize` tool, which allows you to visualize
    your model in Netron after partially lowering in MAX Engine.

  - An early look at the [MAX Graph API](/max/model-formats#max-graph), our
    low-level library for building high-performance inference graphs.

- **MAX Serving**: A preview of our serving wrapper for MAX Engine that
provides full interoperability with existing AI serving systems (such as
Triton) and that seamlessly deploys within existing container infrastructure
(such as Kubernetes).

  - A Docker image that runs MAX Engine as a backend for NVIDIA Triton
    Inference Server.

- **Mojo**: The world's first programming language built from the ground-up for AI
developers, with cutting-edge compiler technology that delivers unparalleled
performance and programmability for any hardware.

  - The latest version of Mojo, the standard library, and the `mojo` command
    line tool. These are always included in MAX, so you don't need to download
    any separate packages.

  - The Mojo changes in each release are often quite long, so we're going to
    continue sharing those in the existing [Mojo changelog](/mojo/changelog).

Additionally, we've started a new [GitHub repo for
MAX](https://github.com/modular/max), where we currently share a bunch of
code examples for our API libraries, including some large model pipelines.
You can also use this repo to [report issues with
MAX](https://github.com/modular/modular/issues/new/choose).

### Model Architecture Support

- Added support for the following model architectures:
  - `OlmoForCausalLM` (such as `allenai/OLMo-1B-0724-hf`)
  - `GraniteForCausalLM` (such as `ibm-granite/granite-3.1-8b-instruct`)
  - `Phi3ForCausalLM` (for Microsoft Phi-3 models)
  - `Qwen2ForCausalLM` (such as Qwen2 models)

  Example usage:

  ```sh
  max-pipelines generate \
    --model-path allenai/OLMo-1B-0724-hf \
    --prompt "Write bubble sort in mojo"
  ```

- The `max.pipelines.dataprocessing.tokenizer` and
`max.pipelines.dataprocessing.gguf_utils` modules have been removed.

- The previously deprecated `PipelineConfig.architecture` field and its
corresponding `--architecture` CLI argument have been removed.

### `max-pipelines` CLI

- The `--devices` CLI argument now supports a comma-separated list of GPU IDs
prefixed with `gpu:` like `--devices=gpu:0,1,2,3`. We no longer support the
previous `--devices=gpu-<N>` format.

  ```sh
  max-pipelines generate --model-path=meta-llama/Llama-3.3-70B-Instruct \
    --quantization-encoding bfloat16 \
    --devices gpu:0,1,2,3 \
    --prompt="Design a self-sustaining colony on Neptune's moon Triton with a myth/science fusion name, three quantum tech breakthroughs, one ethical debate, a neon-lit cultural ritual, and a hidden flaw—presented in bullet points."
  ```

- Removed `--huggingface-repo-id` PipelineConfig option and CLI argument in favor
of `--model-path`.

- Consolidated `-model-path` and `-weight-path`. If valid `-weight-path`(s) are
provided, they'll now override `--model-path`, which in turn handles both local
and remote (Hugging Face) cases. If we cannot derive the weights from the
`--weight-path`(s), we'll now fall back to the `--model-path`, which has to be set
explicitly by the user.

- Added `--huggingface-revision` option, to allow selecting a non-default branch
or a specific commit in a Hugging Face model repository.
