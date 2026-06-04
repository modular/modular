---
title: Nightly (v26.4)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added support for the Tencent Hunyuan Hy3-preview (`HYV3ForCausalLM`)
  architecture: a decoder-only mixture-of-experts model (192 routed experts,
  top-8 plus one shared expert) with sigmoid plus correction-bias routing,
  per-head query/key RMSNorm, and split-half RoPE. Runs multi-GPU with
  tensor-parallel attention and expert-parallel MoE.
- Added NVFP4 quantization support for Gemma 4.
- Added MXFP4 quantization support for MiniMax-M2.
- Added tensor-parallel attention + expert-parallel MoE (TP+EP) support for
  MiniMax-M2. Set `data_parallel_degree: 1` with `runtime.ep_size > 1` to
  shard attention heads across GPUs while distributing MoE experts via
  expert parallelism. Both reduce-scatter (default) and allreduce
  (`runtime.ep_use_allreduce: true`) collective strategies are supported.
- Kimi K2.5 tool calling now supports interleaved thinking: a single
  assistant turn may interleave multiple `<think>...</think>` reasoning
  blocks with multiple tool-call sections and end with `<|im_end|>`. The
  constrained-decoding grammar (used for `tool_choice` and JSON
  `response_format`) admits up to eight tool-call sections with an optional
  reasoning block before each, and lets the model stop before the cap. This
  fixes a `tool_choice=auto` failure where a second tool-call section
  disabled grammar enforcement for the rest of the request.

## MAX framework

### Inference server

- MAX Serve now returns a clearer 400 Bad Request with the underlying
  message when a prompt is too long for the model, instead of a generic
  "Value error." response (or, for streaming completions, a 500 Internal
  Server Error). All architectures now raise a structured
  `PromptTooLongError` exposing `num_tokens` and `max_length` attributes
  so callers can handle the failure programmatically. The user-facing
  message identifies the relevant limit (LLM context window vs. diffusion
  text encoder sequence length): for example, "Prompt is too long: N
  tokens exceeds the configured maximum context length of M tokens.
  Please shorten your prompt."
- Fixed a KV cache offloading correctness bug that corrupted output for
  multi-cache models (such as Gemma 4's interleaved sliding-window plus
  global attention) when the `local` or `tiered` KV connector was enabled.
  These models share one block pool across all of their caches, but the
  connector only offloaded and reloaded the primary cache, so a prefix-cache
  block served from host or disk restored only the primary cache's data and
  left the other caches' halves stale, degrading accuracy. The connector now
  offloads and restores every cache.

- Fixed JSON `response_format` and tool-call grammars not being enforced for
  Kimi K2.5 vision-language checkpoints. The Kimi K2.5 tokenizer did not carry
  grammar enforcement state onto the request context, so constrained-decoding
  requests fell back to an unenforced state and decoded freely (e.g. a
  `response_format=json_schema` request returned prose instead of
  schema-conformant JSON). The tokenizer now derives enforcement state from the
  response format, matching the text tokenizers.

- MAX Serve now accepts `role: "developer"` on `/v1/chat/completions`,
  normalizing it to `system` at the OpenAI-compat route layer. The OpenAI
  o1/o3 chat-completion spec uses `developer` in place of `system`, and
  recent OpenAI SDKs emit it by default. The previous behavior rejected
  the request with a 422 (`literal_error` on the message role).

- Fixed `CreateChatCompletionRequest` rejecting explicit `null` values for
  optional fields such as `tool_choice`, `tools`, and `response_format`.
  OpenAI-compatible clients (LangChain, JS SDKs, anything that serializes
  a dataclass with a `None` field) that emit `"tool_choice": null` instead
  of omitting the key are now accepted, matching the behavior of other
  OpenAI-compatible inference servers.

- Added two opt-in server flags for accepting OpenAI-compatible requests
  that the strict default behavior would reject:

  - `--allow-unsupported-logprobs`: when a request asks for `logprobs`
    against a runtime that cannot honor them (today, the overlap
    scheduler), MAX Serve logs a warning and serves the request without
    logprobs instead of returning a `400`.

  - `--allow-extra-request-fields`: unknown top-level fields on
    `/v1/chat/completions` and `/v1/completions` request bodies are
    dropped (with a warning) before pydantic validation, instead of
    returning a `400`. Useful when an upstream proxy sends vendor-specific
    fields that MAX Serve does not need to honor.

  Both flags default to `False`; the existing strict behavior is
  unchanged. The corresponding `400` error messages now reference the new
  flags. As a side effect, the legacy `/v1/completions` route now surfaces
  `InputError` detail strings to the client instead of the generic
  `"Value error."` message.

- MAX Serve now emits the `maxserve.num_requests_queued` OTel/Prometheus
  metric (changed from an `UpDownCounter` to a synchronous `Gauge`). The
  gauge is sampled once per scheduler iteration from
  `BatchMetrics.publish_metrics` and reports the depth of the scheduler's
  CE / prefill queue (the same value as the `Pending: N reqs` line in
  scheduler logs). It is published by every text-path scheduler that
  drives `BatchMetrics`: `TokenGenerationScheduler` and `PrefillScheduler`
  (via `TextBatchConstructor`), and `DecodeScheduler` (via
  `len(pending_reqs) + len(prefill_reqs)`). Operators can use this metric
  to observe queue buildup during overload conditions.

- Added a `"none"` option for `runtime.tool_parser` and
  `runtime.reasoning_parser` in `PipelineConfig` (CLI flags `--tool-parser`
  and `--reasoning-parser`). Pass `none` (case-insensitive) to explicitly
  disable the parser, overriding any architecture-declared default. Leaving
  the field unset still applies the architecture default as before.

- Added the `nemotron-opencode` benchmark dataset backed by
  `nvidia/Nemotron-SFT-OpenCode-v1`. Each row is a full Qwen3-Coder OpenCode
  trace (system prompt, multi-turn user/assistant/tool messages, and tool
  schemas). Multi-GB per subset, so the loader streams via
  `datasets.load_dataset(..., streaming=True)` and pulls only enough rows to
  satisfy `--num-prompts`. Tool definitions per row are surfaced on
  `NemotronOpenCodeBenchmarkDataset.last_loaded_tool_schemas` and (for
  single-turn) attached to `SampledRequest.tools`.

- Benchmark request payloads now forward an OpenAI-style `tools=[...]` field
  on chat-completions requests. `SampledRequest` and `RequestFuncInput` gained
  a `tools: list[dict] | None = None` field;
  `OpenAIChatCompletionsRequestDriver` serialises it into the POST body when
  set. Datasets that supply per-row tool schemas (currently
  `nemotron-opencode`) now exercise the server's tool-call grammar /
  structured-output path end-to-end. Pass `enable_tool_calls=False` on
  Nemotron-OpenCode to suppress forwarding.

- Removed multi-step decode from the text-generation pipelines. The flag
  `--max-num-steps` no longer works.

### `max` CLI

- Added `--devices=gpu:all` to use every visible GPU (including MAX Serve).
- Removed the `default` value for `--devices`; omit `--devices` to use the model
  or config default.
- The serving benchmark entrypoint (`benchmark_serving`) now defaults `--seed`
  to a fixed value instead of drawing a fresh random seed on each run. The seed
  drives the workload generator (input/output lengths, session structure,
  content), so a fixed default makes repeated and scheduled runs reproducible
  and keeps run-to-run deltas reflecting the change under test rather than
  workload-draw variance. To opt back into a fresh seed, pass `--seed none` on
  the CLI (or `seed: null` in a workload/config YAML); the drawn seed is logged
  and recorded with the results so the run stays reproducible after the fact.
- Added `--profile` to `max pipelines generate` for rudimentary,
  one-command profiling. With Nsight Systems (`nsys`) on `PATH` and an
  NVIDIA GPU, the timed run is captured into an `.nsys-rep` file and a
  ranked top-N GPU kernel summary is printed. Without `nsys`, a Python/CPU
  profile is produced from `cProfile`. The capture window is bounded by
  `cudaProfilerStart`/`Stop` so warmup and graph-compile time are excluded.
  Use `--profile-output` to override the report path.
- Added `--profile` to `max pipelines benchmark` as a synonym for
  `--trace` that also prints a ranked top-N GPU kernel summary at the end
  of the run. The server still needs to be launched under `nsys launch`
  (matching the existing `--trace` requirement); `--profile` removes the
  "now run `nsys stats` by hand" step.

### Python API

- Reduced default signal buffer size from 1025 to 257 MiB per GPU and fixed
  miscalculation of required space in `MOGGKernelAPI.mojo`. Calculation was
  wrong by a factor of `1/num_devices` since each device only needs scratch
  for its own portion of the collective problem. Reduces footprint for current
  heaviest workload (Kimi-K2.5 with `BlockCopyEngine`) from 16GB to 4GB.

- Added `max.driver.CompletionFlag`, an 8-byte completion flag in pinned host
  memory mapped into a device's address space. Lets host code signal a GPU
  stream (or peer host observer) by writing a 64-bit value to a single
  location visible to both. Currently CUDA-only; constructing against any
  other backend raises `RuntimeError`.

- Added `Device.__unsafe_enqueue_async_py_host_func(fn, flag, value, cpu)`
  and `DeviceStream.wait_for_host_value(flag, value)` for dispatching a
  Python callable onto an explicit AsyncRT worker pool from a host-function
  node and gating the GPU stream on its completion (via the
  `CompletionFlag`). The kickoff trampoline returns immediately, letting
  the GPU stream proceed concurrently with the worker; a downstream
  `wait_for_host_value` blocks the stream until the worker stores `value`.
  The `__unsafe_` prefix marks that the API has no safety net for
  callbacks that capture state outliving the compiled graph.

- Added the `mo.wait_host_value` graph op and the
  `max.nn.kernels.wait_host_value()` Python helper that wraps it. Stalls
  the device stream until a 64-bit host-visible flag reaches a given
  value; lowers to CUDA's `cuStreamWaitValue64` and captures cleanly into
  a CUDA graph as a wait-value node. Lets a captured forward graph gate
  a downstream consumer kernel on CPU-produced data while the rest of
  the forward body runs concurrently. Pair with `mo.launch_host_func`
  or `Device.__unsafe_enqueue_async_py_host_func` to issue the host
  work whose completion the consumer waits on.

- Added two new nanobind types to `max._core.engine` that split the
  compile-and-load pipeline at the type level:

  - `CompiledModels` represents the compile artifact returned by
    `compile_from_path` / `compile_from_object` on the
    `max._core.engine.InferenceSession` binding (these methods don't exist on
    the public `max.engine.InferenceSession` class). It holds the MEF bytes
    and one or more sub-models; it is not directly executable.
  - `ModelMetadata` exposes per-sub-model metadata (`name`,
    `input_metadata`, `output_metadata`) and is yielded by iterating a
    `CompiledModels` or indexing it with `[i]`.

  `Model` continues to represent the runnable, post-init handle (still
  produced by `InferenceSession._load_all`). The high-level
  `max.engine.CompiledModel` wrapper now holds a `CompiledModels` instance
  internally.
- Increased the default allreduce signal buffer size from 513 MiB to 1025 MiB
  per GPU (`max.nn.comm.allreduce.Signals.NUM_BYTES` and the matching constant
  in `max.experimental.realization_context`). The previous 512 MiB scratch
  could not hold the per-peer allgather intermediate for models with large
  hidden dimensions (for example, Kimi-K2.5 at `hidden_dim=20480` with
  `max-batch-input-tokens=16384` needs 640 MiB in bf16). This adds ~512 MiB
  of per-GPU memory use for any multi-GPU model.

- `max.experimental.functional.while_loop` now passes `Tensor` (not
  `TensorValue`) into its `predicate` and `body` callbacks. Callbacks can
  use ordinary `Tensor` operations directly, without wrapping arguments
  via `Tensor.from_graph_value(...)` or reaching for the
  underscore-prefixed `_graph_value` attribute on returns.

- `max.experimental.nn.Module.compile()` now emits the same
  `Building and compiling {ClassName}... / Still building... / Building
  {ClassName} graph took Ns / Compiling {ClassName} took Ms / Building and
  compiling {ClassName} took Ts` log sequence that pipeline-level
  `CompilationTimer` produces today, and wraps the compile body in
  `max.profiler.Tracer` spans (`Module.compile({ClassName})`,
  `Module.compile.trace`, `Module.compile.session_load`) so an `nsys` capture
  with `MODULAR_ENABLE_PROFILING=1` shows compilation as named ranges.
  Every ModuleV3 caller — including pixel-generation pipelines that previously
  compiled silently — now gets this observability for free. The outer
  `CompilationTimer("model")` wrappers in `*_modulev3` architectures have been
  removed to avoid nested timing logs.

- `max.experimental.nn.Module.load_state_dict` and
  `Module.compile(weights=...)` now accept an `auto_cast` keyword
  (default `False`). The framework remains strict by default. When
  `auto_cast=True` is passed, loaded weights are automatically cast
  between `float32` and `bfloat16` when shapes match, logging a single
  summary message per load instead of raising. Other dtype mismatches
  (`float16`, `fp8`, `fp4`, integers, etc.) continue to raise as before.
  This removes the need for per-adapter `astype` shims when checkpoint
  dtypes differ from the module's declared parameter dtype. MAX
  pipelines opt in via the `MODULAR_AUTO_CAST_WEIGHTS` environment
  variable (default `true`, parsed by
  `max.pipelines.lib.weight_loading.auto_cast_weights_from_env`).

- `CPUMetricsCollector` in `max.diagnostics.cpu` is now used as a context
  manager instead of `start`/`stop` and now exposes `get_stats()` instead of
  `dump_stats()`, matching the interface of `GPUDiagContext`.

- `max.graph.Module` is now a public class for grouping multiple `Graph`
  instances into a single compilation unit, replacing the previous alias
  for the underlying MLIR module. Construct one with `Module()` and pass
  it as the `module=` argument to each `Graph`; the resulting `Module` is
  what you hand to `InferenceSession.load_all` to compile every graph
  together. `Graph.empty_module()` has been removed in favor of `Module()`,
  and `Graph` now exposes a `module` property returning the `Module` it
  belongs to.

- `InferenceSession.load_all` now returns a `dict[str, Model]` keyed by each
  model's `sym_name` (the name of its `mo.graph` op), instead of a
  `list[Model]` ordered by MEF position. The accepted input type also gained
  `max.graph.Module`, so callers can compile a pre-built module containing
  multiple `mo.graph` ops directly. `Model` now exposes a `name` property.

  Migrate positional unpacking call sites by indexing the returned dict:

  ```python
  # Before
  module = Graph.empty_module()
  with Graph("vision", input_types=..., module=module): ...
  with Graph("language", input_types=..., module=module): ...
  vision_model, language_model = session.load_all(graph, ...)

  # After
  module = Module()
  with Graph("vision", input_types=..., module=module) as vision_graph: ...
  with Graph("language", input_types=..., module=module) as language_graph: ...
  models = session.load_all(module, ...)
  vision_model = models[vision_graph.name]
  language_model = models[language_graph.name]
  ```

## MAX kernels

- The `use_blocking_impl` parameter has been removed from the `foreach` custom
  op helper (and the underlying `elementwise` primitive), and the analogous
  `single_thread_blocking_override` parameter has been removed from the `concat`
  and `concat_shape` kernels and the reduction-based kernels. Work is always
  dispatched the same way, with a single worker used automatically when the
  problem size is small. The dedicated small-tensor `concat` fast path has been
  removed in favor of the existing serial/parallel dispatch.
- Updated `elementwise` call sites across MAX kernels and benchmarks to use
  `Coord`-native indexing, fixing compile failures caused by invalid
  `Coord`/`IndexList` conversions.
- Enabled Programmatic Dependent Launch (PDL) for the SM100 (Blackwell)
  FlashAttention-4 prefill kernel, letting back-to-back attention grids in a
  stream overlap launch and prologue latency. This reduces per-launch overhead
  most for shorter sequences (measured ~1.05x–1.5x faster on B200, bf16,
  head_dim=128 across seq lengths 128–2048). On by default; disable with
  `-D MHA_PDL=false`.

## Breaking changes

- KV cache management has moved from `max.kv_cache` to `max.pipelines.kv_cache`.
  Update imports accordingly:

  ```python
  # Before
  from max.kv_cache import PagedKVCacheManager, DummyKVCache

  # After
  from max.pipelines.kv_cache import PagedKVCacheManager, DummyKVCache
  ```

  Deprecation shims with `DeprecationWarning` remain at the old path.

- Custom Mojo ops used through `max.experimental.torch.CustomOpLibrary` (and
  the rest of the graph-compiler custom-op path) must now declare their
  `ctx` parameter as `DeviceContext` instead of `DeviceContextPtr`. The
  `DeviceContextPtr` type has been removed from the Mojo standard library;
  see the [Mojo nightly
  changelog](https://docs.modular.com/mojo/changelog/) entry under
  *Removed* for the full migration. Multi-device ops should declare their
  variadic context argument as `DeviceContextList[N]` (also new — see the
  Mojo changelog *GPU programming* section).

- GPU and CPU diagnostic tooling has moved from `max.diagnostics` to
  `max.profiler`: `max.diagnostics.gpu` → `max.profiler.gpu` and
  `max.diagnostics.cpu` → `max.profiler.cpu`. Update imports accordingly.
  Deprecation shims with `DeprecationWarning` remain at the old paths.

- `max/python/max/benchmark/benchmark_throughput.py`, deprecated in v0.26.3,
  has been removed.

## Fixes

- Fixed an expert-parallelism dispatch assertion (`Cannot dispatch EP
  kernel with N input tokens when the maximum tokens per rank is N-1`)
  that fired whenever `--max-batch-input-tokens` was not evenly
  divisible by the tensor-parallel degree. The EP per-rank cap now uses
  ceiling division to match the ragged binning of `reducescatter` in
  TP-attention + EP-MoE mode, so the largest shard fits in the
  dispatch buffer. Affects DeepSeek-V3, Kimi-K2.5, MiniMax-M2, Qwen3,
  and Step3.5 deployments configured with non-divisible batch sizes.

- `MODULAR_DEBUG=ir-output-dir=<dir>` (and the equivalent
  `[max-debug] ir-output-dir = <dir>` config-file entry and
  `InferenceSession.debug.ir_output_dir = <dir>` Python setter) now
  actually dumps per-stage MLIR files to the configured directory. The
  option was previously parsed but no compiler stage consulted it, so
  users had to fall back to the legacy `MODULAR_MAX_TEMPS_DIR` env var.
  Both spellings are now honored.

## Mojo language

For all the updates to the Mojo language, standard library, and tools,
see the [Mojo release notes](https://mojolang.org/releases).
