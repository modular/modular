---
title: Nightly (v26.4)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added MXFP4 quantization support for MiniMax-M2.

## MAX framework

### Inference server

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

### `max` CLI

- Added `--devices=gpu:all` to use every visible GPU (including MAX Serve).
- Removed the `default` value for `--devices`; omit `--devices` to use the model
  or config default.

### Python API

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

## Breaking changes

- GPU and CPU diagnostic tooling has moved from `max.diagnostics` to
  `max.profiler`: `max.diagnostics.gpu` → `max.profiler.gpu` and
  `max.diagnostics.cpu` → `max.profiler.cpu`. Update imports accordingly.
  Deprecation shims with `DeprecationWarning` remain at the old paths.

- `max/python/max/benchmark/benchmark_throughput.py`, deprecated in v0.26.3,
  has been removed.

## Fixes

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
