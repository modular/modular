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

### Python API

- `CPUMetricsCollector` in `max.diagnostics.cpu` is now used as a context
  manager instead of `start`/`stop` and now exposes `get_stats()` instead of
  `dump_stats()`, matching the interface of `GPUDiagContext`.

## MAX kernels

## Breaking changes

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
