---
title: Nightly (v26.5)
---

This version is still a work in progress.

## Highlights

## Documentation

## MAX models

- Added tool-calling and reasoning support to Qwen 3.5 / 3.6.

## MAX framework

### Inference server

- Fixed a structured-output runaway: a `response_format` JSON schema that omits
  the root `"type"` (for example `{"properties": {"x": {}}}`, valid JSON Schema)
  previously compiled to a grammar that permitted a bare, unbounded top-level
  value, so a model that looped inside that value could never emit a terminator
  and generated until `max_length` (`finish_reason="length"`). Such schemas with
  an object-implying keyword (`properties`, `required`, `additionalProperties`,
  `patternProperties`) are now normalized to `"type": "object"` before grammar
  compilation, matching the behavior of xgrammar-based engines. A genuinely
  empty `{}` schema is still treated as "any value".
- Retuned the Prometheus/OpenTelemetry histogram buckets for MAX Serve metrics.
  Previously every histogram shared one millisecond-latency bucket range, which
  was inaccurate for non-latency metrics. Each histogram now uses bucket
  boundaries matched to its actual range (percentages bucket 0–100, token and
  occupancy counts use power-of-two buckets, batch size is fine-grained up to
  512, throughput and time metrics use appropriately wide ranges, and time
  metrics now extend out to 30 minutes). Quantile queries become more accurate;
  dashboards that hardcoded specific bucket boundaries may need updating.
- Changed `maxserve.cache.num_used_blocks` and `maxserve.cache.num_total_blocks`
  from counters to gauges. These report an instantaneous level, so a gauge is
  correct; as counters their exported values were meaningless. The Prometheus
  type changes to `gauge` and the exported series drops the counter `_total`
  suffix.
- Added `maxserve.cache.disk_blocks_read` and
  `maxserve.cache.disk_blocks_written` counters, reporting KV blocks read from
  and written to the disk cache tier when tiered (disk) KV caching is enabled.

### `max` CLI

### Python API

- Eager execution in `max.experimental` now routes every realization through
  the `max.experimental.executor.Executor` abstraction. The out-of-the-box
  path is unchanged — graphs within the `MAX_INTERPRETER_MAX_OPS` threshold run
  on the interpreter and fall back to a cached compile otherwise — but it is
  now expressed as a new `CompositeExecutor` selected by
  `MAX_EAGER_EXECUTOR=composite` (the new default). The
  `MAX_USE_EAGER_INTERPRETER` environment variable has been removed; force
  compilation with `MAX_EAGER_EXECUTOR=compile` instead. The
  `EagerRealizationContext(use_interpreter=...)` argument is deprecated in
  favor of `EagerRealizationContext(executor=...)`.

## MAX kernels

## Breaking changes

## Fixes

- Fixed a constrained-decoding bug that could intermittently drop grammar
  enforcement during speculative decoding with grammar-guided tool calling.
  The speculative bitmask walk advanced the matcher through draft tokens and
  restored it with `rollback`, but `rollback` does not correctly restore the
  matcher across certain tool-call structural tags (e.g.
  `<|tool_call_begin|>`). The walk now runs on a deep copy of the matcher,
  leaving the real matcher untouched.

## Mojo language
