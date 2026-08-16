# Cascade serve optimization log

Running record of cascade optimizations measured against the **real** Llama 3.1
8B model (full GPU forward pass), not echo mode. Where
[echo_optimization_log.md](echo_optimization_log.md) isolates framework overhead
by swapping the model for a token-replay worker, this log measures end-to-end
serving on the model an operator actually runs, with **MAX Serve as the
reference point**. Same sharegpt workload (concurrency 600, 10000 prompts). See
[benchmarking.md](benchmarking.md) for the procedure.

The client command is invariant across all runs (only `--result-filename`
changes):

```bash
./bazelw run --config=production //max/python/max/_entrypoints:benchmark -- \
  benchmark \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --dataset-name sharegpt --num-prompts 10000 --max-concurrency 600 \
  --host localhost --port 8000 --seed 0 \
  --result-filename /tmp/cascade_serve_c600_n10000.json --always-save-result
```

Because this is the real model, req/s is GPU-bound at the ceiling: the goal of
each change is to stop the framework from pacing the GPU, i.e. close the gap to
MAX Serve on req/s while keeping cascade's TTFT advantage. Append a new block
per change as we go.

## Test platform

- CPU: Intel Xeon 6972P (Granite Rapids), 2 sockets
- Cores: 192 physical / 384 logical (2 threads per core)
- RAM: 1510 GiB
- GPU: 1x NVIDIA B200 (183 GB)
- Architecture: x86_64

## MAX Serve reference

The production serving stack, for reference. Same client command and workload.

Server command:

```bash
./bazelw run --config=production //max/python/max/_entrypoints:pipelines -- \
  serve --model modularai/Llama-3.1-8B-Instruct-GGUF --port 8000
```

Results:

- Req/s: 80.00
- Output tok/s (mean): 16016
- TTFT mean / p50 / p99 (ms): 3478.6 / 3627.1 / 5747.8
- TPOT mean / p50 / p99 (ms): 18.01 / 17.00 / 37.47
- Requests: 9808 completed, 0 failed; duration 122.6 s

## Cascade baseline (branch HEAD before this change)

Cascade at the transport-coalescing operating point from the echo log (HTTP,
`--local-cpu-workers 8`), before the SSE-formatting change below. This is the
honest starting point on the real model.

Server command:

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path modularai/Llama-3.1-8B-Instruct-GGUF \
  --local-cpu-workers 8 --host localhost --port 8000
```

Results:

- Req/s: 74.74
- Output tok/s (mean): 14590
- TTFT mean / p50 / p99 (ms): 2000.8 / 2006.3 / 5176.7
- TPOT mean / p50 / p99 (ms): 25.37 / 23.05 / 65.74
- Requests: 9127 completed, 0 failed; duration 122.1 s

Cascade reaches the first token much sooner than MAX Serve (2006 vs 3627 ms p50,
its tokenizer worker pool keeps prefill latency low) but trails on req/s (74.7
vs 80.0, -6.6%) and TPOT (23.1 vs 17.0 ms p50). Profiling (see the API
event-loop block in the echo log, plus a real-model CPU profile) traced the
throughput gap to the single API event loop being GIL-bound: it ran a per-token
pydantic `model_dump_json` to frame each OpenAI `chat.completion.chunk` SSE
event, so serialization paced the decode stream.

## Worker-offloaded OpenAI SSE formatting (http, `--local-cpu-workers 8`)

Moves the per-token OpenAI SSE framing off the single API event loop and onto
the round-robin CPU worker pool. A stream-handle contract on `TextGenInterface`
(`open_text_stream` returns a worker stream handle; `generate_text` is a thin
default over it) lets a serving-layer `OpenAIChatCompletionPipeline` chain an
`OpenAIChatFormatter` worker onto that handle. The token stream flows
tokenizer -> formatter worker-to-worker; the API process only forwards finished
byte frames, so the per-token `model_dump_json` parallelizes across workers
instead of serializing on one GIL-bound loop. The OpenAI wire framing lives in
`serve`, keeping the generic interface and pipelines free of protocol concerns.

Server command (unchanged from baseline; the change is in the pipeline wiring):

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path modularai/Llama-3.1-8B-Instruct-GGUF \
  --local-cpu-workers 8 --host localhost --port 8000
```

Results (3-way, same client command and workload):

| config                          | Req/s | Out tok/s | TTFT mean/p50/p99 (ms)   | TPOT mean/p50/p99 (ms) | Completed | Duration (s) |
|---------------------------------|-------|-----------|--------------------------|------------------------|-----------|--------------|
| MAX Serve (reference)           | 80.00 | 16016     | 3478.6 / 3627.1 / 5747.8 | 18.01 / 17.00 / 37.47  | 9808      | 122.6        |
| cascade (baseline)              | 74.74 | 14590     | 2000.8 / 2006.3 / 5176.7 | 25.37 / 23.05 / 65.74  | 9127      | 122.1        |
| cascade (worker SSE formatting) | 80.10 | 15763     | 2136.5 / 2033.5 / 5100.3 | 24.05 / 21.15 / 65.19  | 9773      | 122.0        |

Conclusion: offloading SSE formatting lifts cascade throughput **74.7 -> 80.1
req/s (+7.2%)**, closing the gap to MAX Serve (80.0) and even edging past it,
while keeping cascade's large TTFT advantage (2033 vs 3627 ms p50, ~1.8x lower).
TPOT also improves (23.05 -> 21.15 ms p50). GPU utilization on the cascade run
was 97.7%, so the framework is no longer pacing the GPU at this operating point
— req/s is now GPU-bound, which is the goal. 0 failed requests.

TPOT still trails MAX Serve (21.2 vs 17.0 ms p50): the remaining ceiling is
decode-step pacing (host work interleaved with the in-flight forward), not SSE
framing.

## Native multidict/frozenlist wheels for the aiohttp transport (http, `--local-cpu-workers 8`)

The API event-loop profile in
[echo_optimization_log.md](echo_optimization_log.md) found ~14% of loop time in
aiohttp's pure-Python `_multidict_py.py` fallback: the pinned multidict 6.0.5 /
frozenlist 1.4.1 predated CPython 3.13 native wheels, so the HTTP runtime
transport built request header maps in pure Python. Upgrading to multidict
6.7.1 / frozenlist 1.8.0 (which ship cp313 native extensions) restores the
compiled C backends. The dependency bump lands via a separate change
(SERVSYS-1295), not this branch; this block records its measured effect on the
serve workload, stacked on the worker-SSE-formatting change above.

Server command (unchanged):

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path modularai/Llama-3.1-8B-Instruct-GGUF \
  --local-cpu-workers 8 --host localhost --port 8000
```

Results (4-way, same client command and workload):

| config                          | Req/s | Out tok/s | TTFT mean/p50/p99 (ms)   | TPOT mean/p50/p99 (ms) | Completed | Duration (s) |
|---------------------------------|-------|-----------|--------------------------|------------------------|-----------|--------------|
| MAX Serve (reference)           | 80.00 | 16016     | 3478.6 / 3627.1 / 5747.8 | 18.01 / 17.00 / 37.47  | 9808      | 122.6        |
| cascade (baseline)              | 74.74 | 14590     | 2000.8 / 2006.3 / 5176.7 | 25.37 / 23.05 / 65.74  | 9127      | 122.1        |
| cascade (worker SSE formatting) | 80.10 | 15763     | 2136.5 / 2033.5 / 5100.3 | 24.05 / 21.15 / 65.19  | 9773      | 122.0        |
| cascade (+ native multidict)    | 83.56 | 16372     | 2026.5 / 1857.0 / 5274.2 | 23.58 / 21.88 / 61.08  | 9542      | 114.2        |

Conclusion: the native C multidict backend adds **+4.3% req/s (80.1 -> 83.6)**
on top of worker-SSE formatting, **+11.8% over baseline (74.7 -> 83.6)**.
Cascade now beats MAX Serve on both throughput (83.6 vs 80.0) and output tok/s
(16372 vs 16016), with the lowest TTFT yet (1857 ms p50, ~1.95x below MAX
Serve). Duration drops 122.0 -> 114.2 s and TPOT p99 improves (65.2 -> 61.1 ms);
TPOT p50 is flat (~22 ms). GPU utilization 97.4%, 0 failed requests. CPU user
utilization rose (~530% -> ~681% of one core) as expected: the compiled
multidict lets the API loop dispatch worker RPCs faster, so more host work fits
in the same wall-clock.

TPOT p50 (~22 vs 17 ms) remains the gap to MAX Serve — decode-step pacing, not
transport or serialization, is the next frontier.

## Why cascade's TPOT trails MAX Serve: admission-rate, not a config bug

Cascade wins req/s (83.6 vs 80.0) and TTFT (1857 vs 3627 ms p50) but loses TPOT
(21.9 vs 17.0 ms p50). Since both stacks run the *same* model worker, we traced
whether cascade forwards the scheduler config wrong. It does not:

- `MAXModelWorker.open()` spawns the same `max.serve` model-worker subprocess
  via `start_model_worker`; the `TokenGenerationScheduler` runs *inside* that
  subprocess, so the batching policy is MAX Serve's, not a cascade
  reimplementation.
- `TokenGenerationSchedulerConfig.from_pipeline_config()` derives every batching
  knob (`max_batch_input_tokens`, `enable_chunked_prefill`,
  `enable_in_flight_batching`, `max_batch_total_tokens`, `max_batch_size`) from
  `pipeline_config.runtime.*` and `pipeline.max_batch_size`. `load_scheduler`
  reads only `max_pending_requests` from `Settings`, which is `None` (unbounded)
  in both stacks. The `Settings(offline_inference=True, ...)` in
  `max_model_worker.py` only gates whether an HTTP API is spun up; it touches
  nothing in the scheduler.
- Cascade's `build_pipeline` resolves config through the same
  `PIPELINE_REGISTRY.retrieve_factory` path serve uses. Runtime proof from the
  server log: `max_batch_size: 512`, device-graph-capture cap 128 — matching
  what `pipelines serve` resolves for this model/GPU.

The gap is an operating-point effect. A concurrency sweep on the *same* cascade
server (worker-SSE + native multidict) shows TPOT is a function of how many
requests decode concurrently, not a fixed overhead:

| concurrency | Req/s | TTFT p50 (ms) | TPOT p50 (ms) | GPU util |
|-------------|-------|---------------|---------------|----------|
| 32          | 29.8  | 21            | 4.9           | 95%      |
| 128         | 65.7  | 219           | 7.7           | 95%      |
| 600         | 83.6  | 1857          | 21.9          | 97%      |

TPOT scales ~4.5x with concurrency (4.9 -> 21.9 ms) on identical code, and
cascade's decode floor (4.9 ms at c=32) is well below MAX Serve's 17 ms. What
differs at c600 is admission: cascade's parallel tokenizer worker pool reaches
first token ~1.8x sooner, so by Little's law a larger share of the 600 in-flight
requests sit in the decode stage at any instant (larger, more compute-bound
decode batches with more prefill-chunk interleaving — visible as the bimodal
step-TPOT, p50 0.04 ms / p99 456 ms). MAX Serve parks more requests in the
prefill queue (higher TTFT), running smaller, cleaner decode batches for lower
TPOT.

Conclusion: same Pareto frontier, different operating point. MAX Serve's lower
TPOT is bought with higher TTFT via the same knob; cascade is deliberately
further toward the throughput/TTFT-favoring end. Trading back toward MAX Serve's
TPOT profile is an admission-throttling policy choice (e.g.
`max_pending_requests` or a decode-occupancy cap), not a bug fix.
