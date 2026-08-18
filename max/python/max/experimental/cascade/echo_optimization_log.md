# Cascade optimization log

Running record of cascade framework-overhead optimizations. Each block below is
one change, measured against the same echo-mode sharegpt workload (concurrency
600, 10000 prompts). See [benchmarking.md](benchmarking.md) for the full
procedure.

The client command is invariant across all runs:

```bash
./bazelw run --config=production //max/python/max/_entrypoints:benchmark -- \
  benchmark \
  --model modularai/Llama-3.1-8B-Instruct-GGUF \
  --dataset-name sharegpt --num-prompts 10000 --max-concurrency 600 \
  --host localhost --port 8000 --seed 0 \
  --result-filename /tmp/cascade_sharegpt_c600_n10000.json --always-save-result
```

Append a new block (copy the Baseline block) per change as we go.

## Test platform

- CPU: Intel Xeon Platinum 8488C (Sapphire Rapids), 1 socket
- Cores: 16 physical / 32 logical (2 threads per core)
- Clock: 2.40 GHz base (no min/max scaling exposed; virtualized instance)
- RAM: 61 GiB, no swap
- Architecture: x86_64

## Baseline

(no changes)

Server command:

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path echo:modularai/Llama-3.1-8B-Instruct-GGUF \
  --host localhost --port 8000
```

Results:

- Req/s: 108.5
- Output tok/s (mean): 15847
- TTFT mean / p99 (ms): 3337.9 / 5336.4
- E2E mean / p99 (ms): 3551.1 / 5589.8
- TPOT mean (ms): 0.04
- Requests: 9867/10000 succeeded (133 filtered client-side); duration 90.9 s

## Tokenizer CPU workers (`--local-cpu-workers`)

`--local-cpu-workers` sizes the local CPU worker pool that runs HF
tokenization; each worker is its own subprocess, so raising it parallelizes
encode/detokenize across cores. Default is 2. Swept the value while holding the
client command constant; every other flag matches the baseline server command.

Server command (optimal shown; only `--local-cpu-workers` varied across runs):

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path echo:modularai/Llama-3.1-8B-Instruct-GGUF \
  --local-cpu-workers 8 \
  --host localhost --port 8000
```

Sweep results (req/s is the headline framework-overhead metric):

| `--local-cpu-workers` | Req/s | Out tok/s (mean) | TTFT mean/p99 (ms) | E2E mean/p99 (ms) | Duration (s) |
|-----------------------|-------|------------------|--------------------|-------------------|--------------|
| 2 (default)           | 107.2 | 15515            | 3415.9 / 5610.7    | 3641.2 / 5956.5   | 92.0         |
| 4                     | 111.4 | 17946            | 3341.3 / 5284.4    | 3479.7 / 5571.2   | 88.5         |
| 8                     | 115.2 | 18287            | 3342.9 / 5135.6    | 3455.8 / 5472.2   | 85.6         |
| 16                    | 115.3 | 18286            | 3431.6 / 5248.8    | 3529.5 / 5553.1   | 85.6         |
| 32                    | 112.6 | 18373            | 3326.8 / 5073.2    | 3435.0 / 5396.2   | 87.6         |

Conclusion: throughput climbs from 2 to 8 workers (+7.5%, 107.2 to 115.2 req/s),
then flattens at 16 (115.3, within noise) and regresses at 32 (112.6). The knee
is 8 — half the 16 physical cores, leaving headroom for the echo/GPU worker and
the API process. Pushing to 32 oversubscribes the 16 physical cores (2 logical
threads each) and also inflates startup, since each worker loads the tokenizer.

Recommended: `--local-cpu-workers 8` on this 16-core host. Scale the setting
with physical core count rather than logical CPUs.

## Transport: gRPC vs HTTP (at `--local-cpu-workers 8`)

`--transport` selects how the API process talks to the worker subprocesses:
`http` (default, pickle-over-HTTP via aiohttp) or `grpc`. Compared the two at
the locked-in 8 CPU workers, same echo-mode sharegpt workload.

gRPC server command:

```bash
./bazelw run --config=production \
  //max/python/max/experimental/cascade/serve:main -- \
  --models.main.model-path echo:modularai/Llama-3.1-8B-Instruct-GGUF \
  --transport grpc --local-cpu-workers 8 \
  --host localhost --port 8000
```

Results:

| Transport | Req/s | Out tok/s (mean) | TTFT mean/p99 (ms) | E2E mean/p99 (ms) | TPOT mean (ms) | Duration (s) |
|-----------|-------|------------------|--------------------|-------------------|----------------|--------------|
| http      | 115.2 | 18287            | 3342.9 / 5135.6    | 3455.8 / 5472.2   | 0.04           | 85.6         |
| grpc      | 65.6  | 129              | 1073.5 / 1586.3    | 7954.0 / 27503.5  | 45.35          | 149.7        |

Conclusion: gRPC is ~1.8x slower end-to-end at concurrency 600 (65.6 vs 115.2
req/s; duration 149.7 vs 85.6 s). The signature is diagnostic: gRPC gets to the
first token *faster* (TTFT 1073 vs 3343 ms) but its per-request latency
explodes (E2E p99 27.5 s vs 5.5 s; TPOT 45 vs 0.04 ms), i.e. requests stall
mid-stream, not at dispatch.

Root cause (investigated): it is contention on the single echo/GPU worker's
gRPC server, not a transport round-trip cost and not the HTTP/2 stream cap.

- Ruled out the stream cap: bumping `grpc.max_concurrent_streams` to 2048 on
  the server changed nothing (66.2 req/s, TPOT 44.4 ms — within noise of the
  default). The tokenizer streams fan out over 8 worker channels (~75 each,
  under any cap); all generation funnels to the *one* echo worker.
- Ruled out per-token latency: at `--max-concurrency 1` gRPC is fast — TPOT
  0.18 ms, TTFT 17 ms, 5486 out tok/s. The 44 ms/token only appears under high
  concurrency, so it is queueing, not a fixed per-message cost.

The single echo worker's `grpc.aio` server saturates servicing 600 concurrent
worker-to-worker streams: Python gRPC aio has much higher per-message CPU
overhead than aiohttp, so the one process every token stream funnels through
can't keep up. HTTP scales cleanly through the identical topology (same 8
tokenizer workers, same single echo worker).

Recommended: keep `--transport http` (the default). A real gRPC fix is not a
one-liner — it needs lower per-message overhead on the hot worker (e.g. batch
multiple tokens per stream message, or reduce the chained call_method +
stream_result RPC count per request), not a channel/stream-count knob.

## Detokenization ceiling probe: batch decode (http, `--local-cpu-workers 8`)

Diagnostic, not a shippable change. Temporarily rewrote
`MAXTokenizer.decode_stream` to drain the entire model-worker token stream into
one list and issue a single `tokenizer.decode(all_ids)` at the end, instead of
the offset-based incremental decode (two HF `decode` calls per chunk). This
destroys TPOT/streaming — the whole response arrives in one terminal chunk — but
isolates how much of the framework's req/s ceiling is detokenization CPU. Same
echo-mode sharegpt workload and client command; only `decode_stream` changed
against the http + 8-worker operating point.

Results (vs the incremental-decode http baseline at the same 8 workers):

| decode_stream         | Req/s | TTFT mean/p99 (ms) | E2E mean/p99 (ms) | TPOT mean (ms) | Duration (s) |
|-----------------------|-------|--------------------|-------------------|----------------|--------------|
| incremental (2/chunk) | 115.2 | 3342.9 / 5135.6    | 3455.8 / 5472.2   | 0.04           | 85.6         |
| batch (1/request)     | 299.4 | 1903.6 / 2407.4    | 1903.6 / 2407.5   | 0.00 (n/a)     | 33.4         |

Conclusion: detokenization is the dominant framework-overhead cost at this
operating point — collapsing per-chunk decode to one decode-per-request is a
**2.6x** req/s jump (115.2 -> 299.4) and cuts wall-clock 85.6 -> 33.4 s. TTFT
also nearly halves (3343 -> 1904 ms) because the tokenizer workers are no longer
CPU-saturated re-decoding the growing tail every chunk. All 10000 requests
succeeded. The reported output tok/s is meaningless here (the client sees one
terminal chunk, so its per-token rate is a divide-by-tiny artifact) — req/s and
duration are the honest metrics.

This says the incremental-detokenization path is worth real investment: the
offset-based decode re-runs HF `decode` over the un-emitted tail on every chunk,
and that CPU work — not transport or worker fan-out — is what caps throughput.
Shippable directions (keep streaming, cut the per-chunk cost): decode only newly
completed tokens instead of re-decoding the window (fast tokenizers expose
byte-level piece APIs), coalesce N tokens per detokenize step (bounded-latency
micro-batching) to amortize the two `decode` calls, or move detokenization off
the Python hot path. None wreck TPOT the way this probe does.

## Transport-level token coalescing (http, `--local-cpu-workers 8`)

Shippable realization of the "coalesce N tokens per detokenize step" direction,
implemented entirely in the transport with no API or worker-topology change.
`HttpRuntimeProxy.stream_result` (`http_runtime/client.py`) previously read one
length-prefixed frame at a time and yielded one token array per frame, so
`decode_stream` ran its two HF `decode` calls once per token. The change reads
each drain as a *batch*: `StreamReader.readany` returns all currently buffered
bytes (blocking only when the buffer is empty), so while a synchronous decode
holds the event loop, inbound token frames pile up in the socket buffer and the
next drain scoops the whole backlog. Consecutive 1-D `int32` frames (the token
stream — image/latent/text streams are excluded by dtype/ndim) are concatenated
into one array before yielding, so the detokenizer amortizes one decode over
however many tokens arrived together. The batch size is self-tuning: under light
load it is one token (identical to before, natural back-pressure); under the
detok-bound concurrency-600 load it averaged ~23 tokens per emitted text chunk.

Results (vs the incremental-decode http baseline at the same 8 workers; the
batch-decode probe above is the non-streaming ceiling for reference):

| decode path                     | Req/s | TTFT mean/p99 (ms) | E2E mean/p99 (ms) | TPOT mean (ms) | Nonempty chunks | Duration (s) |
|---------------------------------|-------|--------------------|-------------------|----------------|-----------------|--------------|
| incremental (2 decodes/token)   | 115.2 | 3342.9 / 5135.6    | 3455.8 / 5472.2   | 0.04           | ~1/token        | 85.6         |
| transport coalescing (stream)   | 232.6 | 1929.8 / 2411.1    | 1933.6 / 2415.7   | 0.01           | 71789 (~9/req)  | 34.3         |
| batch decode (1/request, probe) | 299.4 | 1903.6 / 2407.4    | 1903.6 / 2407.5   | n/a (1 chunk)  | ~1/req          | 33.4         |

Conclusion: coalescing at the transport is a **2.0x** req/s win (115.2 -> 232.6)
that *keeps streaming* — 71789 nonempty chunks across 7965 requests (~9 per
request, vs one terminal chunk for the probe), TPOT 0.01 ms, ITL 0.11 ms. It
captures ~78% of the non-streaming ceiling (299.4) while the probe captured 100%
by destroying TPOT. TTFT nearly halves (3343 -> 1930 ms) for the same reason as
the probe: the tokenizer workers stop re-decoding the growing tail on every
single token. 0 failed requests. The change is transport-only; `decode_stream`
and every downstream consumer are untouched, and the dtype/ndim gate leaves
non-token streams (e.g. imgen `UInt8Array` frames) yielded one-per-frame.

## API event-loop profile: router is not saturated (http, `--local-cpu-workers 8`)

Diagnostic, not a change. Checked whether the single-threaded API process (the
FastAPI/asyncio "router" that parses each request and wires the worker RPCs) is
the throughput ceiling after the coalescing win. Profiled the API process with
py-spy during a steady mid-load window (same echo sharegpt c600 workload, 40000
prompts for a long load, 45 s sample) and measured per-thread CPU from `/proc`
CPU-time deltas over the same window.

Saturation (CPU of one core over the window):

| thread                                    | CPU   |
|-------------------------------------------|-------|
| router / asyncio event loop (main thread) | 23.8% |
| whole API process (all threads summed)    | 41.9% |

The event loop has ~76% headroom, so request parsing/translation on it is
**not** the current ceiling — the bottleneck is the tokenizer *worker* processes
(HF detok/encode), which are separate processes not counted above.

py-spy self-time (idle threadpool stripped) shows the loop time that *is* spent
goes mostly to API->worker aiohttp **client** overhead, not incoming-request
parsing: ~14% building request headers in the pure-Python `multidict` fallback
(`_multidict_py.py` — the C speedup appears unused); ~10% per-call socket setup
(`sock_connect`/`_create_connection_transport`/`write` — worker connections are
not reused across calls); ~8% asyncio + context-manager machinery
(`call_method`, `AsyncExitStack`, `call_soon`/`_run_once`, and `get_debug`
firing on the hot path); ~2% SSE output (`model_dump_json` + `sse_starlette`).
Incoming pydantic parsing was only ~3-4% of visible self-time (caveat: pydantic
v2's Rust core is invisible to py-spy, so input-parse cost is under-counted —
but the exact 24% saturation number settles it regardless).

Conclusion: moving request parsing off the loop (e.g. a raw-bytes route +
`tokenizer.process_request`) would not raise throughput now; the loop is not the
bottleneck and parsing is not even its dominant cost. Shelve until the router is
actually hot. If it becomes the ceiling, the higher-value loop levers, in order,
are: reuse worker connections (kill per-call `sock_connect`), verify/fix the C
`multidict` and trim per-RPC header churn, and disable asyncio debug. The real
current ceiling is the tokenizer worker pool — profile those for the next win.
