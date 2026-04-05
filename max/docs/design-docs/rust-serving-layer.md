# MAX Rust Serving Layer: Architecture and Model Worker Startup

## Scope

This document explains the current `max/serve-rust` frontend, how it handles
requests, and where the MAX model worker is actually started in the repository.
It also clarifies what changed in this work and what did not.

## Short answer: where is the model worker launched?

The MAX model worker is launched in Python, not in `max/serve-rust`.

Primary startup path:

1. `start_workers(...)`
   in `max/python/max/entrypoints/workers/__init__.py`
2. Calls `start_model_worker(...)`
   from `max/python/max/serve/pipelines/model_worker.py`
3. `start_model_worker(...)` uses `subprocess_manager(...)` and
   `multiprocessing.get_context("spawn")` to start `ModelWorker()` in a spawned
   subprocess.

Relevant code references:

- `max/python/max/entrypoints/workers/__init__.py:102`
- `max/python/max/serve/pipelines/model_worker.py:336`
- `max/python/max/serve/pipelines/model_worker.py:362`
- `max/python/max/serve/pipelines/model_worker.py:365`
- `max/python/max/serve/process_control.py:115`
- `max/python/max/serve/process_control.py:135`

## What the Rust serving layer does today

`max/serve-rust` is currently an HTTP frontend plus a ZMQ proxy.

Main entrypoint:

- `max/serve-rust/src/main.rs`

At startup it:

1. Loads env-based settings.
2. Generates 3 random IPC endpoints (`request`, `response`, `cancel`).
3. Builds `ZmqModelWorkerProxy`.
4. Starts a response worker task for ZMQ responses.
5. Creates Axum routes:
   - `/v1/*` OpenAI-style routes
   - `/v2/*` KServe routes
6. Binds and starts Axum on `MAX_SERVE_HOST:MAX_SERVE_PORT`.

Relevant code references:

- `max/serve-rust/src/main.rs:25`
- `max/serve-rust/src/main.rs:28`
- `max/serve-rust/src/main.rs:33`
- `max/serve-rust/src/main.rs:50`
- `max/serve-rust/src/main.rs:64`
- `max/serve-rust/src/main.rs:65`

### Environment variables consumed by Rust frontend

Parsed in `max/serve-rust/src/config.rs`:

- `MAX_SERVE_HOST` (default: `0.0.0.0`)
- `MAX_SERVE_PORT` (default: `8000`)
- `MAX_SERVE_METRICS_ENDPOINT_PORT` (default: `8001`)
- `MAX_SERVE_API_TYPES` (default: `openai,sagemaker`)
- `MAX_SERVE_DISABLE_TELEMETRY`
- `MAX_SERVE_OFFLINE_INFERENCE`
- `MAX_SERVE_HEADLESS`
- `MAX_SERVE_LOGS_CONSOLE_LEVEL` (default: `INFO`)
- `MAX_SERVE_RUST_REQUEST_QUEUE_CAPACITY` (default: `4096`)
- `MAX_SERVE_RUST_CANCEL_QUEUE_CAPACITY` (default: `1024`)
- `MAX_SERVE_RUST_REQUEST_BATCH_MAX_SIZE` (default: `32`)
- `MAX_SERVE_RUST_REQUEST_BATCH_WAIT_US` (default: `200`)

## Rust request flow (OpenAI path)

For `POST /v1/chat/completions`:

1. Parse JSON into `ChatCompletionRequest`.
2. Generate `RequestID`.
3. Enqueue request via `proxy.stream(...)`.
4. Receive streamed token IDs from ZMQ.
5. Decode token IDs with `PythonBridge::decode_tokens(...)`, which imports
   `max.serve.decode` using PyO3.
6. Return either:
   - SSE stream (`stream=true`)
   - single JSON completion (`stream=false`)

Relevant code references:

- `max/serve-rust/src/openai.rs:80`
- `max/serve-rust/src/openai.rs:99`
- `max/serve-rust/src/openai.rs:102`
- `max/serve-rust/src/openai.rs:137`
- `max/serve-rust/src/openai.rs:160`
- `max/serve-rust/src/python_bridge.rs:10`

### OpenAI path sequence (today)

1. Client calls `POST /v1/chat/completions`.
2. Rust parses payload and allocates `RequestID`.
3. Rust serializes `{request_id, request}` via MessagePack.
4. Rust `PushSocket` sends to model-worker request queue.
5. Model worker emits `SchedulerResult` map keyed by `RequestID`.
6. Rust response worker routes each result to in-memory pending stream.
7. OpenAI handler decodes token IDs through Python bridge.
8. Rust returns SSE chunks or a final JSON response.

## ZMQ proxy internals

`ZmqModelWorkerProxy` binds three sockets:

- request `PushSocket`
- response `PullSocket`
- cancel `PushSocket`

It keeps an in-memory pending map keyed by `RequestID`, dispatches request
batches to request socket, and runs a response worker that:

- routes scheduler results to per-request channels,
- tracks first-token and end-to-end metrics,
- issues cancellation messages when clients disconnect.

Relevant code references:

- `max/serve-rust/src/zmq_interface.rs:69`
- `max/serve-rust/src/zmq_interface.rs:76`
- `max/serve-rust/src/zmq_interface.rs:82`
- `max/serve-rust/src/zmq_interface.rs:88`
- `max/serve-rust/src/zmq_interface.rs:140`
- `max/serve-rust/src/zmq_interface.rs:237`

## Important current limitation

In the current code, Rust generates fresh random IPC paths in
`main.rs` and does not launch the Python model worker process itself.
So Rust and Python workers only communicate if an external orchestrator starts
the worker side with matching endpoints.

This means `max/serve-rust` is currently a frontend/proxy layer and not a full
end-to-end process supervisor for model-worker lifecycle.

In other words, startup ownership is still Python-side, and Rust-side startup
is currently independent unless another orchestrator wires both with matching
ZMQ endpoints.

## Did this change in this work?

No worker-launch behavior was changed here.

What was added:

- More Rust tests:
  - env/settings parsing defaults, overrides, invalid fallback
  - request-id and IPC-path helpers
  - metrics snapshot counters/averages
  - OpenAI model-list response shape
  - KServe handler response behavior
  - AppError to HTTP response mapping
  - ZMQ round-trip with a fake worker (integration-style unit test)

No change was made to where or how MAX model workers are started.

## Current tests in `max/serve-rust`

- Config tests: `src/config.rs`
- Type helper tests: `src/types.rs`
- Metrics tests: `src/metrics.rs`
- OpenAI tests: `src/openai.rs`
- KServe tests: `src/kserve.rs`
- Error mapping tests: `src/error.rs`
- ZMQ proxy contract test: `src/zmq_interface.rs`

For local run in this environment:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test
```

## Rust-managed orchestration design

### Goal

Define how `max/serve-rust` can launch and supervise the Python MAX model worker
directly, instead of relying on external startup wiring.

### Current state (baseline)

- Rust frontend creates random ZMQ IPC endpoints at startup.
- Rust binds request/response/cancel sockets and serves HTTP.
- Python model worker startup is owned by Python entrypoints.
- There is no shared process supervisor in Rust today.

### Target state

When `max-serve-rust` starts, it should optionally:

1. Generate/choose ZMQ endpoints.
2. Start Python model worker process with those endpoints.
3. Wait for readiness handshake.
4. Serve traffic.
5. On shutdown or worker crash, propagate termination and cleanup.

### Proposed contract between Rust and Python worker

Startup contract:

- Rust passes these env vars to worker process:
  - `MAX_SERVE_ZMQ_REQUEST_ENDPOINT`
  - `MAX_SERVE_ZMQ_RESPONSE_ENDPOINT`
  - `MAX_SERVE_ZMQ_CANCEL_ENDPOINT`
  - `MAX_SERVE_WORKER_MODE=1`
- Python worker boot path should:
  - read endpoint env vars
  - connect worker sockets to Rust-bound endpoints
  - emit readiness signal only after scheduler and pipeline are live

Message contract:

- Request frame body (MessagePack): `TextGenerationContext<T>`
  - `request_id`
  - `request` (OpenAI/KServe parsed payload)
- Response frame body (MessagePack): `Map<RequestID, SchedulerResult<Reply>>`
  - `result` optional payload chunk
  - `is_done` completion marker
- Cancellation frame body (MessagePack): `Vec<RequestID>`

### Rust implementation plan

1. Add worker process manager module `src/worker_process.rs`.
   - Build `tokio::process::Command`.
   - Launch Python entrypoint in worker mode.
   - Set endpoint env vars.
   - Capture stdout/stderr for logs.
   - Expose `start(...)`, `wait_ready(timeout)`, `terminate_gracefully()`,
     `kill_if_needed()`.
2. Add readiness strategy.
   - Preferred: explicit worker-ready control channel over separate IPC
     endpoint.
   - Fallback: parse deterministic stdout token `MODEL_WORKER_READY`.
3. Add runtime mode in `main.rs`.
   - `MAX_SERVE_RUST_MANAGE_WORKER=1`
   - if off: current behavior
   - if on: Rust launches/supervises worker
4. Add shutdown and failure policy.
   - On server shutdown: terminate worker, wait grace period, then kill.
   - On worker crash: mark frontend unhealthy and stop serving or restart based
     on policy.
   - Surface worker liveness in `/health`.
5. Endpoint wiring source of truth in Rust.
   - Generate endpoint strings once.
   - Pass same values to proxy bind and worker env.

### Orchestration test plan

Unit tests:

- command construction includes executable/args/env
- readiness timeout path
- terminate-then-kill behavior

Integration tests:

1. Rust launches fake Python worker fixture.
2. Worker reads env endpoints and connects sockets.
3. Rust sends request.
4. Worker emits response and done marker.
5. Rust stream completes and metrics update.

### Migration path

1. Land process manager behind feature flag/env switch.
2. Keep external orchestration as default.
3. Run dual-path CI until stable.
4. Switch default to Rust-managed mode after soak.
