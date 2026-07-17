# MAX profiler (HTA-compatible)

MAX has an on-demand profiler that emits
[Chrome trace JSON](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
compatible with Meta's
[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
(HTA). It is built on
[`libkineto`](https://github.com/pytorch/kineto/tree/main/libkineto); the
overhead when disabled is ≤0.2% (one predicted branch per kernel launch).

> **Build gate.** Recording requires the libkineto backend, which is opt-in:
> build with `--config=kineto` (Linux x86_64 only, #91288). Default builds —
> including the shipped wheel and conda packages — do not link libkineto, so
> the control surface below is callable but records nothing there. The
> backend links on-demand dlopen stubs for libcuda/libcupti/libcudart that
> the packages do not ship, which is why it cannot be on by default yet.

## Quick start

```python
from max.driver import Accelerator
from max.engine import InferenceSession

session = InferenceSession(devices=[Accelerator()])
model = session.load(my_graph)

session.profiling.start()
for batch in data:
    model.execute(batch)
session.profiling.stop()
session.profiling.wait_for_trace()
```

When `profiling_enabled` is set, the pipeline entrypoints (`max serve`,
`max pipelines generate`, `LLM`) enable the profiler during
`PipelineConfig.configure_session()`, so a capture can also begin without an
explicit `start()`. Constructing a bare `InferenceSession` does not
auto-enable.

The default output path is `/tmp/max-trace.json`. See
[Configuration](#configuration) below for changing it. Open the trace file
in the Chrome trace viewer (`chrome://tracing` or
[Perfetto UI](https://ui.perfetto.dev/)) or import it into HTA:

```python
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="/tmp")
print(analyzer.get_temporal_breakdown())
```

## CLI invocation

`max pipelines generate` (and `max pipelines benchmark`) honor a
`--profiling-enabled` flag that arms libkineto automatically inside
`configure_session()` — before model load and therefore before MLRT
instantiates any CUDA graph. No Python wrapper is required:

```bash
max pipelines generate --model modularai/Llama-3.1-8B-Instruct-GGUF \
    --prompt "Hello, world!" --max-new-tokens 64 --profiling-enabled
```

Arming at session-construction time captures the whole run, including model
load and compilation. It is no longer *required* for CUDA-Graph workloads:
CUPTI in CUDA 13.1 captures kernels launched from graph execs that were
instantiated before profiling was armed, so a mid-run
`session.profiling.start()` (or a Dynolog on-demand request) records
graph-replay kernels too — verified on B200. An `atexit` hook flushes the
trace on interpreter shutdown for callers that never call
`session.profiling.stop()` explicitly.

### Dynolog on-demand capture (no flags required)

Any MAX process launched with `KINETO_USE_DAEMON=1` in its environment
registers with a running [Dynolog](https://github.com/facebookincubator/dynolog)
daemon at device initialization — `--profiling-enabled` is **not** required
and the process holds no trace until asked. A single CLI invocation then
captures any registered PID on demand:

```bash
dyno gputrace --pids <pid> --duration-ms 3000 --log-file /tmp/trace.json
```

libkineto writes the trace to the daemon-supplied path (suffixed with the
PID), independent of `session.debug.profiling_output_path`. Set
`session.debug.profiling_dynolog_enabled = False` (or
`MODULAR_MAX_DEBUG_PROFILING_DYNOLOG_ENABLED=0`) to keep a daemon-mode
process unregistered.

### Enabling via `MODULAR_PROFILING_ENABLED`

For indirect launches (CI runners, wrapper scripts, `max serve` under
systemd), set the env var instead of the flag:

```bash
MODULAR_PROFILING_ENABLED=1 max pipelines generate ...
```

`1`/`true`/`yes`/`on` enable; `0`/`false`/`no`/`off` disable an explicit
`profiling_enabled=True`.

**Caveat for `bazelw run`.** Bazel sandboxes child processes and does
**not** inherit arbitrary environment variables, so the env var above is
filtered out when running via `./bazelw run`. Use the flag instead, or
whitelist the env var with `--action_env`:

```bash
# ❌ Silently dropped — env var does not reach the sandbox.
MODULAR_PROFILING_ENABLED=1 ./bazelw run //max/python/max/entrypoints:pipelines \
    -- generate ...

# ✅ Pass the flag through bazel run.
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
    ... --profiling-enabled

# ✅ Or whitelist the env var on the bazel command line.
./bazelw run --action_env=MODULAR_PROFILING_ENABLED \
    //max/python/max/entrypoints:pipelines -- generate ...
```

Direct invocations of the installed `max` binary (outside bazel) propagate
the env var normally.

## Control surface

The `session.profiling` namespace exposes the runtime lifecycle:

| Method / property  | Effect                                                                                                                                                                 |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start()`          | Enable the profiler. Idempotent — calling while enabled is a no-op.                                                                                                    |
| `stop()`           | Flush and serialize the trace. Idempotent. Never raises on serialization failure — the error is recorded and surfaced by `wait_for_trace()`.                           |
| `wait_for_trace()` | Block until the most recent `stop()` finishes writing. Raises `max.engine.ProfilingError` on serialization failure (see [Error reporting](#error-reporting)).          |
| `state`            | One of `"idle"`, `"warmup"`, `"active"`, `"flushing"` (`"warmup"` is reserved for the step-window state machine — see the note under [Configuration](#configuration)). |
| `is_enabled`       | `True` while enabled via this API. Stays `False` during Dynolog on-demand traces — gate hot-path annotation on `is_recording` instead.                                 |
| `is_recording`     | `True` while a trace of either origin (session API or Dynolog on-demand) is live and `range()` spans record. Cheap; the right hot-path gate.                           |
| `range(name)`      | Context manager annotating a named CPU span; see below.                                                                                                                |

### Semantic ranges

The runtime records every kernel launch automatically; `range()` is for
marking *application-level* phases on top of that:

```python
session.profiling.start()
with session.profiling.range("prefill"):
    model.execute(input_data)
session.profiling.stop()
```

The span appears as a `user_annotation` bar above the kernel timeline in
Perfetto/HTA, and the GPU kernels launched inside it are correlated to it.
Ranges nest. When no trace is live the underlying calls reduce to a single
predicted branch in the runtime, so annotations are safe to leave in
production code. Ranges also record during Dynolog-initiated on-demand
traces — no `start()` required in the annotated process.

## Configuration

Every knob is exposed three ways — pick whichever fits your workflow. For a
knob named `profiling_foo`:

- **Python property**: `session.debug.profiling_foo = value` on a live
  `InferenceSession`.
- **Environment variable**: `MODULAR_MAX_DEBUG_PROFILING_FOO=value` — the
  standard auto-mapping of the `max-debug.profiling-foo` Config key.
- **`modular.cfg` key**: `profiling-foo = value` in the `[max-debug]`
  section.

The `ProfilingConfig` pydantic model used by `max serve` and the pipeline
configs mirrors the same six fields.

| Setting                            | Default | Meaning                                                                                                                                                                             |
|------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `profiling_enabled`                | `False` | Master switch.                                                                                                                                                                      |
| `profiling_output_path`            | `None`  | Output file path; falls back to `/tmp/max-trace.json` when unset. Supports `{pid}` / `{rank}` templates and a directory form — see [Output path expansion](#output-path-expansion). |
| `profiling_dynolog_enabled`        | `True`  | Will let the process listen for Dynolog on-demand-profile requests once fleet collection lands.                                                                                     |
| `profiling_warmup_steps`           | `0`     | Reserved for the step-window state machine (not yet wired — see below).                                                                                                             |
| `profiling_active_steps`           | `10`    | Reserved for the step-window state machine (not yet wired — see below).                                                                                                             |
| `profiling_periodic_flush_seconds` | `60`    | Crash-safe chunk cadence for long-running serving (wired in a follow-up).                                                                                                           |

> **Note**: the warmup/active step windows are not wired yet — today a trace
> covers everything between `start()` (or auto-enable) and `stop()`,
> regardless of `profiling_warmup_steps` / `profiling_active_steps`, and
> `state` reports `"active"` immediately after `start()`. The knobs exist so
> configurations written now keep working when the step-window state machine
> lands.

## Output path expansion

`profiling_output_path` accepts three forms; the expansion applies whenever
the profiler writes a trace:

1. **Literal file path** (`/tmp/my-trace.json`): used as-is; the parent
   directory is created if missing. When the setting is unset, the profiler
   writes `/tmp/max-trace.json`.
2. **Template** (`/tmp/trace-{rank}-{pid}.json`): `{pid}` expands to the
   process ID and `{rank}` to the process rank, so multi-process and
   multi-rank runs don't collide on a single fixed filename.
3. **Directory** (`/tmp/traces/`): if the configured path names an existing
   directory, each capture writes
   `trace_rank<rank>_<pid>_<unix-ts>_<seq>.json` inside it, where `<seq>` is
   a per-process counter that keeps repeated captures in the same process
   from overwriting each other.

`{rank}` resolves to the first of the `MODULAR_RANK` or
`OMPI_COMM_WORLD_RANK` environment variables that is set to a plain integer,
falling back to `"0"` (a set-but-non-numeric value is treated as unset).
The rank comes from whatever launcher spawned the process: OpenMPI's
`mpirun` sets `OMPI_COMM_WORLD_RANK` automatically, while launchers that
don't use MPI (for example, Kubernetes-managed pods) should set
`MODULAR_RANK` explicitly — otherwise every process resolves to rank `0`
and only `{pid}` keeps their outputs distinct.

Directory detection happens on the literal setting, before template
expansion: a path like `/tmp/traces-{rank}/` takes the template branch even
if `/tmp/traces-0/` exists on disk.

## Error reporting

`stop()` never raises. If the trace could not be serialized — most commonly
an unwritable `profiling_output_path` or libkineto failing to flush its
in-memory buffer — the error is recorded and `wait_for_trace()` raises
`max.engine.ProfilingError`. For write failures the exception message
includes the resolved output path, so the failure can be diagnosed without
rerunning the workload. The recorded error is cleared at the next
`start()`, so a stale failure never leaks into a new capture session.

## Fork safety

Host applications that embed `InferenceSession` may fork after profiling is
enabled — Python `multiprocessing` with the `fork` start method, pre-fork
servers such as gunicorn, or a bare `os.fork()`. This is safe: the child
process starts with the profiler **disabled** (the parent's CUPTI
subscriptions and libkineto state are not valid in the child), and the
parent retains its enabled state across the fork. Call `start()` in the
child if you want profiling there too.

Do not call `fork()` while a `start()` / `stop()` is in flight on another
thread (or from inside a signal handler interrupting one): the profiler
cannot snapshot a mid-transition state across a fork and aborts the process
with a diagnostic rather than deadlocking the child.

MAX Serve itself launches its workers with the `spawn` start method, so it
is unaffected by any of this.

## Coexistence with other profilers

- `session.gpu_profiling()` (NVTX/Nsight) is orthogonal. NVTX markers
  continue to feed Nsight Systems independently.
- libkineto only links in `--config=kineto` builds (Linux x86_64 only);
  default builds do not link it, and `start()` / `stop()` are safe no-ops
  there.

## Troubleshooting

**Empty trace**. Confirm the build links libkineto (`--config=kineto` on
Linux x86_64 — default builds record nothing), that `is_enabled` was `True`
while the workload ran, and that a CUDA device existed before the capture
(the CUPTI subscription activates once a CUDA primary context is bound).

> **Note**: CUPTI-driven kernel symbolication and file rotation (500 MB cap,
> periodic flush every `profiling_periodic_flush_seconds`) are wired up in a
> follow-up nightly. Until that lands, traces from long-running captures
> are not chunked, and Mojo kernel names may appear mangled
> (`__mojo_…`) in the trace.

## See also

- [GPU profiling with Nsight Systems](https://docs.modular.com/max/gpu-system-profiling)
