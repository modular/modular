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

## Control surface

The `session.profiling` namespace exposes the runtime lifecycle:

| Method / property  | Effect                                                                                                                                                        |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start()`          | Enable libkineto and subscribe to CUPTI. Idempotent — calling while enabled is a no-op.                                                                       |
| `stop()`           | Flush and serialize the trace. Idempotent. Never raises on serialization failure — the error is recorded and surfaced by `wait_for_trace()`.                  |
| `wait_for_trace()` | Block until the most recent `stop()` finishes writing. Raises `max.engine.ProfilingError` on serialization failure (see [Error reporting](#error-reporting)). |
| `state`            | One of `"idle"`, `"active"`, `"flushing"` (plus `"warmup"` once the step-window state machine lands — see below).                                             |
| `is_enabled`       | `True` while enabled. Cheap; use to elide expensive trace-name construction on the hot path.                                                                  |

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
