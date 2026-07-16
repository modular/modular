# MAX profiler (HTA/Dynolog-compatible)

> **Preview.** This documents the `session.profiling` API surface and its
> configuration, which are available now. The profiler does **not record
> yet** — the libkineto-backed trace capture, Dynolog fleet collection,
> multi-rank captures, and named ranges arrive in later nightlies. Calling
> `start()` / `stop()` today is a safe no-op. This page grows as each piece
> lands.

MAX is gaining an on-demand profiler that will emit
[Chrome trace JSON](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
compatible with Meta's
[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)
(HTA). It is built on
[`libkineto`](https://github.com/pytorch/kineto/tree/main/libkineto); the
off-cost when disabled is ≤0.2% (one predicted branch per kernel launch).

## Control surface

The `session.profiling` namespace exposes the runtime lifecycle. The surface
is final and callable today, but it does not capture a trace until the
libkineto recording path lands in a later nightly:

| Method / property  | Effect                                                                                                                                                        |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start()`          | Enable the profiler. Idempotent — calling while enabled is a no-op. (Overall no-op until recording lands.)                                                    |
| `stop()`           | Flush and serialize the trace. Idempotent. Never raises on serialization failure — the error is recorded and surfaced by `wait_for_trace()`.                  |
| `wait_for_trace()` | Block until the most recent `stop()` finishes writing. Raises `max.engine.ProfilingError` on serialization failure (see [Error reporting](#error-reporting)). |
| `state`            | One of `"idle"`, `"warmup"`, `"active"`, `"flushing"`.                                                                                                        |
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
| `profiling_warmup_steps`           | `0`     | Iterations to skip after `start()` before recording.                                                                                                                                |
| `profiling_active_steps`           | `10`    | Iterations to record.                                                                                                                                                               |
| `profiling_periodic_flush_seconds` | `60`    | Crash-safe chunk cadence for long-running serving.                                                                                                                                  |

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

## See also

- [GPU profiling with Nsight Systems](https://docs.modular.com/max/gpu-system-profiling)
