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

| Method / property  | Effect                                                                                                                                                          |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start()`          | Enable the profiler. Idempotent — calling while enabled is a no-op. (Overall no-op until recording lands.)                                                      |
| `stop()`           | Flush and serialize the trace. Idempotent.                                                                                                                      |
| `wait_for_trace()` | Block until the most recent `stop()` finishes writing. Raises an exception on serialization failure (a dedicated `ProfilingError` type is a follow-up nightly). |
| `state`            | One of `"idle"`, `"warmup"`, `"active"`, `"flushing"`.                                                                                                          |
| `is_enabled`       | `True` while enabled. Cheap; use to elide expensive trace-name construction on the hot path.                                                                    |

## Configuration

Profiler configuration lives on the `ProfilingConfig` model used by
`max serve` and the pipeline configs. Set values in your `MAXConfig` file or
via the matching environment variables:

| Setting                            | Default | Meaning                                                                                         |
|------------------------------------|---------|-------------------------------------------------------------------------------------------------|
| `profiling_enabled`                | `False` | Master switch. Can also be set via `MODULAR_MAX_DEBUG_PROFILING_ENABLED=1`.                     |
| `profiling_output_path`            | `None`  | Output file path. Template-variable and directory forms are a follow-up nightly.                |
| `profiling_dynolog_enabled`        | `True`  | Will let the process listen for Dynolog on-demand-profile requests once fleet collection lands. |
| `profiling_warmup_steps`           | `0`     | Iterations to skip after `start()` before recording.                                            |
| `profiling_active_steps`           | `10`    | Iterations to record.                                                                           |
| `profiling_periodic_flush_seconds` | `60`    | Crash-safe chunk cadence for long-running serving.                                              |

> **Note**: A direct `session.debug.profiling_*` setter surface is a
> follow-up nightly. Until it lands, configure these knobs via
> `ProfilingConfig` before constructing the session:
>
> ```python
> from max.pipelines.lib.config import ProfilingConfig
> ```

## Coexistence with other profilers

- `session.gpu_profiling()` (NVTX/Nsight) is orthogonal. NVTX markers
  continue to feed Nsight Systems independently.
- Tracy (`--config=tracy`) is mutually exclusive with the libkineto profiler
  at build time. `--config=tracy` builds do not link libkineto, so
  `session.profiling.start()` is a no-op there; default builds do not link
  Tracy GPU. Tracy CPU instrumentation is orthogonal and available in
  both configurations.

## See also

- [GPU profiling with Nsight Systems](https://docs.modular.com/max/gpu-system-profiling)
