# Tracy and the libkineto profiler: when to use which

This guide is for code that uses MAX's Tracy zones today and is evaluating
the [new libkineto-backed profiler](profiling.md). The TL;DR: **keep Tracy
where it's already working — the new profiler is additive, not a
replacement**.

## Are they doing the same thing?

No, and that's why both stay supported:

| Aspect             | Tracy                                          | libkineto profiler                                |
|--------------------|------------------------------------------------|---------------------------------------------------|
| Build              | Requires `--config=tracy`                      | Default Linux x86_64 only                         |
| Wire format        | Tracy proprietary (`.tracy` capture)           | Chrome trace JSON                                 |
| Live viewer        | Yes — Tracy GUI streams in real time           | No — written on `stop()`                          |
| Post-hoc analyzer  | Tracy GUI                                      | HTA, Chrome `chrome://tracing`, Perfetto UI       |
| On-demand trigger  | No (developer tool)                            | Dynolog IPC (`KINETO_USE_DAEMON=1`)               |
| CPU overhead (off) | ~0% (compiled out)                             | ≤0.2% (one predicted branch per kernel)           |
| CPU overhead (on)  | Low (sub-microsecond per zone)                 | One libkineto user-correlation call per span      |
| GPU profiling      | Yes (Tracy CUDA / Tracy ROCm)                  | Yes (CUPTI / rocprofiler)                         |

**Use Tracy when:**

- You're doing interactive performance debugging in a dev loop.
- You need real-time visibility into a long-running session without
  collecting + opening a trace file.
- You're profiling Mojo code at fine granularity with `TRACY_VERBOSITY=2`.

**Use the libkineto profiler when:**

- You want a Chrome/HTA-compatible trace to analyze offline, including
  correlating across the ranks of a multi-GPU run.
- You need to start a capture from outside the process — for example, on an
  already-running server — using
  [Dynolog](https://github.com/facebookincubator/dynolog), an on-demand
  profiling daemon, with no flag flip or restart.
- You're collecting in production builds (no `--config=tracy` flag flip).

## Build-time mutual exclusion

Tracy GPU and libkineto are mutually exclusive at build time — both
subscribe to CUPTI's Activity Callback API, and letting both attach in the
same process double-subscribes the queue and yields garbled traces. The
Bazel `linux_x86_64_no_tracy` `config_setting` (Linux x86_64 AND
`--config=tracy` *not* set) is the only condition that links `@libkineto`
into `//Support:Profiling`:

- Default builds on Linux x86_64 link **libkineto** (no Tracy GPU).
- `--config=tracy` builds link **Tracy** (no libkineto, on any platform).
- macOS / Linux aarch64 default builds link **neither** GPU profiler.

Tracy **CPU** instrumentation is orthogonal and stays available in every
configuration. The runtime never hands off between the two GPU profilers —
there's nothing to hand off because at most one is linked into a given
binary. `session.profiling.start()` / `.stop()` are safe no-ops in
configurations that don't link libkineto.

Switching between the two profilers is a build-flag flip: drop or add
`--config=tracy` and rebuild.

## Code-level mapping

Existing Tracy instrumentation (C++):

```cpp
// Before — Tracy zone
#include "Support/Profiling/Tracy.h"
TRACY_ZONE_SCOPED_NC("kernel-launch", TRACY_COLOR_BLUE);
launch_my_kernel();
```

You do **not** need to rewrite this for libkineto coverage. libkineto's
CUPTI Callback API captures `cudaLaunchKernel` automatically — the actual
kernel will show up in the HTA trace without any new instrumentation. Keep
the Tracy zone if you want it for live debugging in dev.

If you want an explicit semantic span that appears in **both** Tracy and the
HTA trace, use the libkineto Range API alongside the Tracy zone:

```cpp
// After — both backends
#include "Support/Profiling/Range.h"
#include "Support/Profiling/Tracy.h"

{
  M::Profiling::RangeScope range("kernel-launch", TRACY_COLOR_BLUE);
  TRACY_ZONE_SCOPED_NC("kernel-launch", TRACY_COLOR_BLUE);
  launch_my_kernel();
}
```

A possible future `mo.profile.range` MLIR op from Mojo could be another
option — it would lower to the libkineto Range bridge and emit a Tracy zone in
the same build.

## Mojo-level mapping

Existing Mojo `Trace[level]` usage continues to work unchanged. The libkineto
profiler captures Mojo kernels automatically via CUPTI, so `Trace`-wrapped
code shows up in HTA without any change. The Mojo
`Range[category]` struct (#86343) is the new
equivalent if you want an explicit semantic span that mirrors `Trace`
semantically but flows into the HTA trace as a labeled CPU span. Both can
coexist in the same kernel.

## CLI / environment

Tracy environment variables (`TRACY_NO_INVARIANT_CHECK`, etc.) are unchanged
and remain only meaningful in `--config=tracy` builds.

New environment variables (always-on):

| Variable                                | Effect                                                                |
|-----------------------------------------|-----------------------------------------------------------------------|
| `MODULAR_MAX_DEBUG_PROFILING_ENABLED=1` | Enables profiling; overrides the `profiling_enabled` config value.    |
| `KINETO_USE_DAEMON=1`                   | Tells libkineto to attempt the Dynolog IPC handshake at startup.      |
| `KINETO_LOG_LEVEL=0`                    | libkineto verbose logging.                                            |

## What you do not need to migrate

- Tracy zones in existing C++ code (they continue to compile, they continue
  to fire when `--config=tracy` is set).
- Mojo `Trace[level]` usage in kernel code.
- Tracy CUDA / Tracy ROCm setup in your dev workflow.

## When to consider replacing Tracy entirely

Not in the near term. Tracy excels at live, sub-microsecond developer-loop
debugging, which libkineto does not target. The expected long-term shape is
Tracy in dev, libkineto in prod and on-demand capture, with both available in
dev builds for cross-checking.

If a piece of code is currently Tracy-only and you want it in HTA, the
cheapest path is to add a `RangeScope` next to the existing Tracy zone — not
to remove the Tracy zone.

## References

- [User guide](profiling.md)
- [Engineer reference](../../docs/internal/Profiling.md)
- [Tracy docs](https://github.com/wolfpld/tracy)
