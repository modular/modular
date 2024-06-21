//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This profiler works by overriding the system malloc, which is functionality
// provided interally by mimalloc.
//
// Example usage on Linux:
//
// 1. cmake-modular-release -D MODULAR_LLCL_MAX_PROFILING_LEVEL=1 -D
// MODULAR_MALLOC_PROFILER=1 && build
// 2. Set `TRACE_IN_REAL_TIME = 1` in TimeProfiler.h.
// 3. Make sure libmimalloc.so is preloaded so that overriding malloc works from
//    the Python API:
//    ```
//    export
//    LD_PRELOAD=$MODULAR_DERIVED_PATH/_deps/_deps/mimalloc-build/libmimalloc.so
//    ```
// 4. Use the Modular Python API with profiling enabled:
//    ```
//    MODULAR_PROFILE_FILENAME=<profile> $MODULAR_PYTHON run_modular_model.py
//    ```
//    Where `run_model_model.py` is a Python script that uses the
//    `max.engine` API.
//
// The above steps print out memory statistics to `llvm::dbgs()` alongside
// instrumented profiler event dumps.
//
// TODO(#28519): Integrate with `ProfilingAllocator`.
// TODO(#28520): Merge into `CompletedEntry` and write to final time trace.
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROFILING_MALLOCPROFILER_H
#define SUPPORT_PROFILING_MALLOCPROFILER_H

#include "Support/LLVMForwardDecls.h"
#include "Support/Profiling/TimeProfiler.h"
#include <string>

namespace M {

#if MODULAR_MALLOC_PROFILER
/// Collects memory statistics into the returned human readable string.
/// Examples of statistics collected:
/// - Committed and reserved memory collected by overriding malloc.
/// - Peak RSS collected from a syscall.
/// Collects multiple statistic including peak, total, and current usage for
/// each metric.
std::string memoryStatistics();
#else
inline std::string memoryStatistics() { return ""; }
#endif // MODULAR_MALLOC_PROFILER

/// Profiler entry for general memory profiling by overriding the system malloc.
using MallocProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

/// Opens a `TimeTraceScope` that also collects and dumps memory statistics.
inline auto mallocTraceScope(StringRef name) {
  return TimeTraceScope(
      MallocProfilerEntry::create(name, ProfilingDetail::Label::kNoIntPayload,
                                  memoryStatistics),
      memoryStatistics);
}

} // namespace M

#endif // SUPPORT_PROFILING_MALLOCPROFILER_H
