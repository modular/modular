//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_COMPILERPROFILING_H
#define KGEN_SUPPORT_COMPILERPROFILING_H

#include "Support/Profiling/TimeProfiler.h"
#include <filesystem>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// TimeTraceScope
//===----------------------------------------------------------------------===//

constexpr bool kIsTracingEnabled = Trace::EnableTrace(Trace::kCompiler, 1);

using InterpreterProfilerEntry =
    ProfilerEntry<kIsTracingEnabled, Trace::kCompiler>;

/// Profiler entry for Mojo compilation passes.
using CompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 2), Trace::kCompiler>;

/// Verbose profiler entry for Mojo compilation passes.
using VerboseCompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 3), Trace::kCompiler>;

struct InterpreterTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            InterpreterProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  InterpreterTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(InterpreterProfilerEntry::create(name, detail)) {}
  InterpreterTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(InterpreterProfilerEntry::create(name, detailFn)) {}
};

struct CompilerTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            CompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  CompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(CompilerProfilerEntry::create(name, detail)) {}
  CompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(CompilerProfilerEntry::create(name, detailFn)) {}
};

struct VerboseCompilerTimeTraceScope
    : public TimeTraceScope<Trace::kCompiler,
                            VerboseCompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  VerboseCompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detail)) {}
  VerboseCompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detailFn)) {}
};

//===----------------------------------------------------------------------===//
// TraceProfiler
//===----------------------------------------------------------------------===//

/// Common trace profiler setup.
struct TraceProfiler {
  TraceProfiler(bool enabled, int timeTraceGranularity) {
    if (enabled)
      initialize(timeTraceGranularity);
  }
  ~TraceProfiler();

private:
  void initialize(int timeTraceGranularity);

  std::optional<TimeTraceProfiler> profiler;
  std::filesystem::path outputFilePath;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_COMPILERPROFILING_H
