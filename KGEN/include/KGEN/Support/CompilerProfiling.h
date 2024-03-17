//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_COMPILERPROFILING_H
#define KGEN_SUPPORT_COMPILERPROFILING_H

#include "Support/Profiling/TimeProfiler.h"

namespace M::KGEN {

/// Profiler entry for Mojo compilation passes.
using CompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 1), Trace::kCompiler>;

/// Verbose profiler entry for Mojo compilation passes.
using VerboseCompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 2), Trace::kCompiler>;

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

} // namespace M::KGEN

#endif // KGEN_SUPPORT_COMPILERPROFILING_H
