//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_COMPILERPROFILING_H
#define KGEN_SUPPORT_COMPILERPROFILING_H

#include "LLCL/Support/Profiling.h"

namespace M::KGEN {

/// Profiler entry for Mojo compilation passes.
using CompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 1)>;

/// Verbose profiler entry for Mojo compilation passes.
using VerboseCompilerProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kCompiler, 2)>;

struct CompilerTimeTraceScope
    : public TimeTraceScope<CompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  CompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(CompilerProfilerEntry::create(name, detail)) {}
  CompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(CompilerProfilerEntry::create(name, detailFn)) {}
};

struct VerboseCompilerTimeTraceScope
    : public TimeTraceScope<VerboseCompilerProfilerEntry::isEnabled()> {
  using TimeTraceScope::TimeTraceScope;

  VerboseCompilerTimeTraceScope(StringRef name, StringRef detail = {})
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detail)) {}
  VerboseCompilerTimeTraceScope(StringRef name, ProfilerPrintFn detailFn)
      : TimeTraceScope(VerboseCompilerProfilerEntry::create(name, detailFn)) {}
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_COMPILERPROFILING_H
