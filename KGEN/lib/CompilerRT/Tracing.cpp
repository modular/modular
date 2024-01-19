//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/MojoCallContext.h"
#include "KGEN/CompilerRT/Registration.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/SymbolExport.h"

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr, size_t nameLen,
                                       const char *detailPtr, size_t detailLen,
                                       size_t parentId) {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true>::createWithParent(parentId,
                                               StringRef(namePtr, nameLen),
                                               StringRef(detailPtr, detailLen))
      .getId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerEnd(size_t id) {
  // NOTE: Must be always enabled.
  ProfilerEntry<true>(id).record();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerGetCurrentId() {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true>::getCurrentId();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerSetCurrentId(size_t id) {
  // NOTE: Must be always enabled.
  ProfilerEntry<true>(id).setAsCurrentId();
}

void M::KGEN::registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerBegin",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerBegin});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerEnd",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerEnd});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerGetCurrentId",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerGetCurrentId});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerSetCurrentId",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerSetCurrentId});
}
