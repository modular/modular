//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/OutputChain.h"
#include "KGEN/CompilerRT/Registration.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/SymbolExport.h"

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr, size_t nameLen,
                                       const char *detailPtr,
                                       size_t detailLen) {
  // NOTE: Must be always enabled.
  ProfilerEntry<true>::createAndPush(StringRef(namePtr, nameLen),
                                     StringRef(detailPtr, detailLen));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerEnd() {
  // NOTE: Must be always enabled.
  ProfilerEntry<true>::endAndPop();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT size_t
KGEN_CompilerRT_TimeTraceProfilerCurrentId() {
  // NOTE: Must be always enabled.
  return ProfilerEntry<true>::currentId();
}

void M::KGEN::registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerBegin",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerBegin});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerEnd",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerEnd});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerCurrentId",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerCurrentId});
}
