//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "Support/SymbolExport.h"
#include "Support/TimeProfiler.h"

using namespace M;

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr, size_t nameLen,
                                       const char *detailPtr,
                                       size_t detailLen) {
  timeTraceProfilerBegin(StringRef(namePtr, nameLen),
                         StringRef(detailPtr, detailLen));
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerEnd() {
  timeTraceProfilerEnd();
}

void M::KGEN::registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerBegin",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerBegin});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerEnd",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerEnd});
}
