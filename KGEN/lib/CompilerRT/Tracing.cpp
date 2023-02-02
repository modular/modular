//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "Support/SymbolExport.h"
#include "Support/TimeProfiler.h"

using namespace M;

MODULAR_EXPORT void
KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr, size_t nameLen,
                                       const char *detailPtr,
                                       size_t detailLen) {
  Detail::timeTraceProfilerBeginImpl(std::string{namePtr, nameLen}, [=]() {
    return std::string{detailPtr, detailLen};
  });
}

MODULAR_EXPORT void KGEN_CompilerRT_TimeTraceProfilerEnd() {
  Detail::timeTraceProfilerEndImpl();
}

void M::KGEN::registerTracing(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerBegin",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerBegin});
  funcs.push_back({"KGEN_CompilerRT_TimeTraceProfilerEnd",
                   (void *)&KGEN_CompilerRT_TimeTraceProfilerEnd});
}
