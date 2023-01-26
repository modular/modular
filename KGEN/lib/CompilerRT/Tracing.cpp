//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/TimeProfiler.h"

using namespace M;

extern "C" void KGEN_CompilerRT_TimeTraceProfilerBegin(const char *namePtr,
                                                       size_t nameLen,
                                                       const char *detailPtr,
                                                       size_t detailLen) {
  Detail::timeTraceProfilerBeginImpl(std::string{namePtr, nameLen}, [=]() {
    return std::string{detailPtr, detailLen};
  });
}

extern "C" void KGEN_CompilerRT_TimeTraceProfilerEnd() {
  Detail::timeTraceProfilerEndImpl();
}
