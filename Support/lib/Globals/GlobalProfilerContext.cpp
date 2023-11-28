//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Globals/GlobalProfilerContext.h"
#include "Support/SymbolExport.h"
#include <atomic>

static std::atomic<M::ProfilingDetail::GlobalProfilerContext *>
    globalProfilerContextInstance = nullptr;

MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
M::Globals::getGlobalProfilerContext() {
  return globalProfilerContextInstance.load();
}

MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
M::Globals::exchangeGlobalProfilerContext(
    M::ProfilingDetail::GlobalProfilerContext *ctx) {
  return globalProfilerContextInstance.exchange(ctx);
}
