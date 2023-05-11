//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Globals/GlobalProfilerContext.h"
#include "Support/SymbolExport.h"
#include <atomic>

static std::atomic<M::GlobalProfilerContext *> globalProfilerContextInstance =
    nullptr;

MODULAR_CXX_EXPORT M::GlobalProfilerContext *
M::Globals::getGlobalProfilerContext() {
  return globalProfilerContextInstance.load();
}

MODULAR_CXX_EXPORT void
M::Globals::setGlobalProfilerContext(M::GlobalProfilerContext *ctx) {
  globalProfilerContextInstance.store(ctx);
}

MODULAR_CXX_EXPORT M::GlobalProfilerContext *
M::Globals::exchangeGlobalProfilerContext(M::GlobalProfilerContext *ctx) {
  return globalProfilerContextInstance.exchange(ctx);
}
