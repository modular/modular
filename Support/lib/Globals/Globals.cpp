//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Globals/Globals.h"
#include "Support/SymbolExport.h"

#include "llvm/ADT/StringMap.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

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

MODULAR_CXX_EXPORT M::Detail::TypeInfoTable &
M::Globals::getTypeInfoTableSingleton(
    const std::function<Detail::TypeInfoTable *()> &ctor) {
  static Detail::TypeInfoTable *table = ctor();
  return *table;
}

MODULAR_CXX_EXPORT std::mutex &M::Globals::getConfigOverridesMutex() {
  static std::mutex m;
  return m;
}

MODULAR_CXX_EXPORT llvm::StringMap<std::string> &
M::Globals::getConfigOverrides() {
  static llvm::StringMap<std::string> m;
  return m;
}
