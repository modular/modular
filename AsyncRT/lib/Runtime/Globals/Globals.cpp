//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Globals/Globals.h"
#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "Support/SymbolExport.h"

#include <atomic>

using namespace M::AsyncRT;

[[maybe_unused]] MODULAR_CXX_EXPORT std::atomic<ssize_t>
    M::AsyncRT::Globals::totalAllocatedAsyncValues{0};

MODULAR_CXX_EXPORT CompactRuntimePtr &
M::AsyncRT::Globals::getCurrentRuntimeInTLS() {
  static thread_local CompactRuntimePtr currentRuntimeInTLS;
  return currentRuntimeInTLS;
}

MODULAR_CXX_EXPORT Detail::RuntimeTable &
M::AsyncRT::Globals::getRuntimeTableSingleton(
    const std::function<Detail::RuntimeTable *()> &ctor) {
  static Detail::RuntimeTable *table = ctor();
  return *table;
}
