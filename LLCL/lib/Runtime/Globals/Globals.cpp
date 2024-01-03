//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Globals/Globals.h"
#include "LLCL/Runtime/AsyncValue.h"
#include "Support/SymbolExport.h"

#include <atomic>

using namespace M::LLCL;

[[maybe_unused]] MODULAR_CXX_EXPORT std::atomic<ssize_t>
    M::LLCL::AsyncValue::totalAllocatedAsyncValues{0};

MODULAR_CXX_EXPORT CompactRuntimePtr &
M::LLCL::Globals::getCurrentRuntimeInTLS() {
  static thread_local CompactRuntimePtr currentRuntimeInTLS;
  return currentRuntimeInTLS;
}

MODULAR_CXX_EXPORT Detail::TypeInfoTable &
M::LLCL::Globals::getTypeInfoTableSingleton(
    const std::function<Detail::TypeInfoTable *()> &ctor) {
  static Detail::TypeInfoTable *table = ctor();
  return *table;
}

MODULAR_CXX_EXPORT Detail::RuntimeTable &
M::LLCL::Globals::getRuntimeTableSingleton(
    const std::function<Detail::RuntimeTable *()> &ctor) {
  static Detail::RuntimeTable *table = ctor();
  return *table;
}
