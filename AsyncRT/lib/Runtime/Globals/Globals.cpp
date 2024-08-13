//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifdef USE_TCMALLOC
#include <gperftools/tcmalloc.h>
#endif

#include "AsyncRT/Runtime/CompactRuntimePtr.h"
#include "AsyncRT/Runtime/Globals/Globals.h"
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

#ifdef USE_TCMALLOC
MODULAR_CXX_EXPORT void *TCMallocGlobals::tc_new(size_t size,
                                                 size_t alignment) {
  return ::tc_new_aligned(size, std::align_val_t(alignment));
}
MODULAR_CXX_EXPORT void TCMallocGlobals::tc_delete(void *ptr) {
  return ::tc_delete(ptr);
}
#endif
