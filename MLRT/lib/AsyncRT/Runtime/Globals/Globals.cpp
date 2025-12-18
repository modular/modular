//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// NOTE: We use the legacy tcmalloc on macOS only because the modern tcmalloc
// doesn't support it
#if defined(__APPLE__)
#include <gperftools/tcmalloc.h>
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wprivate-header"
#include <tcmalloc/tcmalloc.h>
#pragma GCC diagnostic pop
#endif

#include "MLRT/AsyncRT/Runtime/CompactRuntimePtr.h"
#include "MLRT/AsyncRT/Runtime/Globals/Globals.h"
#include "Support/BinaryID.h"

#include <atomic>
#include <cstdio>

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

MODULAR_CXX_EXPORT void *TCMallocGlobals::tc_new(size_t alignment,
                                                 size_t size) {
#if defined(__APPLE__)
  return ::tc_new_aligned(size, std::align_val_t(alignment));
#else
  return TCMallocInternalNewAligned(size, std::align_val_t(alignment));
#endif
}
MODULAR_CXX_EXPORT void TCMallocGlobals::tc_delete(void *ptr) {
#if defined(__APPLE__)
  return ::tc_delete(ptr);
#else
  return TCMallocInternalDelete(ptr);
#endif
}

MODULAR_CXX_EXPORT std::string M::AsyncRT::getRuntimeGlobalsBinaryID() {
  // M::getBinaryID() returns the binary ID of the shared library that contains
  // it. For the purposes of MEF cache invalidation, we need to know when
  // there's been a change in these shared libraries.
  return M::getBinaryID();
}

/// Global counter for assigning unique task IDs across all AsyncRT users.
/// Must live in a shared library to ensure a single instance.
static std::atomic<uint64_t> globalUniqueTaskIdCounter{0};

MODULAR_CXX_EXPORT uint64_t M::AsyncRT::getUniqueTaskIdForWorkItem() {
  uint64_t id =
      globalUniqueTaskIdCounter.fetch_add(1, std::memory_order_relaxed);
  return id;
}
