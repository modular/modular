//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_GLOBALS_H
#define ASYNCRT_RUNTIME_GLOBALS_H

#include "Support/SymbolExport.h"

#include <atomic>
#include <functional>

namespace M::AsyncRT {
class Runtime;
class CompactRuntimePtr;

namespace Detail {
class RuntimeTable;
} // namespace Detail

class Globals {

public:
  /// This is a TLS CompactRuntimePtr pointing to the runtime on behalf of
  /// which the thread is processing work items. That thread may be a 'worker'
  /// thread of the runtime's work queue, or a 'main' thread which is also
  /// donating itself to processing work items for the runtime.
  ///
  /// NOTE: MSVC does not allow a thread_local to have DLL linkage, so we must
  /// hide this under a function.
  static MODULAR_CXX_EXPORT CompactRuntimePtr &getCurrentRuntimeInTLS();

  static MODULAR_CXX_EXPORT Detail::RuntimeTable &
  getRuntimeTableSingleton(const std::function<Detail::RuntimeTable *()> &ctor);

private:
  friend class AsyncValue;
  /// This is a global counter of the number of AsyncValue instances currently
  /// live in the process.  This is intended to be used for debugging only, and
  /// is only kept in sync if `isAllocationTrackingEnabled()` returns true.
  static MODULAR_CXX_EXPORT std::atomic<ssize_t> totalAllocatedAsyncValues;
};

} // namespace M::AsyncRT

#endif
