//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_RUNTIME_GLOBALS_H
#define ASYNCRT_RUNTIME_GLOBALS_H

#include "Support/SymbolExport.h"

#include <functional>

namespace M::AsyncRT {
class Runtime;
class CompactRuntimePtr;

namespace Detail {
class RuntimeTable;
} // namespace Detail

} // namespace M::AsyncRT

namespace M::AsyncRT::Globals {

/// This is a TLS CompactRuntimePtr pointing to the runtime on behalf of
/// which the thread is processing work items. That thread may be a 'worker'
/// thread of the runtime's work queue, or a 'main' thread which is also
/// donating itself to processing work items for the runtime.
///
/// NOTE: MSVC does not allow a thread_local to have DLL linkage, so we must
/// hide this under a function.
extern MODULAR_CXX_EXPORT CompactRuntimePtr &getCurrentRuntimeInTLS();

extern MODULAR_CXX_EXPORT Detail::RuntimeTable &
getRuntimeTableSingleton(const std::function<Detail::RuntimeTable *()> &ctor);

} // namespace M::AsyncRT::Globals

#endif
