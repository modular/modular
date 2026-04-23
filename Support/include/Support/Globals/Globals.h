//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_GLOBALS_GLOBALS_H
#define SUPPORT_GLOBALS_GLOBALS_H

#include "Support/SymbolExport.h"

#include "llvm/ADT/StringMap.h"

#include <functional>
#include <mutex>
#include <string>

namespace M {
namespace Detail {
class TypeInfoTable;
}

namespace ProfilingDetail {
struct GlobalProfilerContext;
}

namespace Globals {

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
getGlobalProfilerContext();

extern MODULAR_CXX_EXPORT M::ProfilingDetail::GlobalProfilerContext *
exchangeGlobalProfilerContext(M::ProfilingDetail::GlobalProfilerContext *ctx);

extern MODULAR_CXX_EXPORT Detail::TypeInfoTable &
getTypeInfoTableSingleton(const std::function<Detail::TypeInfoTable *()> &ctor);

// Process-wide storage for `Config::setGlobalValue()` overrides. These
// live in `libMSupportGlobals.so` so that writes from one shared library
// (e.g. the `max._core` Python extension that wraps `DebugConfig`) are
// visible to reads from another (e.g. `libmax.so` which contains
// `GraphCompiler/FrameworkFrontend`). A function-local static in
// `Support/lib/Configuration.cpp` would give each consumer a distinct
// copy, breaking cross-library propagation.
//
// Callers must hold `getConfigOverridesMutex()` while reading or writing
// `getConfigOverrides()`.
extern MODULAR_CXX_EXPORT std::mutex &getConfigOverridesMutex();
extern MODULAR_CXX_EXPORT llvm::StringMap<std::string> &getConfigOverrides();

} // namespace Globals

} // namespace M

#endif // SUPPORT_GLOBALS_GLOBALS_H
