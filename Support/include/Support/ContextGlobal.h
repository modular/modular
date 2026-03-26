//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Global current M::Context pointer. Implemented in the Globals shared library
// so there is a single definition per process. The public API lives in
// Support/Context.h; this header declares the internal pointer-based accessors.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONTEXT_GLOBAL_H
#define SUPPORT_CONTEXT_GLOBAL_H

#include <mutex>

#include "Support/SymbolExport.h"

namespace M {

class Context;

/// Mutex serializing access to the global context pointer and related TLS
/// coordination. Defined in the Globals shared library (single definition per
/// process).
MODULAR_CXX_EXPORT std::mutex &getGlobalContextMutex();

/// Returns the current global context pointer, or nullptr if none set.
MODULAR_CXX_EXPORT Context *getCurrentMaxContextPointerOrNull();

/// Sets the global context pointer.
MODULAR_CXX_EXPORT void setCurrentMaxContextPointer(Context *ptr);

/// If the global context pointer equals \p ptr, clears it. Called from
/// Context::~Context() so the global is cleared when the last ref is destroyed.
MODULAR_CXX_EXPORT void clearGlobalContextPointerIfEquals(Context *ptr);

} // namespace M

#endif // SUPPORT_CONTEXT_GLOBAL_H
