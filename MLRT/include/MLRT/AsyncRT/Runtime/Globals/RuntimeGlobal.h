//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Global Runtime pointer and mutex. Implemented in the RuntimeGlobals shared
// library so there is a single definition per process (ODR). M::MLRT
// uses these (e.g. MLRT::getOrCreateRuntime) to manage the single global
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
#define MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H

#include "Support/SymbolExport.h"

#include <mutex>

namespace M::MLRT {

class Runtime;
struct RuntimeOptions;

/// Returns the mutex that protects the global runtime pointer.
MODULAR_CXX_EXPORT std::mutex &getGlobalRuntimeMutex();

/// Returns the current global runtime pointer, or nullptr if none set.
/// Caller must hold getGlobalRuntimeMutex().
MODULAR_CXX_EXPORT Runtime *getGlobalRuntimePointer();

/// Sets the global runtime pointer. Caller must hold getGlobalRuntimeMutex().
MODULAR_CXX_EXPORT void setGlobalRuntimePointer(Runtime *ptr);

/// If the global runtime pointer equals \p ptr, clears it. Called from
/// Runtime::~Runtime() when the runtime is destroyed.
MODULAR_CXX_EXPORT void clearGlobalRuntimePointerIfEquals(Runtime *ptr);

/// Options used when the global runtime was first created (Init path). Caller
/// must hold getGlobalRuntimeMutex() when reading or writing.
MODULAR_CXX_EXPORT RuntimeOptions &getStoredGlobalRuntimeCreationOptions();

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
