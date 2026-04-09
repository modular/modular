//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_RUNTIME_MANAGER_H
#define MLRT_ASYNCRT_RUNTIME_RUNTIME_MANAGER_H

#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/SymbolExport.h"

namespace M::AsyncRT {

/// Returns a reference to the process-wide global AsyncRT runtime, creating it
/// on first use with \p source and \p options. If a global runtime already
/// exists, triggers a fatal error if \p options do not match those used at
/// creation, and returns a copy of the existing reference.
MODULAR_CXX_EXPORT RuntimeRef getOrCreateRuntime(
    RuntimeSource source, const RuntimeOptions &options = RuntimeOptions());

/// If the global AsyncRT runtime has already been created, return the options
/// that it was created with. Otherwise, return the default RuntimeOptions.
MODULAR_CXX_EXPORT AsyncRT::RuntimeOptions getDefaultOrExistingRuntimeOptions();

} // namespace M::AsyncRT

#endif // MLRT_ASYNCRT_RUNTIME_RUNTIME_MANAGER_H
