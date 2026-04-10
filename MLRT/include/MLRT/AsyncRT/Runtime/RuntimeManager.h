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
/// \p allowUsingExistingOptions may be set to true to disable the check that
/// the runtime options match and discard the provided options, but the caller
/// should ensure that it is safe to do so.
MODULAR_CXX_EXPORT RuntimeRef getOrCreateRuntime(
    RuntimeSource source, const RuntimeOptions &options = RuntimeOptions(),
    bool allowUsingExistingOptions = false);

} // namespace M::AsyncRT

#endif // MLRT_ASYNCRT_RUNTIME_RUNTIME_MANAGER_H
