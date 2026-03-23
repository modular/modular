//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Manages a single global Runtime (similar to the global M::Context). Provides
// getOrCreateRuntime() to obtain a reference to the global runtime, creating it
// on first use.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_RUNTIMEMANAGER_H
#define MLRT_ASYNCRT_RUNTIME_RUNTIMEMANAGER_H

#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/SymbolExport.h"

namespace M::AsyncRT {

/// Manages a single global Runtime. The static runtime pointer is set on first
/// call to getOrCreateRuntime() and remains valid until the runtime is
/// destroyed.
class RuntimeManager {
public:
  /// Returns a reference to the global runtime. If the global runtime has not
  /// been set, creates it with the given \p source and \p options and then
  /// returns a copy. If already set, asserts that \p options equals the
  /// options used at creation and returns a copy of the existing reference.
  /// Calling with different options triggers an assertion failure.
  static MODULAR_CXX_EXPORT RuntimeRef getOrCreateRuntime(
      RuntimeSource source, const RuntimeOptions &options = RuntimeOptions());
};

} // namespace M::AsyncRT

#endif // MLRT_ASYNCRT_RUNTIME_RUNTIMEMANAGER_H
