//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_INIT_INIT_H
#define ASYNCRT_INIT_INIT_H

#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/Context.h"
#include "Support/ErrorOr.h"
#include "Support/SymbolExport.h"

namespace M {
namespace Init {

class Options {
public:
  Options() = default;
  Options(const Options &) = default;
  Options &withForceDisableCrashReporting(bool v = true) {
    forceDisableCrashReporting = v;
    return *this;
  }

  Options &withRuntimeOptions(
      const AsyncRT::RuntimeOptions &v = AsyncRT::RuntimeOptions()) {
    runtimeOptions.emplace(v);
    return *this;
  }

private:
  bool forceDisableCrashReporting = false;
  std::optional<AsyncRT::RuntimeOptions> runtimeOptions;

  friend ErrorOr<ContextRef> createContext(StringRef, const Options &,
                                           StringRef);
};

/// Create a new context, load all local configurations and entitlements,
/// save them in the context and return. This is expected to be the normal
/// path for context creation.
///
/// The context will come loaded with a TelemetryContext, Config and other
/// basic common functionality. The function will also initialize crash
/// reporting with the given programName.
ErrorOr<ContextRef> createContext(StringRef programName,
                                  const Options &options = {},
                                  StringRef subCommand = "");

/// Returns a reference to the process-wide global AsyncRT runtime, creating it
/// on first use with \p source and \p options. If a global runtime already
/// exists, triggers a fatal error if \p options do not match those used at
/// creation, and returns a copy of the existing reference.
MODULAR_CXX_EXPORT AsyncRT::RuntimeRef getOrCreateRuntime(
    AsyncRT::RuntimeSource source,
    const AsyncRT::RuntimeOptions &options = AsyncRT::RuntimeOptions());

} // namespace Init
} // namespace M

#endif // ASYNCRT_INIT_INIT_H
