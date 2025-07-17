//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_INIT_INIT_H
#define ASYNCRT_INIT_INIT_H

#include "AsyncRT/Runtime/Runtime.h"
#include "Support/Context.h"
#include "Support/ErrorOr.h"

namespace M {
namespace Init {

class Options {
public:
  Options() = default;
  Options &withForceDisableCrashReporting(bool v = true) {
    forceDisableCrashReporting = v;
    return *this;
  }
  Options &withRuntimeOptions(const AsyncRT::RuntimeOptions &v = {}) {
    runtimeOptions.emplace(v);
    return *this;
  }

private:
  bool forceDisableCrashReporting = false;
  std::optional<AsyncRT::RuntimeOptions> runtimeOptions;

  friend ErrorOr<ContextRef> createContext(StringRef, const Options &);
};

/// Create a new context, load all local configurations and entitlements,
/// save them in the context and return. This is expected to be the normal
/// path for context creation.
///
/// The context will come loaded with a TelemetryContext, Config and other
/// basic common functionality. The function will also initialize crash
/// reporting with the given programName.
ErrorOr<ContextRef> createContext(StringRef programName,
                                  const Options &options = {});

} // namespace Init
} // namespace M

#endif // ASYNCRT_INIT_INIT_H
