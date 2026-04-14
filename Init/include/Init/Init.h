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

#include <optional>

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

  Options &
  withRuntimeOptions(const MLRT::RuntimeOptions &v = MLRT::RuntimeOptions()) {
    runtimeOptions.emplace(v);
    return *this;
  }

  /// Returns true if crash-reporting and runtime options match \p other.
  bool operator==(const Options &other) const {
    if (forceDisableCrashReporting != other.forceDisableCrashReporting)
      return false;
    return runtimeOptions == other.runtimeOptions;
  }

  bool operator!=(const Options &other) const { return !(*this == other); }

  bool forceDisableCrashReportingEnabled() const {
    return forceDisableCrashReporting;
  }

  std::optional<MLRT::RuntimeOptions> getRuntimeOptions() const {
    return runtimeOptions;
  }

private:
  bool forceDisableCrashReporting = false;
  std::optional<MLRT::RuntimeOptions> runtimeOptions;

  friend ErrorOr<ContextRef> getOrCreateContext(StringRef, const Options &,
                                                StringRef);
};

// Creates the process-wide \c M::Context, reports a fatal error if a global
// context already exists. Intended for when there is a single point where the
// context is created.
MODULAR_CXX_EXPORT ErrorOr<ContextRef>
createContext(StringRef programName, const Options &options = {},
              StringRef subCommand = "");

/// Returns a reference to the existing process-wide \c M::Context, or creates
/// it if it doesn't already exist. Throws a fatal error if the \p options
/// requested do not match those used by the existing context. This function
/// should only be used when there is multiple possible paths to creating a
/// context and the order cannopt be guaranteed. Using createContext() and
/// getContext() is preferred when possible.
MODULAR_CXX_EXPORT ErrorOr<ContextRef>
getOrCreateContext(StringRef programName, const Options &options = {},
                   StringRef subCommand = "");

/// Returns a reference to the existing process-wide \c M::Context, reports a
/// fatal error if no context has been created yet.
MODULAR_CXX_EXPORT ContextRef getContext();

} // namespace Init
} // namespace M

#endif // ASYNCRT_INIT_INIT_H
