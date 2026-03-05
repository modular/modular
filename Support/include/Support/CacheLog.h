//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CACHELOG_H
#define SUPPORT_CACHELOG_H

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

namespace M {

/// Returns true if the MODULAR_ENABLE_CACHE_LOGGING env var is set to a
/// non-empty, non-"0" value. The result is cached after the first call.
inline bool isCacheLogEnabled() {
  static const bool enabled = [] {
    auto env = llvm::sys::Process::GetEnv("MODULAR_ENABLE_CACHE_LOGGING");
    return env.has_value() && !env->empty() && *env != "0";
  }();
  return enabled;
}

/// Returns a logging stream for cache diagnostics. When cache logging is
/// disabled (the default), returns llvm::nulls(). When enabled, returns
/// llvm::errs() with a "[modular-cache][<prefix>] " prefix already written.
inline llvm::raw_ostream &cacheLog(llvm::StringRef prefix) {
  if (!isCacheLogEnabled())
    return llvm::nulls();
  return llvm::errs() << "[modular-cache][" << prefix << "] ";
}

} // namespace M

/// Log to the cache diagnostic stream. When MODULAR_ENABLE_CACHE_LOGGING is
/// not set, the entire statement (including operator<< arguments) is skipped,
/// so arguments are never evaluated. Usage:
///   MODULAR_CACHE_LOG("mef") << "cache hit for " << path.string() << "\n";
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define MODULAR_CACHE_LOG(prefix)                                              \
  switch (0)                                                                   \
  default:                                                                     \
    if (!::M::isCacheLogEnabled()) {                                           \
    } else                                                                     \
      ::M::cacheLog(prefix)

#endif // SUPPORT_CACHELOG_H
