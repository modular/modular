//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H
#define KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H

#include "Support/Driver/DriverSupport.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Option/OptTable.h"
#include <thread>

namespace M {

// Simple RAII utility wrapper around a `std::thread` that joins it once
// destructed.
class ScopedThread {
public:
  template <typename... ThreadArgs>
  ScopedThread(ThreadArgs &&...args) : thread(args...) {}

  ~ScopedThread() {
    if (thread.joinable())
      thread.join();
  }

private:
  ScopedThread(const ScopedThread &) = delete;
  ScopedThread &operator=(const ScopedThread &) = delete;

  std::thread thread;
};
/// Log in a new thread the invocation of a mojo tool, including its arguments.
/// An additional set of "private" arguments can be provided, which will be
/// redacted from telemetry events.
///
/// This returns a `ScopedThread` that performs the logging in a different
/// thread than the main one, thus preventing slow downs due to network
/// communication.
ScopedThread logToolInvocationEventAsync(
    M::Telemetry::TelemetryContext &telemetryCtx, StringRef message,
    const llvm::opt::InputArgList &args, ArrayRef<unsigned> privateArgs = {});

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_TELEMETRY_H
