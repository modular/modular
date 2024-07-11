//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H
#define KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H

#include "Support/Telemetry/Telemetry.h"

namespace M::AsyncRT {
class Runtime;
}

namespace M::Mojo::LSP {
/// Utility that manages the objects used to perform telemetry related to the
/// LSP.
class LSPTelemetryContext {
public:
  LSPTelemetryContext(Telemetry::TelemetryContext &ctx);

  /// Record a metric after a successful response of a specific request.
  void recordResponseTime(StringRef request,
                          std::chrono::microseconds microseconds);

  /// Record a metric signaling a request with an invalid input.
  void recordInvalidRequest(StringRef request);

  /// Record a metric signaling a request that has gone stale, e.g. the
  /// document has changed while the request was in the queue.
  void recordOutdatedRequest(StringRef request);

private:
  Telemetry::Histogram<uint64_t> responseTimeHistogram;
  Telemetry::Counter<uint64_t> outdatedRequestCounter;
  Telemetry::Counter<uint64_t> invalidRequestCounter;
};
} // namespace M::Mojo::LSP

#endif // KGEN_LIB_MOJO_LSP_LSPTELEMETRYCONTEXT_H
