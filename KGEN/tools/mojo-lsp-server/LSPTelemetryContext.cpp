//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LSPTelemetryContext.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace M::Mojo;
using namespace M::Mojo::LSP;

LSPTelemetryContext::LSPTelemetryContext(Telemetry::TelemetryContext &ctx)
    : responseTimeHistogram(ctx.createUInt64Histogram(
          "mojo.lsp.request.time", Telemetry::Level::L1,
          /*attributes=*/{},
          "Time it took to respond valid LSP requests that were effectively "
          "computed.",
          /*unit=*/"microsecond")),
      outdatedRequestCounter(ctx.createUInt64Counter(
          "mojo.lsp.request.outdated", Telemetry::Level::L1, /*attributes=*/{},
          "Number of outdated LSP requests.")),
      invalidRequestCounter(ctx.createUInt64Counter(
          "mojo.lsp.request.invalid", Telemetry::Level::L1, /*attributes=*/{},
          "Number of invalid LSP requests.")) {}

void LSPTelemetryContext::recordResponseTime(
    StringRef request, std::chrono::microseconds microseconds) {
  responseTimeHistogram.record(microseconds.count(),
                               {{"request", request.str()}});
}

void LSPTelemetryContext::recordInvalidRequest(StringRef request) {
  invalidRequestCounter.add(1, {{"request", request.str()}});
}

void LSPTelemetryContext::recordOutdatedRequest(StringRef request) {
  outdatedRequestCounter.add(1, {{"request", request.str()}});
}
