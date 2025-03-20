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
          "Number of invalid LSP requests.")),
      ctx(ctx) {}

void LSPTelemetryContext::recordResponseTime(
    StringRef request, std::chrono::microseconds microseconds) {
#ifdef MODULAR_ENABLE_TELEMETRY
  responseTimeHistogram.record(microseconds.count(),
                               {{"request", request.str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::recordInvalidRequest(StringRef request) {
#ifdef MODULAR_ENABLE_TELEMETRY
  invalidRequestCounter.add(1, {{"request", request.str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::recordOutdatedRequest(StringRef request) {
#ifdef MODULAR_ENABLE_TELEMETRY
  outdatedRequestCounter.add(1, {{"request", request.str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::reportInitialization(
    std::optional<StringRef> clientName) {
#ifdef MODULAR_ENABLE_TELEMETRY
  ctx.getLogger("mojo")->emitL0Event(
      "lsp.initialized", {{"client_name", clientName.value_or("").str()}});
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::reportShutdown() {
#ifdef MODULAR_ENABLE_TELEMETRY
  ctx.getLogger("mojo")->emitL0Event("lsp.shutdown");
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::flush() {
#ifdef MODULAR_ENABLE_TELEMETRY
  ctx.flush();
#endif // MODULAR_ENABLE_TELEMETRY
}

void LSPTelemetryContext::recordParseTime(std::chrono::microseconds duration,
                                          size_t byteSize, bool notebook) {
#ifdef MODULAR_ENABLE_TELEMETRY
  ctx.getLogger("mojo")->emitL1Event(
      "lsp.parse", {
                       {"duration", duration.count()},
                       {"size", byteSize},
                       {"documentType", notebook ? "notebook" : "text"},
                   });
#endif // MODULAR_ENABLE_TELEMETRY
}
