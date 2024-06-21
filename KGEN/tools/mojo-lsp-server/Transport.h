//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H

#include "../common/lsp-protocol/Protocol.h"
#include "LSPTelemetryContext.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

namespace M::Mojo::LSP {
/// Class used to dispatch a response to the client and perform telemetry at the
/// request level. It also helps creating responses for invalid inputs and
/// tracking them with telemetry.
template <typename Result>
class LSPResponder {
public:
  LSPResponder(LSPTelemetryContext &lspTelemetryCtx, StringRef request,
               mlir::lsp::Callback<Result> replyCallback)
      : lspTelemetryCtx(lspTelemetryCtx), request(request),
        start(std::chrono::steady_clock::now()),
        replyCallback(std::move(replyCallback)) {}
  LSPResponder(LSPResponder &&) = default;

  /// Used to reply to the client with the input data is invalid, e.g. the
  /// input location is not valid.
  void replyInvalidRequest() {
    replyError(mlir::lsp::LSPError("invalid request",
                                   mlir::lsp::ErrorCode::InvalidRequest));
    lspTelemetryCtx.recordInvalidRequest(request);
  }

  /// Used to reply to the client when the request has gone stale, e.g. the
  /// document has changed while the request was in the queue.
  void replyOutdatedRequest() {
    replyError(mlir::lsp::LSPError("outdated request",
                                   mlir::lsp::ErrorCode::ContentModified));
    lspTelemetryCtx.recordOutdatedRequest(request);
  }

  /// Use to reply to the client when an actual response value was computed.
  void reply(Result result) {
    auto end = std::chrono::steady_clock::now();
    replyCallback(std::move(result));

    lspTelemetryCtx.recordResponseTime(
        request,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start));
  }

  /// Use to reply to the client with an arbitrary error.
  void replyError(mlir::lsp::LSPError error) {
    auto end = std::chrono::steady_clock::now();
    replyCallback(llvm::make_error<mlir::lsp::LSPError>(std::move(error)));

    lspTelemetryCtx.recordResponseTime(
        request,
        std::chrono::duration_cast<std::chrono::microseconds>(end - start));
  }

private:
  LSPResponder(const LSPResponder &) = delete;
  LSPResponder &operator=(const LSPResponder &) = delete;

  LSPTelemetryContext &lspTelemetryCtx;
  std::string request;
  std::chrono::steady_clock::time_point start;
  mlir::lsp::Callback<Result> replyCallback;
};
} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H
