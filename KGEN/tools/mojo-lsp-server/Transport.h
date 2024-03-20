//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H

#include "Protocol.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

namespace M::Mojo::LSP {
/// Class used to dispatch a response to the client. It also helps creating
/// responses for invalid inputs.
template <typename Result>
class LSPResponder {
public:
  LSPResponder(StringRef request, mlir::lsp::Callback<Result> replyCallback)
      : request(request), start(std::chrono::steady_clock::now()),
        replyCallback(std::move(replyCallback)){};
  LSPResponder(LSPResponder &&) = default;

  /// Used to reply to the client with the input data is invalid, e.g. the
  /// input location is not valid.
  void replyInvalidRequest() { reply(Result{}); }

  /// Used to reply to the client when the request has gone stale, e.g. the
  /// document has changed while the request was in the queue.
  void replyOutdatedRequest() { reply(Result{}); }

  /// Use to reply to the client when an actual response value was computed.
  void reply(Result result) { replyCallback(std::move(result)); }

private:
  LSPResponder(const LSPResponder &) = delete;
  LSPResponder &operator=(const LSPResponder &) = delete;

  std::string request;
  std::chrono::steady_clock::time_point start;
  mlir::lsp::Callback<Result> replyCallback;
};
} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_TRANSPORT_H
