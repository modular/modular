//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H
#define KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H

#include "Document.h"
#include "JSONUtils.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"

namespace M {

/// Helper struct containing the paths to the files that are used as stdio when
/// invoking the LSP server.
struct LSPServerStdioFiles {
  LSPServerStdioFiles(const std::filesystem::path &parentDir);

  std::string serverStdin;
  std::string serverStdout;
  std::string serverStderr;
};

/// Non-interactive batch client for the Mojo Language Server that collects
/// requests and sends them as a single JSON input to the server at once when
/// the `execute` method is invoked. Each registered request has an assigned
/// callback that is invoked when the corresponding response from the server is
/// received.
///
/// This class aborts if it is destroyed and the `execute` method hasn't been
/// invoked, or if such method is executed twice.
class LSPBatchClient {
private:
  /// Type-erasure class used to dispatch a response of an LSP request.
  class ResponseHandler {
  public:
    virtual ~ResponseHandler() = default;
    virtual ErrorOrSuccess onResponse(const llvm::json::Value &response) = 0;
  };

  /// Implementation of `ResponseHandler` with a specific Result type.
  template <typename Result>
  class ResponseHandlerImpl : public ResponseHandler {
  public:
    ResponseHandlerImpl(std::function<void(const Result &)> callback)
        : callback(std::move(callback)) {}

    ErrorOrSuccess onResponse(const llvm::json::Value &response) override {
      // Some requests might just want the raw JSON value.
      if constexpr (std::is_same_v<llvm::json::Value, Result>) {
        callback(response);
      } else {
        if (auto resultOr = llvm::json::parse<Result>(response))
          callback(*resultOr);
        else
          return toModularErrorOr(resultOr.takeError());
      }
      return success();
    }

  private:
    std::function<void(const Result &)> callback;
  };

public:
  using RequestId = int;

  struct ExecutionResult {
    ErrorOrSuccess err;
    std::optional<LSPServerStdioFiles> serverIOFiles = std::nullopt;
  };

  LSPBatchClient(std::optional<std::function<void(const ExecutionResult &)>>
                     onExecuteCallback = std::nullopt);
  ~LSPBatchClient();

  /// textDocument/didOpen
  LSPBatchClient &open(const Document &doc);

  /// textDocument/definition
  LSPBatchClient &definition(
      const Document &doc, const mlir::lsp::Position &position,
      std::function<void(const std::vector<mlir::lsp::Location> &)> callback);

  /// testDocument/codeAction
  LSPBatchClient &codeAction(
      const Document &doc, const mlir::lsp::Range &range,
      std::initializer_list<mlir::lsp::Diagnostic> diags,
      std::function<void(const std::vector<mlir::lsp::CodeAction> &)> callback);

  /// textDocument/hover
  LSPBatchClient &
  hover(const Document &doc, const mlir::lsp::Position &position,
        std::function<void(const mlir::lsp::Hover2 &)> callback);

  /// Actual `execute` logic.
  ErrorOrSuccess doExecute(const LSPServerStdioFiles &ioFiles,
                           StringRef lspServerPath);

  /// Create a single JSON input for the server, then parse the response and
  /// invoke the corresponding response callbacks.
  ///
  /// This method aborts if it's invoked twice.
  ExecutionResult execute();

private:
  friend class LSPBatchClientExecutor;

  /// Register a notification to be sent to the server.
  void notify(StringRef method, const llvm::json::Value &params);

  /// Register a request to be sent to the server.
  template <typename Result>
  void request(StringRef method, const llvm::json::Value &params,
               std::function<void(const Result &)> callback) {
    RequestId id = requestId++;
    responseHandlers.try_emplace(
        id, std::unique_ptr<ResponseHandler>(
                new ResponseHandlerImpl<Result>(std::move(callback))));

    appendJSONRequest(id, method, params);
  }

  /// Append a JSON request to the server input stream.
  void appendJSONRequest(RequestId id, StringRef method,
                         const llvm::json::Value &params);

  /// Append to the server input the shutdown and exit messages, which marks
  /// the end of the input.
  void appendShutdownAndExit();

  /// Dispatch a single response given its JSON contents.
  ErrorOrSuccess dispatchResponse(StringRef json);

  /// Parses the JSON output from the server and invoke the corresponding
  /// handlers, erasing them as they are responded.
  ErrorOrSuccess dispatchResponses(StringRef serverStdout);

  std::optional<std::function<void(const ExecutionResult &)>> onExecuteCallback;
  /// Flag that indicates whether `execute` has been invoked.
  bool didExecute = false;
  /// The collected JSON input to send to the server.
  std::string serverJSONInput;
  /// An output stream for `jsonInput`.
  llvm::raw_string_ostream serverJSONInputOS;
  /// A monotonically increasing request id used when collecting requests.
  RequestId requestId = 0;
  /// A map from request id to response handler.
  llvm::DenseMap<RequestId, std::unique_ptr<ResponseHandler>> responseHandlers;
};

} // namespace M

#endif // KGEN_TOOLS_MOJO_LSP_TEST_CLIENT_LSPBATCHCLIENT_H
