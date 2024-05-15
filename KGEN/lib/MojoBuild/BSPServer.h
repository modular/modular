//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_BSPSERVER_H
#define KGEN_MOJOBUILD_BSPSERVER_H

#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"

#include "mlir/Tools/lsp-server-support/Transport.h"

#include <cstddef>

namespace M::Build {

/// Implements generic requests and responses for a build server, loosely
/// following the specification defined here:
/// https://build-server-protocol.github.io/docs/specification
///
/// This class does not (and should not) implement logic specifically related to
/// Mojo and building Mojo projects.
class BSPServer {
public:
  /// Initializes the server. When `debug` is true, the underlying JSON
  /// transport is configured to accept messages delimited by `// -----`, for
  /// testing and debugging purposes.
  BSPServer(bool debug);

  /// Starts the server runloop, blocking until an error occurs or the server
  /// receives a shutdown request and exits normally.
  ErrorOrSuccess run();

private:
  /// Handles the `build/initialize` request.
  void onBuildInitialize(const InitializeBuildParams &params,
                         mlir::lsp::Callback<InitializeBuildResult> callback);
  /// Handles the `build/shutdown` request.
  void onBuildShutdown(const NoParams &params,
                       mlir::lsp::Callback<NoParams> callback);

  /// Fails the server and returns an error with the given message and code.
  llvm::Error error(Twine message, mlir::lsp::ErrorCode code);

  /// A JSON-RPC transport that reads requests from stdin, and writes responses
  /// and notifications to stdout.
  mlir::lsp::JSONTransport transport;
  /// A message handler that maps request types to response callbacks.
  mlir::lsp::MessageHandler messageHandler;

  /// If non-null, indicates that the server should shut down and exit with the
  /// specified result.
  std::optional<mlir::LogicalResult> serverResult = std::nullopt;
};
} // namespace M::Build

#endif
