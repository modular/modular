//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_BSPCLIENT_H
#define KGEN_MOJOBUILD_BSPCLIENT_H

#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"

#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace M::Build {

/// Implements generic requests and responses for a build server client, loosely
/// following the specification defined here:
/// https://build-server-protocol.github.io/docs/specification
///
/// This class does not (and should not) implement logic specifically related to
/// Mojo and requests for Mojo projects to be built.
class BSPClient {
public:
  /// Initializes the client with the given file stream for stdin (for messages
  /// sent to the client, from the server) and file descriptor for stdout (for
  /// messages sent from the client, to the server).
  BSPClient(TempFile &&in, std::FILE *inFile, TempFile &&out, int outFD,
            const std::string &displayName, const std::string &rootUri,
            const std::filesystem::path &serverPath);

  /// Starts the client-server communication runloop, blocking until either an
  /// error occurs or the server exits. If the server fails to launch, or exits
  /// unsuccessfully, returns an error.
  ErrorOrSuccess run();

private:
  /// Handles the response to the `build/initialize` request.
  void onBuildInitializeResponse(llvm::json::Value id,
                                 llvm::Expected<InitializeBuildResult> result);
  /// Handles the response to the `buildTarget/compile` request.
  void onBuildTargetCompileResponse(llvm::json::Value id,
                                    llvm::Expected<CompileResult> result);
  /// Handles the response to the `build/shutdown` request.
  void onBuildShutdownResponse(llvm::json::Value id,
                               llvm::Expected<NoParams> result);

  /// The backing file for the client's input file stream.
  TempFile in;
  /// The input file stream from which the client transport reads data.
  std::FILE *inFile;

  /// The backing file for the client's output file stream.
  TempFile out;
  /// The output file stream to which the client transport writes data.
  llvm::raw_fd_ostream outOS;

  /// A JSON-RPC transport that reads requests and writes responses.
  mlir::lsp::JSONTransport transport;
  /// A message handler that maps request types to response callbacks.
  mlir::lsp::MessageHandler messageHandler;

  /// The name of the client.
  std::string displayName;
  /// The root URI of the workspace.
  std::string rootUri;
  /// The path to the server executable.
  std::filesystem::path serverPath;

  /// An internal counter that represents the ID of the next request to be sent.
  /// This is incremented each time a request is sent to the server.
  int currentRequestID = 0;
  /// The result of the client's communications with the build server.
  ErrorOrSuccess clientResult;

  /// A function that, when invoked, sends a "build/initialize" request with the
  /// given params and request ID.
  mlir::lsp::OutgoingRequest<InitializeBuildParams> initializeRequestFn;
  /// A function that, when invoked, sends a "buildTarget/compile" request with
  /// the given params and request ID.
  mlir::lsp::OutgoingRequest<CompileParams> buildFn;
  /// A function that, when invoked, sends a "build/shutdown" request with the
  /// given request ID.
  mlir::lsp::OutgoingRequest<NoParams> shutdownFn;
  /// A function that, when invoked, sends an "exit" notification.
  mlir::lsp::OutgoingNotification<NoParams> exitFn;
};
} // namespace M::Build

#endif // KGEN_MOJOBUILD_BSPCLIENT_H
