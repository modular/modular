//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"
#include "MojoBuildServer.h"

#include "Config/Version.h"
#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

using namespace M;
using namespace M::Build;
using namespace mlir;

BSPServer::BSPServer(bool debug)
    : transport(stdin, llvm::outs(),
                debug ? lsp::JSONStreamStyle::Delimited
                      : lsp::JSONStreamStyle::Standard,
                /*prettyOutput=*/debug),
      messageHandler(transport) {
  messageHandler.method("build/initialize", this,
                        &BSPServer::onBuildInitialize);
  messageHandler.method("buildTarget/compile", this,
                        &BSPServer::onBuildTargetCompile);
  messageHandler.method("build/shutdown", this, &BSPServer::onBuildShutdown);
}

ErrorOrSuccess BSPServer::run() {
  auto runTransport =
      [](lsp::JSONTransport &transport,
         lsp::MessageHandler &messageHandler) -> ErrorOrSuccess {
    if (llvm::Error error = transport.run(messageHandler)) {
      llvm::consumeError(std::move(error));
      if (feof(stdin)) {
        clearerr(stdin);
      } else {
        lsp::Logger::error("server transport error: {0}", error);
        return Error(llvm::formatv("JSON transport error: {0}", error));
      }
    }
    return success();
  };

  while (!serverResult.has_value()) {
    ErrorOrSuccess result = runTransport(transport, messageHandler);
    if (failed(result))
      return result.takeError();
  }

  // If the server needs to be shut down due to an error, exit immediately.
  if (failed(*serverResult))
    return Error("server shut down due to an error");

  // Otherwise, after receiving a shutdown request, run the transport one last
  // time, to process any communication related to shutting down.
  ErrorOrSuccess result = runTransport(transport, messageHandler);
  if (failed(result))
    return result.takeError();

  return success();
}

//===----------------------------------------------------------------------===//
// Request handlers
//===----------------------------------------------------------------------===//

void BSPServer::onBuildInitialize(
    const InitializeBuildParams &params,
    mlir::lsp::Callback<InitializeBuildResult> callback) {
  if (isInitialized())
    return callback(error("server has already been initialized",
                          lsp::ErrorCode::InvalidRequest));

  // Initialize the server with a workspace path based on the rootUri.
  // (In the future, a rootUri may be a specific project manifest file. For now,
  // we assume it is a directory, and file paths result in errors.)
  workspacePath = params.rootUri;

  auto respondWithFilesystemError = [&](const std::error_code &ec) {
    return callback(error(
        llvm::formatv("server could not be initialized, an error occurred when "
                      "accessing rootUri '{0}': {1}",
                      workspacePath, ec.message()),
        lsp::ErrorCode::InvalidParams));
  };

  std::error_code ec;
  bool exists = std::filesystem::exists(workspacePath, ec);
  if (ec)
    return respondWithFilesystemError(ec);
  if (!exists)
    return callback(error(
        llvm::formatv(
            "server could not be initialized, rootUri '{0}' does not exist",
            workspacePath),
        lsp::ErrorCode::InvalidParams));

  bool isDirectory = std::filesystem::is_directory(workspacePath, ec);
  if (ec)
    return respondWithFilesystemError(ec);
  if (!isDirectory)
    return callback(
        error(llvm::formatv("rootUri '{0}' must be a directory", workspacePath),
              lsp::ErrorCode::InvalidParams));

  callback(InitializeBuildResult{
      "mojo-build-server", getModularVersionString(),
      /*bspVersion=*/"2.2.0",
      BuildServerCapabilities{CompileProvider{{"mojo"}}}});
}

void BSPServer::onBuildTargetCompile(
    const CompileParams &params, mlir::lsp::Callback<CompileResult> callback) {
  if (auto err = errorIfUninitialized())
    return callback(std::move(err));

  ErrorOr<BuildResult> buildResultOr =
      MojoBuildServer::buildWorkspace(workspacePath);
  if (buildResultOr.isError())
    return callback(
        error(buildResultOr.getError(), lsp::ErrorCode::RequestFailed));

  CompileResult result;
  if (params.originId)
    result.originId = *params.originId;
  switch (*buildResultOr) {
  case BuildResult::Success:
    result.statusCode = StatusCode::Ok;
    break;
  case BuildResult::Failure:
    result.statusCode = StatusCode::Error;
    break;
  case BuildResult::NothingToDo:
    result.statusCode = StatusCode::Cancelled;
    break;
  }
  callback(std::move(result));
}

void BSPServer::onBuildShutdown(const NoParams &params,
                                mlir::lsp::Callback<NoParams> callback) {
  if (auto err = errorIfUninitialized())
    return callback(std::move(err));

  serverResult = success();
  callback(NoParams{});
}

llvm::Error BSPServer::error(Twine message, mlir::lsp::ErrorCode code) {
  serverResult = failure();
  return llvm::make_error<lsp::LSPError>(message.str(), code);
}

bool BSPServer::isInitialized() const { return !workspacePath.empty(); }

llvm::Error BSPServer::errorIfUninitialized() {
  if (!isInitialized())
    return error("server has not been initialized",
                 lsp::ErrorCode::ServerNotInitialized);
  return llvm::Error::success();
}
