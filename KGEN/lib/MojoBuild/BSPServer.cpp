//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BSPServer.h"

#include "Config/Version.h"
#include "KGEN/MojoBuild/Protocol.h"
#include "KGEN/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "Support/Filesystem/Paths.h"

#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
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

/// Returns the path to the `mojo` executable, or an error if none could be
/// found.
static ErrorOr<std::string> getMojoDriverPath(KGEN::MojoConfig &config) {
  std::error_code ec;
  StringRef path = config.getDriverPath();
  if (!std::filesystem::exists(path.str(), ec) || ec)
    return Error(
        llvm::formatv("unable to resolve the mojo path at '{0}'", path));
  return path.str();
}

/// Returns all Mojo source package directories at the top level of the given
/// `root` directory.
static SmallVector<std::filesystem::path>
getMojoSourcePackageDirectories(const std::filesystem::path &root,
                                std::error_code &ec) {
  SmallVector<std::filesystem::path> result;
  std::filesystem::directory_iterator it(root, ec);
  if (ec)
    return result;

  for (const auto &path : it)
    if (Filesystem::isMojoSourcePackagePath(path))
      result.push_back(path);

  return result;
}

void BSPServer::onBuildTargetCompile(
    const CompileParams &params, mlir::lsp::Callback<CompileResult> callback) {
  if (auto err = errorIfUninitialized())
    return callback(std::move(err));

  // Grab the path to the `mojo` driver.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (configOr.isError())
    return callback(error(
        llvm::formatv("could not read 'modular.cfg': {0}", configOr.getError()),
        lsp::ErrorCode::RequestFailed));
  ErrorOr<std::string> driverPathOr = getMojoDriverPath(*configOr);
  if (driverPathOr.isError())
    return callback(
        error(driverPathOr.getError(), lsp::ErrorCode::RequestFailed));

  // Prepare the compile result object.
  CompileResult result;
  if (params.originId)
    result.originId = *params.originId;

  // We do not yet define a Mojo project manifest, so we cannot be certain what
  // to build within our workspace. For the time being, look for a directory
  // named 'src' at the root of the workspace, and build a Mojo package from
  // each Mojo source package directory within 'src'. If there's nothing to
  // build, return a "cancelled" result.
  auto respondWithCancelled = [&]() {
    result.statusCode = StatusCode::Cancelled;
    callback(std::move(result));
  };

  std::error_code ec;
  std::filesystem::path srcDir = workspacePath / "src";
  if (!std::filesystem::is_directory(srcDir, ec) || ec)
    return respondWithCancelled();

  SmallVector<std::filesystem::path> packageDirs =
      getMojoSourcePackageDirectories(srcDir, ec);
  if (packageDirs.empty() || ec)
    return respondWithCancelled();

  // Create a directory for build artifacts.
  std::filesystem::path buildDirectory = workspacePath / ".build";
  std::filesystem::create_directory(buildDirectory, ec);
  if (ec)
    return callback(
        error(llvm::formatv("could not create build directory '{0}': {1}",
                            buildDirectory, ec.message()),
              lsp::ErrorCode::RequestFailed));

  // For now, as a simple proof of concept, this sequentially builds the
  // packages in the workspace. In the future, this should be parallelized.
  for (const auto &packageDir : packageDirs) {
    SmallVector<StringRef> driverArgs{*driverPathOr, "package",
                                      packageDir.c_str(), "-o",
                                      buildDirectory.c_str()};
    std::string errorMessage;
    int exitCode = llvm::sys::ExecuteAndWait(
        *driverPathOr, driverArgs,
        /*Env=*/std::nullopt, /*Redirects=*/{}, /*SecondsToWait=*/0,
        /*MemoryLimit=*/0, /*ErrMsg=*/&errorMessage);

    // An error launching the driver is an internal server error, and we treat
    // it as distinct from a compilation error.
    if (!errorMessage.empty())
      return callback(error(errorMessage, lsp::ErrorCode::InternalError));

    // For now, we stop building as soon as any package fails to build.
    if (exitCode) {
      result.statusCode = StatusCode::Error;
      return callback(std::move(result));
    }
  }

  // We built one or more packages successfully.
  result.statusCode = StatusCode::Ok;
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
