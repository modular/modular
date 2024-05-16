//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoBuild/BSPClient.h"
#include "Config/Version.h"
#include "KGEN/MojoBuild/Protocol.h"
#include "Support/ErrorOr.h"

#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <cstdio>
#include <filesystem>
#include <string>
#ifndef _WIN32_
#include <unistd.h>
#endif

using namespace M;
using namespace M::Build;
using namespace mlir;

BSPClient::BSPClient(TempFile &&in, std::FILE *inFile, TempFile &&out,
                     int outFD, const std::string &displayName,
                     const std::string &rootUri,
                     const std::filesystem::path &serverPath)
    : in(std::move(in)), inFile(inFile), out(std::move(out)),
      outOS(outFD, /*shouldClose=*/true), transport(inFile, outOS),
      messageHandler(transport), displayName(displayName), rootUri(rootUri),
      serverPath(serverPath) {
  initializeRequestFn = messageHandler.outgoingRequest<InitializeBuildParams,
                                                       InitializeBuildResult>(
      "build/initialize",
      [&](llvm::json::Value id, llvm::Expected<InitializeBuildResult> result) {
        onBuildInitializeResponse(std::move(id), std::move(result));
      });
  buildFn = messageHandler.outgoingRequest<CompileParams, CompileResult>(
      "buildTarget/compile",
      [&](llvm::json::Value id, llvm::Expected<CompileResult> result) {
        onBuildTargetCompileResponse(std::move(id), std::move(result));
      });
  shutdownFn = messageHandler.outgoingRequest<NoParams, NoParams>(
      "build/shutdown",
      [&](llvm::json::Value id, llvm::Expected<NoParams> result) {
        onBuildShutdownResponse(std::move(id), std::move(result));
      });
  exitFn = messageHandler.outgoingNotification<NoParams>("exit");
}

ErrorOrSuccess BSPClient::run() {
  // Launch the build server, redirecting its stdin (where it reads data from)
  // from the output of the client, and its stdout (where it writes data to) to
  // the input of the client.
  const std::optional<StringRef> redirects[] = {
      /*stdin=*/out.getPath().c_str(),
      /*stdout=*/in.getPath().c_str(),
      /*stderr=*/std::nullopt,
  };
  std::string executeError;
  bool executionFailed = false;
  llvm::sys::ProcessInfo processInfo = llvm::sys::ExecuteNoWait(
      serverPath.c_str(), {serverPath.c_str()},
      /*Env=*/std::nullopt, redirects,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed)
    return Error(llvm::formatv("could not execute '{0}': {1}", serverPath,
                               executeError));

  // Send the initialization request to the server.
  initializeRequestFn(InitializeBuildParams{displayName,
                                            getModularVersionString(),
                                            /*bspVersion=*/"2.2.0", rootUri},
                      currentRequestID++);

  // Repeatedly wait on the server until it exits.
  while (true) {
    std::string waitError;
    llvm::sys::ProcessInfo waitInfo =
        llvm::sys::Wait(processInfo, /*SecondsToWait=*/0, &waitError);
    // The error message is populated whenever the process could not be waited
    // upon, or whenever it exits abnormally, such as due to a core dump.
    if (!waitError.empty())
      return Error(
          llvm::formatv("'{0}' exited abnormally: {1}", serverPath, waitError));

    // The process ID is not set if the process being waited upon has not
    // changed state. If it is set, the server must have exited.
    if (waitInfo.Pid) {
      if (waitInfo.ReturnCode == 0)
        return std::move(clientResult);
      return Error(llvm::formatv("'{0}' exited unsuccessfully: exit code {1}",
                                 serverPath, waitInfo.ReturnCode));
    }

    // We've checked above that the server is actually running; pump our client
    // JSON transport to send and receive messages.
    if (llvm::Error error = transport.run(messageHandler)) {
      llvm::consumeError(std::move(error));
      if (feof(inFile)) {
        // We're using a temporary file as a message buffer; we don't care about
        // reaching EOF, even though the transport treats this as an error.
        // Clear the error and move on.
        clearerr(inFile);
      } else {
        lsp::Logger::error("client transport error: {0}", error);
        return Error(llvm::formatv("JSON transport error: {0}", error));
      }
    }
  }
}

void BSPClient::onBuildInitializeResponse(
    llvm::json::Value id, llvm::Expected<InitializeBuildResult> result) {
  if (!result) {
    return llvm::handleAllErrors(
        result.takeError(), [&](const lsp::LSPError &err) {
          lsp::Logger::error("<--- client reply:build/initialize({0}): {1}", id,
                             err.message);
        });
  }

  // There's only one Mojo build server this tool speaks to, and it only
  // supports compiling Mojo.
  assert(result->capabilities.compileProvider->languageIds.front() == "mojo" &&
         "build server does not support compiling Mojo");

  // Send the build request to the server.
  buildFn(CompileParams{}, currentRequestID++);
}

void BSPClient::onBuildTargetCompileResponse(
    llvm::json::Value id, llvm::Expected<CompileResult> result) {
  std::string logPrefix =
      llvm::formatv("<--- client reply:buildTarget/compile({0}): ", id);
  if (!result) {
    return llvm::handleAllErrors(
        result.takeError(), [&](const lsp::LSPError &err) {
          lsp::Logger::error("<--- client reply:buildTarget/compile({0}): {1}",
                             id, err.message);
        });
  }

  // TODO: Build errors and build cancellation reasons are not yet communicated
  // from server to client.
  switch (result->statusCode) {
  case StatusCode::Ok:
    break;
  case StatusCode::Error:
    clientResult = Error("server could not build");
    break;
  case StatusCode::Cancelled:
    clientResult = Error("server cancelled build");
    break;
  }

  // We're done building; send the shutdown request to the server.
  shutdownFn(NoParams{}, currentRequestID++);
}

void BSPClient::onBuildShutdownResponse(llvm::json::Value id,
                                        llvm::Expected<NoParams> result) {
  if (!result) {
    return llvm::handleAllErrors(
        result.takeError(), [&](const lsp::LSPError &err) {
          lsp::Logger::error("<--- client reply:build/shutdown({0}): {1}", id,
                             err.message);
        });
  }

  // Server has shut down; tell it to exit.
  exitFn(NoParams{});
}
