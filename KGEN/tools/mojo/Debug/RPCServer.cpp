//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RPCServer.h"
#include "Support/Configuration.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include <filesystem>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
typedef int socklen_t;
#else
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#endif

using namespace M;

namespace json = llvm::json;

#if defined(_WIN32)
extern char **_environ;
#else
extern char **environ;
#endif

char **getEnv() {
#ifdef _WIN32
  static char **envp = _environ;
#else
  static char **envp = environ;
#endif
  return envp;
}

/// Create an object with the common fields that are sent to the RPC server
/// to start the different kinds of requests.
static ErrorOr<json::Object>
createBasicRPCPayload(const std::optional<StringRef> &secret) {
  ErrorOr<std::filesystem::path> modularHome =
      Config::getModularConfigFolderPath();
  if (failed(modularHome))
    return modularHome.takeError();

  json::Object payload{{"modularHomePath", modularHome->string()},
                       {"type", "mojo-lldb"}};

  if (secret)
    payload.insert({"token", *secret});

  return payload;
}

/// Send the given payload to the RPC server at the specified port. If `dryRun`
/// is specified, then the payload is printed to the standard output instead.
static ErrorOrSuccess invokeRPC(bool dryRun, int port, json::Object payload) {
  std::string payloadStr =
      llvm::formatv("{0:2}", json::Value(std::move(payload)));
  if (dryRun) {
    llvm::outs() << payloadStr << "\n";
    return success();
  }

  SOCKET sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    return Error(
        Twine("can't open socket to communicate with the RPC debug server: ") +
        strerror(errno));
  }

  struct sockaddr_in serverAddress;
  memset((char *)&serverAddress, 0, sizeof(serverAddress));
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_port = htons(port);
  serverAddress.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  ErrorOrSuccess status = success();
  if (connect(sockfd, (struct sockaddr *)&serverAddress,
              sizeof(serverAddress)) < 0) {
    status = Error(Twine("can't connect to the RPC debug server socket: ") +
                   strerror(errno));
  } else {
    ssize_t sentBytes =
        send(sockfd, payloadStr.c_str(), payloadStr.length(), 0);
    if (sentBytes < 0) {
      status = Error(Twine("can't send data to the RPC debug server: ") +
                     strerror(errno));
    }
  }

#if defined(_WIN32)
  closesocket(sockfd);
#else
  close(sockfd);
#endif
  return status;
}

ErrorOrSuccess M::invokeAttachRPC(bool dryRun, int rpcPort,
                                  const std::optional<StringRef> &secret,
                                  const std::optional<StringRef> &pid,
                                  const std::optional<StringRef> &processName) {
  ErrorOr<json::Object> payload = createBasicRPCPayload(secret);
  if (failed(payload))
    return payload.takeError();
  payload->insert({"request", "attach"});
  if (pid)
    payload->insert({"pid", *pid});
  if (processName)
    payload->insert({"program", *processName});
  return invokeRPC(dryRun, rpcPort, *payload);
}

ErrorOrSuccess M::invokeLaunchRPC(bool dryRun, int rpcPort,
                                  const std::optional<StringRef> &secret,
                                  StringRef target,
                                  ArrayRef<std::string> runArgs,
                                  StringRef rpcTerminal) {
  ErrorOr<json::Object> payload = createBasicRPCPayload(secret);
  if (failed(payload))
    return payload.takeError();

  std::error_code ec;
  std::filesystem::path fullTarget =
      std::filesystem::absolute(target.str(), ec);
  if (ec)
    return Error("failed to get absolute path to the target '" + target +
                 "': " + ec.message());

  std::filesystem::path cwd = std::filesystem::current_path(ec);
  if (ec)
    return Error("failed to get the current working path: " + ec.message());

  json::Array env;
  for (char **entry = getEnv(); *entry; ++entry)
    env.push_back(*entry);

  payload->insert({"program", fullTarget.string()});
  payload->insert({"request", "launch"});
  payload->insert({"cwd", cwd.string()});
  payload->insert({"args", json::Array{runArgs}});
  payload->insert({"env", std::move(env)});
  payload->insert({"runInTerminal", rpcTerminal == "dedicated"});

  return invokeRPC(dryRun, rpcPort, *payload);
}
