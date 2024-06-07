//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "RPCServer.h"
#include "Support/Configuration.h"
#include "Support/Process.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include <filesystem>
#include <future>
#include <thread>

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
    payload.insert({"secret", *secret});

  return payload;
}

namespace {
struct RPCResponse {
  bool success;
  std::optional<std::string> message;
};
} // namespace

namespace llvm::json {
bool fromJSON(const json::Value &value, RPCResponse &response, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("success", response.success) &&
         o.mapOptional("message", response.message);
}
} // namespace llvm::json

static ErrorOrSuccess doSendRequest(SOCKET sockfd, StringRef payloadStr) {
  ssize_t sentBytes = send(sockfd, payloadStr.data(), payloadStr.size(), 0);
  if (sentBytes < 0)
    return Error(Twine("can't send data to the RPC debug server: ") +
                 strerror(errno));

  std::string rawResponse;
  while (true) {
    char buff[256];
    ssize_t recvBytes = 0;
    recvBytes = recv(sockfd, buff, sizeof(buff) - 1, 0);
    if (recvBytes < 0)
      return Error(Twine("can't receive response from the RPC debug server: ") +
                   strerror(errno));

    if (recvBytes == 0)
      break;

    rawResponse.append(buff, recvBytes);
  }

  llvm::Expected<RPCResponse> response =
      llvm::json::parse<RPCResponse>(rawResponse);
  if (!response) {
    llvm::consumeError(response.takeError());
    return Error(Twine("can't parse response from the RPC debug server: ") +
                 rawResponse);
  }
  if (response->success)
    return success();
  if (response->message)
    return Error(*response->message);
  return Error("couldn't initialize the debug session");
}

/// Send the given payload to the RPC server at the specified port. If `dryRun`
/// is specified, then the payload is printed to the standard output instead.
static ErrorOrSuccess invokeRPC(bool dryRun, int port, json::Object payload) {
  std::string payloadStr =
      llvm::formatv("{0:2}", json::Value(std::move(payload)));
  if (dryRun) {
    llvm::outs() << "port: " << port << "\n";
    llvm::outs() << "payload: " << payloadStr << "\n";
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
    // We create a dangling thread to easily handle timeouts. The thread will
    // die anyway as soon as we close the socket.
    auto future = new std::future<ErrorOrSuccess>(
        std::async(doSendRequest, sockfd, payloadStr));
    auto timeout = std::chrono::seconds(5);
    if (future->wait_for(timeout) == std::future_status::timeout) {
      status = Error("timeout when communicating with the RPC debug server");
    } else {
      status = future->get();
      delete future;
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
  for (StringRef entry : getEnv())
    env.push_back(entry);

  payload->insert({"program", fullTarget.string()});
  payload->insert({"request", "launch"});
  payload->insert({"cwd", cwd.string()});
  payload->insert({"args", json::Array{runArgs}});
  payload->insert({"env", std::move(env)});
  payload->insert({"runInTerminal", rpcTerminal == "dedicated"});

  return invokeRPC(dryRun, rpcPort, *payload);
}
