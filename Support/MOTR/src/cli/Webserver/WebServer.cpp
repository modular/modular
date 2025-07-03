//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#define MOTR_JSON_ENABLED
#include "WebServer.h"
#include "HostInfo.h"
#include "civetweb.h"
#include "motr/EventTree.h"
#include "motr/Hash.h"
#include "motr/Log.h"
#include "motr/MString.h"
#include "motr/RPC.h"
#include "motr/RPCMailbox.h"
#include "motr/Tags.h"
#include "motr/Time.h"
#include "motr/Types/Types.h"
#include "motr/motr.h"

#include "nlohmann/json.hpp"
#include <thread>
#include <vector>

using namespace M;

WebServer &webserverRef(void *ptr) {
  assert(ptr != nullptr);
  return *static_cast<WebServer *>(ptr);
}

static int websocketConnectHandler(const struct mg_connection *conn,
                                   void *cbdata) {
  auto &webserver = webserverRef(cbdata);
  const auto *req_info = mg_get_request_info(conn);
  std::string ip = req_info->remote_addr;
  MOTR_LOG("WebServer[{}] client websocket connected: ip={}, port={}, path={}",
           webserver.config.name, ip, req_info->remote_port,
           req_info->request_uri);

  webserver.addWebSocketClient(conn);
  return 0;
}

static void websocketReadyHandler(struct mg_connection *conn, void *cbdata) {
  auto &webserver = webserverRef(cbdata);
  MOTR_LOG("WebServer[{}] websocket ready", webserver.config.name);
}

static void handleWebsocketBinary(WebServer &webserver,
                                  std::string_view binary) {
  using namespace M::motr;

  if (binary.size() % sizeof(Message) != 0) {
    MOTR_LOG("WebServer[{}] handleWebsocketBinary: invalid binary size",
             webserver.config.name);
    return;
  }
  const size_t count = binary.size() / sizeof(Message);
  const Message *messages = reinterpret_cast<const Message *>(binary.data());
  EventTree &eventTree = EventTree::getSingleton();
  for (size_t i = 0; i < count; i++)
    eventTree.addMessage(messages[i]);
}

static void handleWebsocketMessage(WebServer &webserver,
                                   std::string_view message) {
  MOTR_LOG("WebServer[{}] handleWebsocketMessage: {}", webserver.config.name,
           message);
  using namespace M::motr;

  MString msgMStr{message};

  /*
  constexpr Hash::Value getHostInfoKey{"get_host_info"};
  MOTR_LOG("WebServer[{}] websocket message key=0x{:016x},
  get_host_info=0x{:016x}", webserver.config.name, msgMStr.hash.v,
  getHostInfoKey.v);

  switch (msgMStr.hash.v) {
  case Hash::Value("get_host_info").v:
    M::motr::RPC::sendRPCResult(getHostInfo());
    M::motr::RPC::sendRPCResult(getMotrServerInfo());
    break;
  default:
    MOTR_LOG("WebServer[{}] handleWebsocketMessage: unknown message",
             webserver.config.name);
    break;
  }
  */
}

constexpr const char *opcodeToString(int opcode) {
  switch (opcode) {
  case MG_WEBSOCKET_OPCODE_CONTINUATION:
    return "CONTINUATION";
  case MG_WEBSOCKET_OPCODE_TEXT:
    return "TEXT";
  case MG_WEBSOCKET_OPCODE_BINARY:
    return "BINARY";
  case MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE:
    return "CONNECTION_CLOSE";
  case MG_WEBSOCKET_OPCODE_PING:
    return "PING";
  case MG_WEBSOCKET_OPCODE_PONG:
    return "PONG";
  default:
    return "UNKNOWN";
  }
}

static int websocketDataHandler(struct mg_connection *conn, int operation,
                                char *data, size_t len, void *cbdata) {
  auto &webserver = webserverRef(cbdata);

  std::string_view message(data, len);
  /* WebSocket OpcCodes, from http://tools.ietf.org/html/rfc6455 */
  int opcode = operation & 0xF;

  if (true) {
    const mg_request_info *req_info = mg_get_request_info(conn);
    std::string connectionInfo = fmt::format(
        "server={}, client_ip={}, client_port={}, path={}, method={}",
        webserver.config.name, req_info->remote_addr, req_info->remote_port,
        req_info->request_uri, req_info->request_method);
    MOTR_LOG("{}, opcode[0x{:02x}]={}, len={}", connectionInfo, opcode,
             opcodeToString(opcode), len);
  }

  switch (opcode) {
  case MG_WEBSOCKET_OPCODE_CONTINUATION:
    assert(false);
    MOTR_LOG("ERROR: {}", "NOT IMPLEMENTED: MG_WEBSOCKET_OPCODE_CONTINUATION");
    return 1;
    break;
  case MG_WEBSOCKET_OPCODE_TEXT:
    handleWebsocketMessage(webserver, message);
    return 1;
    break;
  case MG_WEBSOCKET_OPCODE_BINARY:
    handleWebsocketBinary(webserver, message);
    return 1;
    break;
  case MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE:
    return 0;
    break;
  case MG_WEBSOCKET_OPCODE_PING:
    return 1;
    break;
  case MG_WEBSOCKET_OPCODE_PONG:
    return 1;
    break;
  default:
    assert(false);
    MOTR_LOG("ERROR: {}", "NOT IMPLEMENTED: MG_WEBSOCKET_OPCODE_UNKNOWN");
    return 1;
    break;
  }
  return 1;
}

static void websocketCloseHandler(const struct mg_connection *conn,
                                  void *cbdata) {
  auto &webserver = webserverRef(cbdata);
  MOTR_LOG("WebServer[{}] websocket close", webserver.config.name);
  webserver.removeWebSocketClient(conn);
}

static int authHandler(struct mg_connection *conn, void *cbdata) {
  auto &webserver = webserverRef(cbdata);
  MOTR_LOG("WebServer[{}] authHandler", webserver.config.name);
  return 1;
}

WebServer::WebServer(const motr::ServerConfig &config)
    : ctx(nullptr), config(config) {
  start();
}

WebServer::~WebServer() { stop(); }

void WebServer::start() {
  std::string num_threads = std::to_string(config.numThreads);
  const char *options[] = {
      // clang-format off
    "document_root", config.documentRoot.c_str(),
      "listening_ports", config.listeningPorts.c_str(),
      "num_threads", num_threads.c_str(),
      "access_control_allow_origin", config.accessControlAllowOrigin.c_str(),
      "enable_auth_domain_check", "no",
      "enable_directory_listing", "no",
      nullptr,
      // clang-format on
  };

  ctx = mg_start(nullptr, nullptr, options);
  if (!ctx) {
    MOTR_LOG("Failed to start WebServer[{}]", config.name);
    return;
  }

  mg_set_websocket_handler(ctx, "/ws", websocketConnectHandler,
                           websocketReadyHandler, websocketDataHandler,
                           websocketCloseHandler, this);

  // mg_set_auth_handler(ctx, "/ws", authHandler, this);
  // mg_set_auth_handler(ctx, "/", authHandler, this);
}

void WebServer::stop() {
  MOTR_LOG("WebServer[{}] stopping", config.name);
  if (ctx) {
    mg_stop(ctx);
    ctx = nullptr;
  }
  wsClients.clear();
}

void WebServer::addWebSocketClient(const mg_connection *conn) {
  wsClients.push_back({conn});
  MOTR_LOG("WebServer[{}] client connected, total: {}", config.name,
           wsClients.size());
}

void WebServer::removeWebSocketClient(const mg_connection *conn) {
  std::erase_if(wsClients, [conn](const WebSocketClient &client) {
    return client.conn == conn;
  });
  MOTR_LOG("WebServer[{}] client disconnected, remaining: {}", config.name,
           wsClients.size());
}

void WebServer::sendWebsocketMessage(const nlohmann::json &js) {
  sendWebsocketMessage(js.dump());
}

void WebServer::sendWebsocketMessage(const std::string &message) {
  sendWebsocketText(message);
}

void WebServer::sendWebsocketBinary(std::string_view msg) {
  int nclients = wsClients.size();
  for (const auto &client : wsClients) {
    mg_websocket_write(client.downcast(), MG_WEBSOCKET_OPCODE_BINARY,
                       msg.data(), msg.size());
  }
}

void WebServer::sendWebsocketText(std::string_view msg) {
  int nclients = wsClients.size();
  for (const auto &client : wsClients) {
    mg_websocket_write(client.downcast(), MG_WEBSOCKET_OPCODE_TEXT, msg.data(),
                       msg.length());
  }
}
