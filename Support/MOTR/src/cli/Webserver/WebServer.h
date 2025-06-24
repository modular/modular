//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_WEBSERVER_H
#define MOTR_WEBSERVER_H
#include "Config/ConfigFile.h"
#include "nlohmann/json.hpp"
#include <civetweb.h>
#include <string>
#include <vector>

struct WebSocketClient {
  const struct mg_connection *conn;
  struct mg_connection *downcast() const {
    return const_cast<struct mg_connection *>(conn);
  }
};

struct WebServer {
  WebServer(const M::motr::ServerConfig &config);
  ~WebServer();
  void start();
  void stop();

  void addWebSocketClient(const struct mg_connection *conn);
  void removeWebSocketClient(const struct mg_connection *conn);
  void sendWebsocketMessage(const nlohmann::json &js);
  void sendWebsocketMessage(const std::string &message);
  void sendWebsocketBinary(std::string_view msg);
  void sendWebsocketText(std::string_view msg);
  struct mg_context *ctx;
  std::vector<WebSocketClient> wsClients;
  M::motr::ServerConfig config;
};
#endif
