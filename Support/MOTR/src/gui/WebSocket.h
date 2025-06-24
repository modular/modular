//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_GUI_WEBSOCKET_H
#define MOTR_GUI_WEBSOCKET_H

#include <functional>
#include <string>
#include <string_view>

namespace M::motr::Gui {

struct WebSocket {
  using Callback = std::function<bool(WebSocket &, std::string_view)>;
  WebSocket(std::string_view url, Callback textCallback,
            Callback binaryCallback);
  ~WebSocket();

  WebSocket(const WebSocket &) = delete;
  WebSocket &operator=(const WebSocket &) = delete;

  WebSocket(WebSocket &&other) = delete;
  WebSocket &operator=(WebSocket &&other) = delete;

  void sendText(std::string_view sv);
  void sendBinary(std::string_view sv);

  std::string url;

  enum class State {
    Disconnected,
    Connecting,
    Connected,
    Error,
    Closed,
  };

  State state() const;
  bool isClosed() const;

  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace M::motr::Gui

#endif // MOTR_GUI_WEBSOCKET_H
