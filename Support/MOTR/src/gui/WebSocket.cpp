//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// WebSocket.cpp
#include "WebSocket.h"

#define MOTR_JSON_ENABLED 1
#include "motr/EventTree.h"
#include "motr/Hash.h"
#include "motr/Log.h"
#include "motr/Message.h"
#include "motr/Time.h"
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <emscripten/websocket.h>
#include <string_view>
#include <vector>

#include "motr/StringLibrary.h"

extern std::vector<std::string> messages;

using namespace M;
using namespace M::motr::Gui;

struct WebSocket::Impl {
  using Callback = WebSocket::Callback;
  WebSocket &parentref;
  Callback textCallback;
  Callback binaryCallback;
  EMSCRIPTEN_WEBSOCKET_T ws = {};

  State state = State::Disconnected;

  std::string_view url() const { return parentref.url; }

  Impl(WebSocket &parentref, Callback textCallback, Callback binaryCallback);
  ~Impl();

  void initWebSocket();
  void cleanup();

  bool recvText(std::string_view sv);
  bool recvBinary(std::string_view sv);

  void sendText(std::string_view sv);
  void sendBinary(std::string_view sv);
};

void asyncCallImplInitWebSocket(void *userData) {
  WebSocket &websocket = *static_cast<WebSocket *>(userData);
  websocket.impl->initWebSocket();
}

WebSocket::WebSocket(std::string_view url, Callback textCallback,
                     Callback binaryCallback)
    : url{url},
      impl{std::make_unique<Impl>(*this, textCallback, binaryCallback)} {

  // Asynchronously initialize the Emscripten web socket
  // If done synchronously, the Chrome dev tools
  // will not load source maps correctly
  // and debug symbols / files will not load
  emscripten_set_timeout(asyncCallImplInitWebSocket, 100, this);
}

WebSocket::~WebSocket() = default;

WebSocket::Impl::~Impl() {
  cleanup(); //
}

static WebSocket::Impl &refImpl(void *ptr) {
  assert(ptr);
  WebSocket::Impl *impl = static_cast<WebSocket::Impl *>(ptr);
  assert(impl);
  return *impl;
}

void WebSocket::Impl::cleanup() {
  // MOTR_LOG("WebSocket::Impl::cleanup()", "");

  if (ws) {
    // websocket close codes defined at
    // https://datatracker.ietf.org/doc/html/rfc6455#section-7.4.1
    // 1000 is normal closure
    // 1001 is going away
    // 1002 is protocol error
    // 1003 is unacceptable data
    // 1004 is reserved
    // 1005 is no status code
    // 1006 is abnormal closure
    // 1007 is invalid frame payload data
    // 1008 is policy violation
    // 1009 is message too big
    // 1010 is missing extension
    // 1011 is internal server error
    // 1015 is TLS handshake
    emscripten_websocket_close(ws, 1000, "Page unload");
    emscripten_websocket_delete(ws);
    ws = 0;
  }
  state = WebSocket::State::Closed;
}

const char *onPageUnload(int eventType, const void *reserved, void *userData) {
  MOTR_LOG("WebSocket::Impl::onPageUnload(eventType={}, userData={})",
           int(eventType), userData);
  // emscripten_websocket_deinitialize();

  // not safe as the WebSocket has been deleted
  // MOTR_LOG("WebSocket::Impl::onPageUnload()", "");
  refImpl(userData).cleanup();
  return nullptr;
}

EM_BOOL onOpen(int eventType,
               const EmscriptenWebSocketOpenEvent *websocketEvent,
               void *userData) {
  WebSocket::Impl &impl = refImpl(userData);
  impl.state = WebSocket::State::Connected;
  MOTR_LOG("WebSocket[{}] connected", impl.url());
  // MOTR_LOG("WebSocket::Impl::onOpen(eventType={}, websocketEvent={},
  // userData={})",
  //          int(eventType), fmt::ptr(websocketEvent), fmt::ptr(userData));
  EMSCRIPTEN_RESULT result = emscripten_websocket_send_utf8_text(
      websocketEvent->socket, "client_connected");
  if (result) {
    printf("open failed to send response");
  }
  return EM_TRUE;
}

EM_BOOL onError(int eventType,
                const EmscriptenWebSocketErrorEvent *websocketEvent,
                void *userData) {
  refImpl(userData).state = WebSocket::State::Error;
  // MOTR_LOG("WebSocket[{}] error", refImpl(userData).url());
  // MOTR_LOG(
  //     "WebSocket::Impl::onError(eventType={}, websocketEvent={},
  //     userData={})", int(eventType), fmt::ptr(websocketEvent),
  //     fmt::ptr(userData));
  return EM_TRUE;
}

EM_BOOL onClose(int eventType,
                const EmscriptenWebSocketCloseEvent *websocketEvent,
                void *userData) {
  refImpl(userData).cleanup();
  // MOTR_LOG("WebSocket[{}] closed", refImpl(userData).url());
  // MOTR_LOG(
  //     "WebSocket::Impl::onClose(eventType={}, websocketEvent={},
  //     userData={})", int(eventType), fmt::ptr(websocketEvent),
  //     fmt::ptr(userData));
  return EM_TRUE;
}

void removeNullTerminatorFromView(std::string_view &data) {
  // cli WebServer::sendWebSocketText sends
  // the text with a null terminator included in length
  assert(data.size());
  assert(data.back() == '\0');
  if (data.size() > 1 && data.back() == '\0') {
    data.remove_suffix(1);
  }
}

EM_BOOL onMessage(int eventType,
                  const EmscriptenWebSocketMessageEvent *websocketEvent,
                  void *userData) {
  if (!websocketEvent) {
    MOTR_LOG("WebSocket[{}] onMessage() - no websocketEvent",
             refImpl(userData).url());
    return EM_FALSE;
  }

  std::string_view data((const char *)websocketEvent->data,
                        websocketEvent->numBytes);
  if (websocketEvent->isText) {
    removeNullTerminatorFromView(data);
    refImpl(userData).recvText(data);
  } else {
    refImpl(userData).recvBinary(data);
  }
  return EM_TRUE;
}

bool WebSocket::Impl::recvBinary(std::string_view sv) {
  if (binaryCallback) {
    binaryCallback(parentref, sv);
  }
  return binaryCallback != nullptr;
}

bool WebSocket::Impl::recvText(std::string_view sv) {
  if (textCallback) {
    textCallback(parentref, sv);
  }
  return textCallback != nullptr;
}

WebSocket::Impl::Impl(WebSocket &parentref, Callback textCallback,
                      Callback binaryCallback)
    : parentref(parentref), textCallback(textCallback),
      binaryCallback(binaryCallback) {}

void WebSocket::Impl::initWebSocket() {
  EmscriptenWebSocketCreateAttributes ws_attrs = {url().data(), NULL, EM_TRUE};
  // MOTR_LOG("WebSocket[{}] initWebSocket()", url());
  state = WebSocket::State::Connecting;
  ws = emscripten_websocket_new(&ws_attrs);
  if (ws <= 0) {
    MOTR_LOG("WebSocket[{}] initWebSocket() - failed to create websocket",
             url());
    state = WebSocket::State::Error;
    return;
  }

  emscripten_websocket_set_onopen_callback(ws, this, onOpen);
  emscripten_websocket_set_onerror_callback(ws, this, onError);
  emscripten_websocket_set_onclose_callback(ws, this, onClose);
  emscripten_websocket_set_onmessage_callback(ws, this, onMessage);

  emscripten_set_beforeunload_callback(this, onPageUnload);
}

void WebSocket::Impl::sendText(std::string_view sv) {
  if (state != WebSocket::State::Connected || !ws) {
    MOTR_LOG("WebSocket[{}] sendText() - not connected", url());
    return;
  }
  emscripten_websocket_send_utf8_text(ws, sv.data());
}

void WebSocket::Impl::sendBinary(std::string_view sv) {
  if (state != WebSocket::State::Connected || !ws) {
    MOTR_LOG("WebSocket[{}] sendBinary() - not connected", url());
    return;
  }
  emscripten_websocket_send_binary(ws, (void *)(sv.data()), sv.size());
}

void WebSocket::sendText(std::string_view sv) {
  if (impl) {
    impl->sendText(sv);
  }
}

void WebSocket::sendBinary(std::string_view sv) {
  if (impl) {
    impl->sendBinary(sv);
  }
}

WebSocket::State WebSocket::state() const {
  if (impl) {
    return impl->state;
  }
  return WebSocket::State::Disconnected;
}

bool WebSocket::isClosed() const { return state() == WebSocket::State::Closed; }
