//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "WebSocket.h"
#include "WebSocketAsync.h"
#include "motr/Log.h"
#include <chrono>
#include <thread>

// Enforce no-exceptions and no-RTTI requirements
#ifdef __cpp_exceptions
#error "This code requires -fno-exceptions. Exceptions are not supported."
#endif

#ifdef __GXX_RTTI
#error "This code requires -fno-rtti. RTTI is not supported."
#endif

namespace M::motr::Gui {

// Example: Simple async function call
void callGetHostInfoAsync(WebSocket &websocket) {
  auto task = getHostInfo(websocket, false);

  // Poll until complete (or integrate with your event loop)
  while (!task.isDone()) {
    // Process other events, yield, etc.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto result = task.get();
  if (result.hasValue()) {
    auto &info = result.value();
    MOTR_LOG("Host info: hostname={}, uname={}", info.hostname, info.uname);
    MOTR_LOG("IPv4: {}.{}.{}.{}", info.ipv4[0], info.ipv4[1], info.ipv4[2],
             info.ipv4[3]);
  } else {
    MOTR_LOG("Failed to get host info: error={}",
             static_cast<int>(result.error()));
  }
}

// Example: Generic request/response
void callCustomRequestAsync(WebSocket &websocket) {
  auto task = sendRequestAndWait(websocket, "get_system_stats",
                                 "system_stats_response");

  while (!task.isDone()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto result = task.get();
  if (result.hasValue()) {
    MOTR_LOG("System stats response: {}", result.value());
  } else {
    MOTR_LOG("Failed to get system stats: error={}",
             static_cast<int>(result.error()));
  }
}

// Example: Multiple concurrent requests
void callMultipleRequestsAsync(WebSocket &websocket) {
  auto hostInfoTask = getHostInfo(websocket, true);
  auto statsTask = sendRequestAndWait(websocket, "get_system_stats",
                                      "system_stats_response");

  // Wait for both to complete
  while (!hostInfoTask.isDone() || !statsTask.isDone()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto hostResult = hostInfoTask.get();
  auto statsResult = statsTask.get();

  if (hostResult.hasValue()) {
    MOTR_LOG("Got host info: {}", hostResult.value().hostname);
  }

  if (statsResult.hasValue()) {
    MOTR_LOG("Got stats: {}", statsResult.value());
  }
}

} // namespace M::motr::Gui
