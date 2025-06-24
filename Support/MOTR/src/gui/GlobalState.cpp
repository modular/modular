//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GlobalState.h"
#include "FlameGraphRenderer.h"
#include "WebUtilities.h"
#include "motr/Log.h"
#include "motr/MString.h"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

constexpr const int MOTR_DEFAULT_WEBSOCKET_PORT = 6687;

namespace M::motr::Gui {

GlobalState::~GlobalState() = default;

GlobalState &GlobalState::getSingleton() {
  static GlobalState singleton;
  return singleton;
}

EventTree &GlobalState::getEventTree() {
  static EventTree eventTree(1024 * 1024 * 32);
  return eventTree;
}

StringLibrary &GlobalState::getStringLibrary() {
  return MString::getStringLibrary();
}

TagLibrary &GlobalState::getTagLibrary() {
  static TagLibrary tagLibrary;
  static bool once = false;
  if (!once) {
    once = true;
    tagLibrary.setString({"ProgramName"}, "motr");
    tagLibrary.setString({"foo"}, "bar");
    tagLibrary.setString({"bar"}, "center");
  }
  return tagLibrary;
}

Time::Duration GlobalState::elapsed(Time::Timestamp t) {
  if (t.v == 0)
    t = Time::Timestamp::now();
  return t - programStart;
}

static size_t mailboxSendMessagesCallback(const Message *messages,
                                          size_t count) {
  auto &state = GlobalState::getSingleton();
  std::string_view messagesBinaryStringView{(const char *)messages,
                                            count * sizeof(Message)};
  for (auto &websocket : state.websockets) {
    websocket.sendBinary(messagesBinaryStringView);
  }
  return count;
}

static size_t
mailboxSendStringViewsCallback(const std::vector<std::string_view> &strs) {
  auto &state = GlobalState::getSingleton();
  for (auto &websocket : state.websockets) {
    for (auto &str : strs) {
      websocket.sendText(str);
    }
  }
  return strs.size();
}

void GlobalState::initWebSockets() {
  // Register the callback to send binary Motr messages to all websockets
  ServerOutbox::getQueue().sendMessagesCallback = mailboxSendMessagesCallback;
  ServerOutboxString::getQueue().sendStringViewsCallback =
      mailboxSendStringViewsCallback;

  int port = MOTR_DEFAULT_WEBSOCKET_PORT;
  std::vector<std::pair<int, int>> ranges = {
      {6680, 6689},
  };
  for (auto [start, end] : ranges) {
    for (int port = start; port <= end; port++) {
      std::string websocketUrl = fmt::format("ws://localhost:{}/ws", port);
      MOTR_LOG("Connecting to websocket at {}", websocketUrl);
      this->websockets.emplace_back(websocketUrl, onMsgRecvText,
                                    onMsgRecvBinary);
    }
  }
}

#if 0

  for (int idx = 0; port != 0; idx++) {
    port = getWebsocketPort(idx);
    if (port == 0 && idx == 0)
      port = MOTR_DEFAULT_WEBSOCKET_PORT;

    if (port != 0) {
      std::string websocketUrl = fmt::format("ws://localhost:{}/ws", port);
      MOTR_LOG("Connecting to websocket at {}", websocketUrl);
      this->websockets.emplace_back(websocketUrl, onMsgRecvText, onMsgRecvBinary);
    }
  }
}
#endif

GlobalState &globalState() { return GlobalState::getSingleton(); }

static void logMessage(const Message &msg, std::string_view suffix) {
  return;
  std::string type = toString(msg.type);

  MOTR_LOG("msg[{}] [id=0x{:016x}] [pid=0x{:016x}] [type={}] [flags={}] {}",
           globalState().generation, msg.id, msg.pid, type, toString(msg.flags),
           suffix);
}

bool GlobalState::onMsgRecvBinary(WebSocket &ws, std::string_view sv) {
  assert(sv.size() % sizeof(Message) == 0);
  GlobalState &state = globalState();
  state.generation++;
  EventTree &tree = state.getEventTree();

  const Message *messages = reinterpret_cast<const Message *>(sv.data());
  size_t N = sv.size() / sizeof(Message);

  for (size_t i = 0; i < N; ++i) {
    const Message &msg = messages[i];
    logMessage(msg, ws.url);

    EventTreeNode::Ptr node = tree.addMessage(msg);

    switch (msg.type) {
    case MessageType::Reload:
#ifdef __EMSCRIPTEN__
      // clang-format off
      EM_ASM({ location.reload(); });
      // clang-format on
#endif
      break;
    case MessageType::RPCResult:
      if (msg.flags == MessageFlags::Push)
        state.rpcResults.push_back(node);
      break;
    default:
      break;
    }
  }

  return true;
}

bool GlobalState::onMsgRecvText(WebSocket &ws, std::string_view sv) {
  MString{sv}; // MString will auto-add sv to the SringLibrary singleton
  return true;
}

bool GlobalState::maybeRebuildFlatEventTree() {
  const Time::Timestamp now = Time::Timestamp::now();

  // Check if we need to rebuild the flattened tree
  if (flatEventTree.generation == generation) {
    return false;
  }

  // debounce logic
  {
    static Time::Duration debounceDuration = Time::Duration::fromSeconds(1);
    const Time::Duration duration = now - flatEventTree.timestamp;
    if (duration < debounceDuration) {
      return false;
    }
  }

  flatEventTree.nodes = getEventTree().getAllNodes();
  flatEventTree.generation = generation;
  flatEventTree.timestamp = now;

  return true;
}

std::vector<EventTreeNode::Ptr> &GlobalState::getFlatEventTree() {
  maybeRebuildFlatEventTree();
  return flatEventTree.nodes;
}

} // namespace M::motr::Gui
