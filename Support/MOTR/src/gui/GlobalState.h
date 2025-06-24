//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_GLOBAL_STATE_H
#define M_MOTR_GUI_GLOBAL_STATE_H

#include "LayoutLibrary.h"
#include "LayoutNode.h"
#include "RenderWindow.h"
#include "WebSocket.h"
#include "motr/EventTree.h"
#include "motr/StringLibrary.h"
#include "motr/TagLibrary.h"
#include "motr/Time.h"

#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace M::motr::Gui {

// Forward declaration only - no unordered_map of incomplete type
struct FlameWindow;

struct GlobalState {
  ~GlobalState();
  int generation = 0;
  uint64_t highlightId = 0;
  std::list<WebSocket> websockets;

  std::unique_ptr<RenderWindow> renderWindow;
  std::map<std::string, WindowLayoutNode::Ptr> layoutWindows;
  WindowLayoutNode::Ptr rootLayout;
  std::vector<uint64_t> pickIds;

  // Use forward declaration with unique_ptr - no complete type needed
  std::unordered_map<EventTreeNode::Ptr, std::unique_ptr<FlameWindow>>
      flameWindows;

  bool show_demo_window = false;
  Time::Timestamp programStart = Time::Timestamp::now();

  std::vector<EventTreeNode::Ptr> rpcResults;

  Time::Timestamp frameStart;

  struct FlatEventTree {
    Time::Timestamp timestamp;
    int generation = 0;
    std::vector<EventTreeNode::Ptr> nodes;
  };

  FlatEventTree flatEventTree;
  bool maybeRebuildFlatEventTree();
  std::vector<EventTreeNode::Ptr> &getFlatEventTree();

  // Static methods
  static GlobalState &getSingleton();
  static bool onMsgRecvText(WebSocket &ws, std::string_view sv);
  static bool onMsgRecvBinary(WebSocket &ws, std::string_view sv);

  // Instance methods
  void initWebSockets();

  // Accessors
  LayoutLibrary &getLayoutLibrary() { return LayoutLibrary::instance(); }
  EventTree &getEventTree();
  StringLibrary &getStringLibrary();
  TagLibrary &getTagLibrary();
  Time::Duration elapsed(Time::Timestamp t = Time::Timestamp());
};

GlobalState &globalState();

template <int T>
inline bool checkGeneration() {
  auto &state = globalState();
  static int generation = 0;
  if (generation == state.generation)
    return false;
  generation = state.generation;
  return true;
}

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_GLOBAL_STATE_H
