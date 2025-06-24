//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_MOTR_GUI_FLAME_GRAPH_RENDERER_H
#define M_MOTR_GUI_FLAME_GRAPH_RENDERER_H

#include "Color.h"
#include "TriangleWindow.h"
#include "motr/EventTree.h"
#include "motr/Time.h"
#include <memory>
#include <optional>
#include <unordered_map>

namespace M::motr::Gui {

struct FlameSpan {
  EventTreeNode *push{};
  EventTreeNode *pop{};
  Time::Range range;
  std::string_view name;
  TagLibrary::Ptr tagLibrary;
  int depth{};
  Color::RGBA color;

  std::string_view getName() const;
  std::optional<std::string_view> locStr();
};

struct FlameWindow {
  EventTreeNode::Ptr node;
  std::unique_ptr<TriangleWindow> triangleWindow;
  std::unordered_map<uint64_t, FlameSpan> spans;

  FlameWindow(EventTreeNode::Ptr node, WGPUDevice device);
  ~FlameWindow() = default;

  FlameWindow(const FlameWindow &) = delete;
  FlameWindow &operator=(const FlameWindow &) = delete;
  FlameWindow(FlameWindow &&) noexcept = delete;
  FlameWindow &operator=(FlameWindow &&) noexcept = delete;

  void regenerate();
  uint32_t getPickValue() const;
  uint64_t getPickId() const;
  FlameSpan *getPickSpan();
};

// Helper functions
bool isSpanPush(EventTreeNode *node);
bool isSpanPop(EventTreeNode *node);
std::unordered_map<uint64_t, FlameSpan>
createFlameSpans(std::vector<EventTreeNode::Ptr> &nodes);
std::pair<int, int>
getFlameGraphDepthRange(const std::unordered_map<uint64_t, FlameSpan> &spans);

} // namespace M::motr::Gui

#endif // M_MOTR_GUI_FLAME_GRAPH_RENDERER_H
