//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "FlameGraphRenderer.h"
#include "Color.h"
#include "GlobalState.h"
#include "motr/Log.h"
#include "motr/MString.h"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

namespace M::motr::Gui {

bool isSpanPush(EventTreeNode *node) {
  return node && node->message.flags == MessageFlags::Push &&
         node->message.type == MessageType::Span;
}

bool isSpanPop(EventTreeNode *node) {
  return node->message.flags == MessageFlags::Pop &&
         node->message.type == MessageType::Span;
}

std::string_view FlameSpan::getName() const {
  std::string_view name = tagLibrary->getString({"TraceName"});
  if (!name.empty())
    return name;
  if (push && push->message.type == MessageType::Process) {
    auto str = fmt::format("Process {}", push->message.procid);
    MString mstr{str};
    return mstr.sv();
  }
  if (push && push->message.type == MessageType::Thread) {
    auto thread_id = tagLibrary->getU64({"ThreadId"});
    if (thread_id) {
      auto str = fmt::format("Thread id={}", thread_id);
      MString mstr{str};
      return mstr.sv();
    }
  }
  return "<unknown>";
}

std::optional<std::string_view> FlameSpan::locStr() {
  auto file = tagLibrary->getString({"SourceFile"});
  auto line = tagLibrary->getOptionalU64({"SourceLine"});
  if (!file.empty() && line) {
    auto str = fmt::format("{}:{}", file, *line);
    return globalState().getStringLibrary().addString(str);
  }
  return std::nullopt;
}

std::unordered_map<uint64_t, FlameSpan>
createFlameSpans(std::vector<EventTreeNode::Ptr> &nodes) {
  std::unordered_map<uint64_t, FlameSpan> spans;
  for (auto &nodeSharedPtr : nodes) {
    auto node = nodeSharedPtr.get();
    Message &msg = node->message;
    bool is_push = isSpanPush(node);
    bool is_pop = isSpanPop(node);
    if (false) { // disabled to bypass program and thread
      is_push = msg.flags == MessageFlags::Push;
      is_pop = msg.flags == MessageFlags::Pop;
    }
    if (!is_push && !is_pop)
      continue;

    if (is_push) {
      FlameSpan &span = spans[msg.id];
      span.push = node;
      span.range.add(msg.ts);
      span.tagLibrary = TagLibrary::create(*node);
      span.depth = node->numAncestors();
      using namespace Color::Palette;
      span.color = getColorAt<TwelveBitRainbow>(span.depth);
    } else {
      FlameSpan &span = spans[msg.pid];
      span.pop = node;
      span.range.add(msg.ts);
    }
  }

  return spans;
}

std::pair<int, int>
getFlameGraphDepthRange(const std::unordered_map<uint64_t, FlameSpan> &spans) {
  int min_depth = std::numeric_limits<int>::max();
  int max_depth = std::numeric_limits<int>::min();
  for (auto &[id, span] : spans) {
    min_depth = std::min(min_depth, span.depth);
  }
  return {min_depth, max_depth};
}

FlameWindow::FlameWindow(EventTreeNode::Ptr node, WGPUDevice device)
    : node(node), triangleWindow(std::make_unique<TriangleWindow>(device, "")) {
}

uint32_t FlameWindow::getPickValue() const {
  return triangleWindow->trianglePipeline.pickValue;
}

uint64_t FlameWindow::getPickId() const {
  auto value = getPickValue();
  if (!value)
    return 0;
  auto &state = globalState();
  if (value >= state.pickIds.size())
    return 0;
  return state.pickIds[value];
}

FlameSpan *FlameWindow::getPickSpan() {
  auto id = getPickId();
  if (!id)
    return nullptr;
  auto it = spans.find(id);
  if (it == spans.end())
    return nullptr;
  return &it->second;
}

void FlameWindow::regenerate() {
  auto &state = globalState();

  auto nodes_dfs =
      node->getDescendants<EventTreeNode::TraverseMode::DFSPreOrder>();

  if (!nodes_dfs.empty() &&
      nodes_dfs.front()->message.type == MessageType::Process) {
    auto processNode = nodes_dfs.front();
    triangleWindow->title =
        fmt::format("Process {}", processNode->message.procid);
  }

  spans = createFlameSpans(nodes_dfs);

  auto depth_range = getFlameGraphDepthRange(spans);

  Time::Range time_range;
  for (auto &[id, span] : spans) {
    time_range.add(span.range);
  }

  auto precision = Time::Precision::Seconds;

  double scale = time_range.scale(precision);
  double bias = time_range.bias(precision);

  scale = 2.0 * scale;
  bias = 2.0 * bias - 1.0;

  double span_height = 1.0 / 5.0;

  auto &trianglePipeline = triangleWindow->trianglePipeline;
  trianglePipeline.vertices.clear();

  auto &pickIds = state.pickIds;

  float one_pixel_wide = 2.0 / trianglePipeline.width;
  float one_pixel_high = 2.0 / trianglePipeline.height;
  span_height = 40 * one_pixel_high;

  for (auto &[id, span] : spans) {
    double t0 = span.range.start.to(precision);
    double t1 = span.range.end.to(precision);

    double x0 = t0 * scale + bias;
    double x1 = t1 * scale + bias;
    x0 += one_pixel_wide;
    x1 -= one_pixel_wide;

    int adjusted_depth = span.depth - depth_range.first;

    double y0 = 1.0 - adjusted_depth * span_height;
    double y1 = y0 - span_height;
    y0 -= one_pixel_high;
    y1 += one_pixel_high;

    using namespace Color::Palette;
    auto color = getColorAt<TwelveBitRainbow>(span.depth);

    auto pickValue = pickIds.size();
    pickIds.push_back(span.push->message.id);

    glm::vec3 colorVec = {color.r, color.g, color.b};
    trianglePipeline.addRect(x0, y0, x1, y1, colorVec, pickValue);
  }
}

} // namespace M::motr::Gui
