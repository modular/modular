//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LayoutNode.h"
#include "Color.h"
#include "LayoutNodeJson.h"
#include "PlotLayoutNode.h"
#include "RefLayoutNode.h"
#include "attr/AttributeResolver.h"
#include "imgui.h"
#include "motr/JSON.h" // Please include first to hook up the error handler
#include "yoga/Yoga.h"
#include <fmt/args.h>
#include <fmt/format.h>
#include <motr/TagLibrary.h>
#include <string>
#include <tweeny.h>
#include <unordered_map>

/*

+-----------------------------------+
|              margin               |
|   +---------------------------+   |
|   |          border           |   |
|   |   +-------------------+   |   |
|   |   |      padding      |   |   |
|   |   |   +-----------+   |   |   |
|   |   |   |           |   |   |   |
|   |   |   |  content  |   |   |   |
|   |   |   |           |   |   |   |
|   |   |   +-----------+   |   |   |
|   |   |                   |   |   |
|   |   +-------------------+   |   |
|   |                           |   |
|   +---------------------------+   |
|                                   |
+-----------------------------------+

*/

namespace M::motr::Gui {

LayoutNode::LayoutNode(std::shared_ptr<LayoutNode> parent)
    : parent(parent), node(YGNodeNew()) {}

size_t LayoutNode::appendChild(Ptr child) {
  if (child) {
    assert(child->parent.expired() && "Child already has a parent");
    child->parent = shared_from_this();
    YGNodeInsertChild(node, child->node, children.size());
    children.push_back(child);
  }
  assert(children.size() == YGNodeGetChildCount(node));

  return children.size() - 1;
}

LayoutNode::~LayoutNode() { YGNodeFree(node); }

LayoutNode::Ptr LayoutNode::getChildByName(std::string_view name) const {
  for (auto &child : children) {
    if (child->get_name() == name)
      return child;
  }
  return nullptr;
}

const Attribute::DynamicValue *LayoutNode::getAttr(const MString &key) const {
  auto itr = attrs.find(key);
  if (itr == attrs.end())
    return nullptr;
  return &itr->second;
}

Attribute::DynamicValue *LayoutNode::getAttr(const MString &key) {
  auto itr = attrs.find(key);
  if (itr == attrs.end())
    return nullptr;
  return &itr->second;
}

YGNodeRef cloneStyle(YGNodeRef srcNode, YGNodeRef dstNode) {
  if (!srcNode || !dstNode)
    return dstNode;

  // Direction
  YGNodeStyleSetDirection(dstNode, YGNodeStyleGetDirection(srcNode));

  // Dimensions
  YGValue width = YGNodeStyleGetWidth(srcNode);
  YGValue height = YGNodeStyleGetHeight(srcNode);
  if (width.unit == YGUnitPercent)
    YGNodeStyleSetWidthPercent(dstNode, width.value);
  else if (width.unit == YGUnitPoint)
    YGNodeStyleSetWidth(dstNode, width.value);
  else
    YGNodeStyleSetWidthAuto(dstNode);

  if (height.unit == YGUnitPercent)
    YGNodeStyleSetHeightPercent(dstNode, height.value);
  else if (height.unit == YGUnitPoint)
    YGNodeStyleSetHeight(dstNode, height.value);
  else
    YGNodeStyleSetHeightAuto(dstNode);

  // Min dimensions
  YGValue minWidth = YGNodeStyleGetMinWidth(srcNode);
  YGValue minHeight = YGNodeStyleGetMinHeight(srcNode);
  if (minWidth.unit == YGUnitPercent)
    YGNodeStyleSetMinWidthPercent(dstNode, minWidth.value);
  else if (minWidth.unit == YGUnitPoint)
    YGNodeStyleSetMinWidth(dstNode, minWidth.value);

  if (minHeight.unit == YGUnitPercent)
    YGNodeStyleSetMinHeightPercent(dstNode, minHeight.value);
  else if (minHeight.unit == YGUnitPoint)
    YGNodeStyleSetMinHeight(dstNode, minHeight.value);

  // Max dimensions
  YGValue maxWidth = YGNodeStyleGetMaxWidth(srcNode);
  YGValue maxHeight = YGNodeStyleGetMaxHeight(srcNode);
  if (maxWidth.unit == YGUnitPercent)
    YGNodeStyleSetMaxWidthPercent(dstNode, maxWidth.value);
  else if (maxWidth.unit == YGUnitPoint)
    YGNodeStyleSetMaxWidth(dstNode, maxWidth.value);

  if (maxHeight.unit == YGUnitPercent)
    YGNodeStyleSetMaxHeightPercent(dstNode, maxHeight.value);
  else if (maxHeight.unit == YGUnitPoint)
    YGNodeStyleSetMaxHeight(dstNode, maxHeight.value);

  // Position
  YGNodeStyleSetPositionType(dstNode, YGNodeStyleGetPositionType(srcNode));
  for (YGEdge edge = YGEdgeLeft; edge <= YGEdgeAll; edge = (YGEdge)(edge + 1)) {
    YGValue position = YGNodeStyleGetPosition(srcNode, edge);
    if (position.unit == YGUnitPercent)
      YGNodeStyleSetPositionPercent(dstNode, edge, position.value);
    else if (position.unit == YGUnitPoint)
      YGNodeStyleSetPosition(dstNode, edge, position.value);
  }

  // Margin
  for (YGEdge edge = YGEdgeLeft; edge <= YGEdgeAll; edge = (YGEdge)(edge + 1)) {
    YGValue margin = YGNodeStyleGetMargin(srcNode, edge);
    if (margin.unit == YGUnitPercent)
      YGNodeStyleSetMarginPercent(dstNode, edge, margin.value);
    else if (margin.unit == YGUnitPoint)
      YGNodeStyleSetMargin(dstNode, edge, margin.value);
    else if (margin.unit == YGUnitAuto)
      YGNodeStyleSetMarginAuto(dstNode, edge);
  }

  // Padding
  for (YGEdge edge = YGEdgeLeft; edge <= YGEdgeAll; edge = (YGEdge)(edge + 1)) {
    YGValue padding = YGNodeStyleGetPadding(srcNode, edge);
    if (padding.unit == YGUnitPercent)
      YGNodeStyleSetPaddingPercent(dstNode, edge, padding.value);
    else if (padding.unit == YGUnitPoint)
      YGNodeStyleSetPadding(dstNode, edge, padding.value);
  }

  // Border
  for (YGEdge edge = YGEdgeLeft; edge <= YGEdgeAll; edge = (YGEdge)(edge + 1)) {
    float border = YGNodeStyleGetBorder(srcNode, edge);
    YGNodeStyleSetBorder(dstNode, edge, border);
  }

  // Flex properties
  YGNodeStyleSetFlex(dstNode, YGNodeStyleGetFlex(srcNode));
  YGNodeStyleSetFlexGrow(dstNode, YGNodeStyleGetFlexGrow(srcNode));
  YGNodeStyleSetFlexShrink(dstNode, YGNodeStyleGetFlexShrink(srcNode));

  YGValue flexBasis = YGNodeStyleGetFlexBasis(srcNode);
  if (flexBasis.unit == YGUnitPercent)
    YGNodeStyleSetFlexBasisPercent(dstNode, flexBasis.value);
  else if (flexBasis.unit == YGUnitPoint)
    YGNodeStyleSetFlexBasis(dstNode, flexBasis.value);
  else
    YGNodeStyleSetFlexBasisAuto(dstNode);

  YGNodeStyleSetFlexDirection(dstNode, YGNodeStyleGetFlexDirection(srcNode));
  YGNodeStyleSetFlexWrap(dstNode, YGNodeStyleGetFlexWrap(srcNode));

  // Alignment
  YGNodeStyleSetJustifyContent(dstNode, YGNodeStyleGetJustifyContent(srcNode));
  YGNodeStyleSetAlignItems(dstNode, YGNodeStyleGetAlignItems(srcNode));
  YGNodeStyleSetAlignSelf(dstNode, YGNodeStyleGetAlignSelf(srcNode));
  YGNodeStyleSetAlignContent(dstNode, YGNodeStyleGetAlignContent(srcNode));

  // Display
  YGNodeStyleSetDisplay(dstNode, YGNodeStyleGetDisplay(srcNode));
  YGNodeStyleSetOverflow(dstNode, YGNodeStyleGetOverflow(srcNode));

  // Aspect ratio
  float aspectRatio = YGNodeStyleGetAspectRatio(srcNode);
  if (!isnan(aspectRatio)) {
    YGNodeStyleSetAspectRatio(dstNode, aspectRatio);
  }

  return dstNode;
}

template <typename T>
static LayoutNode::Ptr cloneLayoutNode(const T *srcNode) {
  auto dstNode = std::make_shared<T>(nullptr);
  dstNode->type = srcNode->type;
  dstNode->fmt = srcNode->fmt;
  dstNode->args = srcNode->args;

  cloneStyle(srcNode->node, dstNode->node);

  // Clone children recursively
  for (const auto &child : srcNode->children) {
    auto childClone = child->clone();
    dstNode->appendChild(childClone);
  }

  return dstNode;
}

LayoutNode::Ptr LayoutNode::clone() const { return cloneLayoutNode(this); }

WindowLayoutNode::WindowLayoutNode(std::shared_ptr<LayoutNode> parent)
    : LayoutNode(parent) {
  type = LayoutNodeType::Window;

  YGNodeStyleSetWidthPercent(node, 100);
  YGNodeStyleSetHeightPercent(node, 100);

  // Center the root node within the window node
  YGNodeStyleSetJustifyContent(node, YGJustifyCenter);
  YGNodeStyleSetAlignItems(node, YGAlignCenter);
}

bool copyAttribute(const LayoutNode &src, LayoutNode &dst, MString key) {
  if (const Attribute::DynamicValue *dv = src.getAttr(key); dv) {
    dst.attrs.insert_or_assign(key, *dv);
    return true;
  }
  return false;
}

bool LayoutNode::copyAttrFrom(const LayoutNode &src, MString key) {
  if (const Attribute::DynamicValue *dv = src.getAttr(key); dv) {
    this->attrs.insert_or_assign(key, *dv);
    return true;
  }
  return false;
}
std::shared_ptr<WindowLayoutNode>
WindowLayoutNode::wrap(std::shared_ptr<LayoutNode> node) {
  auto windowLayoutNode = std::make_shared<WindowLayoutNode>(nullptr);

  // Copy attribute "name" as a dynamic value before resolution
  windowLayoutNode->copyAttrFrom(*node, {"name"});

  windowLayoutNode->appendChild(node);

  return windowLayoutNode;
}

std::shared_ptr<WindowLayoutNode>
WindowLayoutNode::wrapJsonStringView(std::string_view json) {
  return wrap(LayoutNode::makeFromJsonStrView(json));
}

void LayoutNode::relayout() {
  YGNodeCalculateLayout(node, YGUndefined, YGUndefined, YGDirectionLTR);
}

ImU32 getNextColor(bool reset = false) {
  static int colorIndex = -1;
  if (reset) {
    colorIndex = -1;
  }

  auto ncolors = Color::Palette::twelveBitRainbowArray.size();
  colorIndex = (colorIndex + 1) % ncolors;
  uint32_t rgba = uint32_t(Color::Palette::twelveBitRainbowArray[colorIndex]);
  uint32_t argb = Color::RGBA32toARGB32(rgba);
  return argb;
}

ImU32 getColorAtDepth(int depth, int steps) {
  float normalizedDepth = static_cast<float>(depth) / steps;
  float sineValue =
      (sin(normalizedDepth * M_PI) + 1) / 2; // Normalize sine wave to [0, 1]
  int grayscale = static_cast<int>(sineValue * 255);
  return IM_COL32(grayscale, grayscale, grayscale, 255);
}

bool isaTagExpr(std::string_view arg) {
  return arg.substr(0, 2) == "${" && arg.back() == '}';
}

std::string_view getTagExpr(std::string_view arg) {
  if (isaTagExpr(arg))
    return arg.substr(2, arg.size() - 3);
  return {};
}

std::string evalTagExpr(TagLibrary *tagLibrary, std::string_view arg) {
  if (!tagLibrary)
    return std::string(arg);

  auto expr = getTagExpr(arg);
  if (expr.empty())
    return std::string(arg);

  std::string result{expr};

  if (isaTagExpr(result)) {
    result = evalTagExpr(tagLibrary, result);
  }

  Hash::Value hash(result);
  if (tagLibrary->hasTagStr(hash)) {
    result = tagLibrary->getString(hash);
  } else if (tagLibrary->hasTagInt(hash)) {
    // todo: handle int formatting
    result = fmt::format("{}", tagLibrary->getU64(hash));
  } else {
    result = std::string(arg);
  }

  return result;
}

std::string interpolateText(TagLibrary *tagLibrary, std::string_view fmt,
                            const std::vector<std::string> &args) {
  assert(!fmt.empty() && "fmt is empty");

  // Format the text using fmt and args
  fmt::dynamic_format_arg_store<fmt::format_context> store;
  for (const auto &arg : args) {
    auto str = evalTagExpr(tagLibrary, arg);
    store.push_back(str);
  }

  // Use vformat to format the string
  std::string formattedText = fmt::vformat(fmt, store);

  return formattedText;
}

void LayoutNode::setContextPosition(DrawContext &context,
                                    bool updateImGuiCursor) {
  context.x = context.offset.x + YGNodeLayoutGetLeft(node);
  context.y = context.offset.y + YGNodeLayoutGetTop(node);
  context.width = YGNodeLayoutGetWidth(node);
  context.height = YGNodeLayoutGetHeight(node);
  if (updateImGuiCursor) {
    ImGui::SetCursorScreenPos(ImVec2(context.x, context.y));
  }
}

void LayoutNode::draw(DrawContext &context) {
  if (context.tagLibrary)
    Attribute::resolveNode(*this, *context.tagLibrary);
  setContextPosition(context, true);
  ImU32 color = getColorAtDepth(context.depth, 10);
  if (const Attribute::DynamicValue *dv = getAttr({"backgroundColor"}); dv) {
    auto rgba = get_backgroundColor();
    color = IM_COL32(rgba.r, rgba.g, rgba.b, rgba.a);
  }

  ImU32 borderColor = IM_COL32(0, 0, 0, 255); // Black color

  // Draw filled rectangle
  context.draw_list->AddRectFilled(
      ImVec2(context.x, context.y),
      ImVec2(context.x + context.width, context.y + context.height), color);

  // Draw border
  context.draw_list->AddRect(
      ImVec2(context.x, context.y),
      ImVec2(context.x + context.width, context.y + context.height),
      borderColor, 0.0f, 0, 1.0f);
}

void LayoutNode::traverse(DrawContext &context) {
  draw(context);
  ImVec2 childOffset = context.offset;
  childOffset.x += YGNodeLayoutGetLeft(node);
  childOffset.y += YGNodeLayoutGetTop(node);
  for (auto &childLayoutNode : children) {
    DrawContext childContext = context;
    childContext.offset = childOffset;
    childContext.depth++;
    childLayoutNode->traverse(childContext);
  }
}

void WindowLayoutNode::draw(DrawContext &context) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  auto windowFlags =
      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing;

  const auto &name = get_name();
  ImGui::Begin(name.c_str(), nullptr, windowFlags);
  ImGui::SetWindowSize(ImVec2(400, 600), ImGuiCond_Once);

  ImVec2 size = ImGui::GetWindowSize();
  float titleBarHeight = ImGui::GetFrameHeight();
  float drawableHeight = size.y - titleBarHeight;

  ImGui::SetCursorPosY(titleBarHeight);
  ImGui::SetCursorPosX(0);

  YGNodeStyleSetWidth(node, size.x);
  YGNodeStyleSetHeight(node, drawableHeight);
  relayout();

  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();

  getNextColor(true);

  assert(children.size() == 1 &&
         "WindowLayoutNode should have exactly one child");

  DrawContext childContext = context;
  childContext.offset = ImGui::GetCursorScreenPos();
  childContext.draw_list = ImGui::GetWindowDrawList();
  LayoutNode::draw(childContext);
  childContext.depth++;
  children.back()->traverse(childContext);

  ImGui::End();
  ImGui::PopStyleVar(2);
}

LayoutNode::Ptr
LayoutNode::makeFromJsonStrView(const std::string_view &layout) {
  getGlobalJsonTextRef() = layout.data();
  nlohmann::json jsonNode = nlohmann::json::parse(layout);
  getGlobalJsonTextRef() = "";
  return makeLayoutNodeFromJsonNode(jsonNode);
}

ButtonLayoutNode::ButtonLayoutNode(std::shared_ptr<LayoutNode> parent)
    : LayoutNode(parent) {
  type = LayoutNodeType::Button;
}

bool textHasCodepointsInRange(std::string_view text, uint32_t range_min,
                              uint32_t range_max) {
  auto it = text.begin();
  auto end = text.end();

  auto getNextCodepoint = [&it, end]() -> uint32_t {
    if (it == end)
      return 0;

    uint8_t byte = static_cast<uint8_t>(*it);

    uint32_t codePoint = 0;
    if (byte < 0x80) {
      // ASCII character (1 byte)
      codePoint = *it++;
    } else if ((byte & 0xE0) == 0xC0 && it + 1 < end) {
      // 2-byte UTF-8 character
      codePoint =
          ((byte & 0x1F) << 6) | (static_cast<uint8_t>(*(it + 1)) & 0x3F);
      it += 2;
    } else if ((byte & 0xF0) == 0xE0 && it + 2 < end) {
      // 3-byte UTF-8 character
      codePoint = ((byte & 0x0F) << 12) |
                  ((static_cast<uint8_t>(*(it + 1)) & 0x3F) << 6) |
                  (static_cast<uint8_t>(*(it + 2)) & 0x3F);
      it += 3;
    } else if ((byte & 0xF8) == 0xF0 && it + 3 < end) {
      // 4-byte UTF-8 character
      codePoint = ((byte & 0x07) << 18) |
                  ((static_cast<uint8_t>(*(it + 1)) & 0x3F) << 12) |
                  ((static_cast<uint8_t>(*(it + 2)) & 0x3F) << 6) |
                  (static_cast<uint8_t>(*(it + 3)) & 0x3F);
      it += 4;
    } else {
      // Invalid UTF-8 byte, skip
      ++it;
    }

    return codePoint;
  };

  while (it != end) {
    uint32_t codePoint = getNextCodepoint();
    if (codePoint >= range_min && codePoint <= range_max) {
      return true;
    }
  }
  return false;
}
bool textHasGlyphs(std::string_view text) {
  return textHasCodepointsInRange(text, 0xe700, 0xe7ff);
}

void ButtonLayoutNode::draw(DrawContext &context) {
  auto text = interpolateText(context.tagLibrary, fmt, args);
  if (text.empty())
    return;
  setContextPosition(context, true);
  ImVec2 size(context.width, context.height);

  bool pushFont = textHasGlyphs(text);
  if (pushFont) {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[15]);
  }

  auto color = get_color();
  ImGui::PushStyleColor(ImGuiCol_Text,
                        IM_COL32(color.r, color.g, color.b,
                                 color.a)); // Convert RGBA32 to ImGui color
  ImGui::Button(text.c_str(), size);
  ImGui::PopStyleColor();
  if (pushFont) {
    ImGui::PopFont();
  }
}

LayoutNode::Ptr ButtonLayoutNode::clone() const {
  return cloneLayoutNode(this);
}

TextLayoutNode::TextLayoutNode(std::shared_ptr<LayoutNode> parent)
    : LayoutNode(parent) {
  type = LayoutNodeType::Text;
}

void TextLayoutNode::draw(DrawContext &context) {
  std::string formattedText = interpolateText(context.tagLibrary, fmt, args);
  if (formattedText.empty())
    return;

  std::string_view text = formattedText;
  auto rgba = get_color();
  auto imcol = IM_COL32(rgba.r, rgba.g, rgba.b, rgba.a);

  setContextPosition(context, true);
  ImGui::PushStyleColor(ImGuiCol_Text, imcol);

  float x = context.x;
  float y = context.y;
  float textWidth =
      ImGui::CalcTextSize(text.data(), text.data() + text.size()).x;
  float textHeight =
      ImGui::CalcTextSize(text.data(), text.data() + text.size()).y;

  // Horizontal alignment
  switch (get_horizontalAlign()) {
  case Attribute::HorizontalAlign::Center:
    x += (context.width - textWidth) * 0.5f;
    break;
  case Attribute::HorizontalAlign::Right:
    x += context.width - textWidth;
    break;
  default: // Left
    break;
  }

  // Vertical alignment
  switch (get_verticalAlign()) {
  case Attribute::VerticalAlign::Center:
    y += (context.height - textHeight) * 0.5f;
    break;
  case Attribute::VerticalAlign::Bottom:
    y += context.height - textHeight;
    break;
  default: // Top
    break;
  }

  ImGui::SetCursorScreenPos(ImVec2(x, y));
  ImGui::TextUnformatted(text.data(), text.data() + text.size());
  ImGui::PopStyleColor();
}

LayoutNode::Ptr TextLayoutNode::clone() const { return cloneLayoutNode(this); }

LayoutNode::Ptr WindowLayoutNode::clone() const {
  return cloneLayoutNode(this);
}

LayoutNode::Ptr RefLayoutNode::clone() const {
  auto clonePtr = cloneLayoutNode(this);
  // Cast to RefLayoutNode* to set specific properties
  auto refClone = std::static_pointer_cast<RefLayoutNode>(clonePtr);
  refClone->refName = refName;
  return clonePtr;
}

LayoutNode::Ptr PlotLayoutNode::clone() const {
  auto clonePtr = cloneLayoutNode(this);
  // Cast to PlotLayoutNode* to set specific properties
  auto plotClone = std::static_pointer_cast<PlotLayoutNode>(clonePtr);

  // Clone PlotLayoutNode specific properties
  plotClone->plotType = plotType;
  plotClone->xData = xData;
  plotClone->yData = yData;
  plotClone->xLabel = xLabel;
  plotClone->yLabel = yLabel;
  plotClone->plotTitle = plotTitle;
  plotClone->showLegend = showLegend;
  plotClone->showGrid = showGrid;

  return clonePtr;
}

} // namespace M::motr::Gui
