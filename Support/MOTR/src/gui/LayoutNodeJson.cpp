//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// clang-format off
/*
JSON Schema for Layout Configuration:

Root Object:
{
  "name": <string>, // Unique identifier for the node
  "type": <string>, // "node", "window", "text", "image", "button", "plot"
  "text": <string|object>, // e.g., "Label" or {"fmt": "Hello {}", "args": ["world"]}
  "width": <number|string>, // e.g., 500 or "50%"
  "height": <number|string>, // e.g., 500 or "50%"
  "flexDirection": <string>, // "row", "row-reverse", "column", "column-reverse"
  "children": [<Child Object>], // Array of child objects
  "padding": <number|string|object>, // e.g., 10, "10%", or {"left": 10, "right": "5%"}
  "justifyContent": <string>, // "flex-start", "center", "flex-end", "space-between", "space-around", "space-evenly"
  "alignItems": <string>, // e.g., "flex-start", "center", "flex-end", "stretch", "baseline", "space-between", "space-around"
  "alignSelf": <string>, // Overrides the parent's alignItems for this node
  "alignContent": <string>, // Alignment of lines with extra space in cross-axis
  "position": <string>, // "absolute" or "relative"
  "top": <number|string>, // Position from top edge, e.g., 10 or "5%"
  "bottom": <number|string>, // Position from bottom edge
  "left": <number|string>, // Position from left edge
  "right": <number|string>, // Position from right edge
  "aspectRatio": <number>, // e.g., 1.5
  "marginInline": <number|string>, // e.g., 10 or "10%"
  "minWidth": <number|string>, // e.g., 100 or "50%"
  "maxWidth": <number|string>, // e.g., 100 or "50%"
  "minHeight": <number|string>, // e.g., 100 or "50%"
  "maxHeight": <number|string>, // e.g., 100 or "50%"
  "overflow": <string>, // "hidden", "scroll", or "visible"
  "border": <number|object>, // e.g., 1 or {"left": 1, "right": 2}
  "direction": <string>, // "ltr", "rtl", or "inherit"
  "margin": <number|string|object>, // e.g., 10, "10%", or {"top": 10, "bottom": "5%"}
  "gap": <number|string>, // e.g., 10 or "10%"
  "rowGap": <number|string>, // e.g., 10 or "10%"
  "columnGap": <number|string>, // e.g., 10 or "10%"
  "color": <string>, // Color in hex format "#RRGGBBAA" or "#RRGGBB" (defaut #FFFFFFFF)
  "ref": <string>, // Reference to a named layout component in the LayoutLibrary

  // Plot specific attributes (only used when type is "plot")
  "plot": {
    "type": <string>, // "line", "scatter", "bar", "histogram", "pie"
    "title": <string>, // Title of the plot
    "xLabel": <string>, // Label for the x-axis
    "yLabel": <string>, // Label for the y-axis
    "showLegend": <boolean>, // Whether to show the legend
    "showGrid": <boolean>, // Whether to show the grid
    "xData": <array>, // Array of x-values
    "yData": <array>  // Array of y-values
  }
}

Child Object:
{
  "flexGrow": <number>, // e.g., 1
  "flexShrink": <number>, // e.g., 1
  "flexBasis": <number>, // e.g., 100
  "flex": <number>, // e.g., 1
  "children": [<Child Object>] // Recursive structure
}
*/
// clang-format on

#include "LayoutNodeJson.h"
#include "LayoutLibrary.h"
#include "LayoutNode.h"
#include "PlotLayoutNode.h"
#include "RefLayoutNode.h"
#include "motr/JSON.h"
#include "motr/MString.h"
#include "motr/StringLibrary.h"
#include "yoga/Yoga.h"
#include <fmt/args.h>
#include <string>
#include <unordered_map>

namespace M::motr::Gui {

static const std::unordered_map<std::string, LayoutNodeType> layoutNodeTypeMap =
    {
        {"node", LayoutNodeType::Node},     //
        {"window", LayoutNodeType::Window}, //
        {"text", LayoutNodeType::Text},     //
        {"image", LayoutNodeType::Image},   //
        {"button", LayoutNodeType::Button}, //
        {"plot", LayoutNodeType::Plot},     //
};

static MString internJsonValue(const nlohmann::json &jsonValue) {
  // Intern value into StringLibrary
  if (jsonValue.is_string()) {
    const std::string &str = jsonValue.get_ref<const std::string &>();
    return {std::string_view(str)};
  }
  std::string str = jsonValue.dump();
  return {std::string_view(str)};
}

bool setDynamicAttributesByEdgeObject(std::string_view key,
                                      const nlohmann::json &jsonValue, //
                                      LayoutNode::Ptr layoutNode) {
  if (!jsonValue.is_object())
    return false;

  for (const auto &[subKey, subValue] : jsonValue.items()) {
    MString combinedKey{fmt::format("{}-{}", key, subKey)};
    auto combinedEdgeKind = Attribute::DVKindFromMString(combinedKey);
    if (combinedEdgeKind == Attribute::DVKind::COUNT) {
      MOTR_LOG("Unknown edge kind: {}", combinedKey.sv());
      continue;
    }

    MString combinedVal = internJsonValue(subValue);

    layoutNode->attrs.insert_or_assign(
        {combinedKey}, std::move(Attribute::DynamicValue::make(
                           combinedKey, combinedVal, combinedEdgeKind)));
  }

  return true;
}

// Helper function to set dynamic attribute for various attribute types
bool setDynamicAttribute(const nlohmann::json &jsonNode, //
                         LayoutNode::Ptr layoutNode,     //
                         Attribute::DVKind kind) {
  assert(kind < Attribute::DVKind::COUNT);
  const MString key = Attribute::MStringFromDVKind(kind);
  std::string_view key_sv = key.sv();

  if (!jsonNode.contains(key_sv))
    return false;

  auto jsonValue = jsonNode[key_sv];

  if (setDynamicAttributesByEdgeObject(key_sv, jsonValue, layoutNode))
    return true;

  MString valMstr = internJsonValue(jsonValue);

  layoutNode->attrs.insert_or_assign(
      key, std::move(Attribute::DynamicValue::make(key, valMstr, kind)));
  return true;
}

template <typename EnumType>
static EnumType
lookupEnumValue(const nlohmann::json &jsonNode, const std::string &key,
                const std::unordered_map<std::string, EnumType> &map,
                EnumType defaultValue) {
  if (!jsonNode.contains(key))
    return defaultValue;

  if (jsonNode[key].is_string()) {
    std::string valueStr = jsonNode[key].get<std::string>();
    auto it = map.find(valueStr);
    if (it != map.end())
      return it->second;
  }

  return defaultValue;
}

static bool handleText(const nlohmann::json &jsonNode,
                       LayoutNode::Ptr layoutNode, std::string_view key) {
  if (!jsonNode.contains(key))
    return false;
  const auto &textValue = jsonNode[key.data()];
  if (textValue.is_string()) {
    layoutNode->fmt = "{}";
    layoutNode->args = {textValue.get<std::string>()};
  } else if (textValue.is_object() && textValue.contains("fmt") &&
             textValue.contains("args")) {
    layoutNode->fmt = textValue["fmt"].get<std::string>();
    for (const auto &arg : textValue["args"]) {
      if (arg.is_string())
        layoutNode->args.push_back(arg.get<std::string>());
    }
  }
  return true;
}

static bool handleMarginInline(const nlohmann::json &jsonNode,
                               LayoutNode::Ptr layoutNode,
                               std::string_view key) {
  if (!jsonNode.contains(key))
    return false;

  auto marginInlineValue = jsonNode[key.data()];
  if (marginInlineValue.is_number_float() ||
      marginInlineValue.is_number_integer()) {
    float marginValue = marginInlineValue.get<float>();
    YGNodeStyleSetMargin(layoutNode->node, YGEdgeStart, marginValue);
    YGNodeStyleSetMargin(layoutNode->node, YGEdgeEnd, marginValue);
  } else if (marginInlineValue.is_string()) {
    std::string valueStr = marginInlineValue.get<std::string>();
    if (valueStr.back() == '%') {
      float percentage = std::stof(valueStr.substr(0, valueStr.size() - 1));
      YGNodeStyleSetMarginPercent(layoutNode->node, YGEdgeStart, percentage);
      YGNodeStyleSetMarginPercent(layoutNode->node, YGEdgeEnd, percentage);
    } else if (valueStr.find("pt") != std::string::npos) {
      float points = std::stof(valueStr.substr(0, valueStr.size() - 2));
      YGNodeStyleSetMargin(layoutNode->node, YGEdgeStart, points);
      YGNodeStyleSetMargin(layoutNode->node, YGEdgeEnd, points);
    } else {
      float value = std::stof(valueStr);
      YGNodeStyleSetMargin(layoutNode->node, YGEdgeStart, value);
      YGNodeStyleSetMargin(layoutNode->node, YGEdgeEnd, value);
    }
  }
  return true;
}

static bool handleChildren(const nlohmann::json &jsonNode,
                           LayoutNode::Ptr layoutNode, std::string_view key) {
  if (!jsonNode.contains(key))
    return false;
  for (const auto &childJsonNode : jsonNode[key.data()])
    layoutNode->appendChild(makeLayoutNodeFromJsonNode(childJsonNode));
  return true;
}

static bool handlePlotAttributes(const nlohmann::json &jsonNode,
                                 LayoutNode::Ptr layoutNode) {
  if (auto plotNode = layoutNode->as<PlotLayoutNode>()) {
    // Check if we have a plot object
    if (!jsonNode.contains("plot") || !jsonNode["plot"].is_object())
      return false;

    const auto &plotObj = jsonNode["plot"];

    // Handle plot type
    if (plotObj.contains("type")) {
      std::string plotTypeStr = plotObj["type"].get<std::string>();
      if (plotTypeStr == "line")
        plotNode->plotType = PlotType::Line;
      else if (plotTypeStr == "scatter")
        plotNode->plotType = PlotType::Scatter;
      else if (plotTypeStr == "bar")
        plotNode->plotType = PlotType::Bar;
      else if (plotTypeStr == "histogram")
        plotNode->plotType = PlotType::Histogram;
      else if (plotTypeStr == "pie")
        plotNode->plotType = PlotType::Pie;
    }

    // Handle plot title
    if (plotObj.contains("title"))
      plotNode->plotTitle = plotObj["title"].get<std::string>();

    // Handle axis labels
    if (plotObj.contains("xLabel"))
      plotNode->xLabel = plotObj["xLabel"].get<std::string>();
    if (plotObj.contains("yLabel"))
      plotNode->yLabel = plotObj["yLabel"].get<std::string>();

    // Handle plot options
    if (plotObj.contains("showLegend"))
      plotNode->showLegend = plotObj["showLegend"].get<bool>();
    if (plotObj.contains("showGrid"))
      plotNode->showGrid = plotObj["showGrid"].get<bool>();

    // Handle plot data
    if (plotObj.contains("xData") && plotObj["xData"].is_array()) {
      plotNode->xData.clear();
      for (const auto &value : plotObj["xData"]) {
        if (value.is_number())
          plotNode->xData.push_back(value.get<float>());
      }
    }

    if (plotObj.contains("yData") && plotObj["yData"].is_array()) {
      plotNode->yData.clear();
      for (const auto &value : plotObj["yData"]) {
        if (value.is_number())
          plotNode->yData.push_back(value.get<float>());
      }
    }

    return true;
  }
  return false;
}

LayoutNode::Ptr
makeTypedLayoutNodeFromJsonNodeType(const nlohmann::json &jsonNode,
                                    LayoutNode::Ptr parent) {
  // Check if it has a "ref" attribute first

  auto type = lookupEnumValue(jsonNode, "type", layoutNodeTypeMap,
                              LayoutNodeType::Node);

  if (jsonNode.contains("ref")) {
    assert(type == LayoutNodeType::Ref ||
           type == LayoutNodeType::Node &&
               "\"type\" conflict with \"ref\" attribute.");
    type = LayoutNodeType::Ref;
  }

  switch (type) {
  case LayoutNodeType::Node:
    return std::make_shared<LayoutNode>(parent);
  case LayoutNodeType::Window:
    return std::make_shared<WindowLayoutNode>(parent);
  case LayoutNodeType::Text:
    return std::make_shared<TextLayoutNode>(parent);
  case LayoutNodeType::Image:
    assert(false && "ImageLayoutNode not implemented");
    return std::make_shared<LayoutNode>(parent);
  case LayoutNodeType::Button:
    return std::make_shared<ButtonLayoutNode>(parent);
  case LayoutNodeType::Plot:
    return std::make_shared<PlotLayoutNode>(parent);
  case LayoutNodeType::Ref:
    return std::make_shared<RefLayoutNode>(parent);
  }
  return nullptr;
}

std::shared_ptr<LayoutNode>
makeLayoutNodeFromJsonNode(const nlohmann::json &jsonNode) {

  auto type = lookupEnumValue(jsonNode, "type", layoutNodeTypeMap,
                              LayoutNodeType::Node);

  if (jsonNode.contains("ref")) {
    auto refname = jsonNode["ref"].get<std::string>();
    assert(!refname.empty() && "ref attribute is not allowed to be empty.");
    LayoutLibrary &layoutLibrary = LayoutLibrary::instance();
    auto refNode = layoutLibrary.getLayout(refname);
    // assert(refNode && "ref attribute refers to a non-existent layout node.");
    return refNode ? refNode->clone() : nullptr;
  }

  LayoutNode::Ptr layoutNode =
      makeTypedLayoutNodeFromJsonNodeType(jsonNode, nullptr);

  // "ref" attribute is handled first as a special case
  if (auto refNode = layoutNode->as<RefLayoutNode>()) {
    refNode->refName = jsonNode["ref"].get<std::string>();
    // TODO: Validate there are no other attributes after the "ref" attribute
    return layoutNode;
  }

  using DVKind = M::motr::Gui::Attribute::DVKind;

  // Refer to DynamicValue.h for the mapping of keys to DVKind
  for (size_t idx = 0; idx < static_cast<size_t>(DVKind::COUNT); ++idx) {
    DVKind kind = static_cast<DVKind>(idx);
    setDynamicAttribute(jsonNode, layoutNode, kind);
  }

  handleText(jsonNode, layoutNode, "text");

  // Handle plot-specific attributes
  if (layoutNode->type == LayoutNodeType::Plot) {
    handlePlotAttributes(jsonNode, layoutNode);
  }

  // Handle layout properties

  handleMarginInline(jsonNode, layoutNode, "marginInline");

  // Parse children recursively
  handleChildren(jsonNode, layoutNode, "children");

  return layoutNode;
}

// Utility function to format numbers to the necessary precision
float formatNumber(float value) {
  // Round to two decimal places
  return std::round(value * 100.0f) / 100.0f;
}

nlohmann::json makeJsonNodeFromLayoutNode(const LayoutNode *node) { return {}; }

} // namespace M::motr::Gui
