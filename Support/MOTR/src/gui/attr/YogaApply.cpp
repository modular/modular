//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "YogaApply.h"
#include "../LayoutNode.h"
#include "Parse.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>

#include <yoga/Yoga.h>

namespace M::motr::Gui::YogaApply {

using namespace M::motr::Gui::Attribute;

template <typename T>
const T *get_variant(const DynamicValue &dv, bool expected = true) {
  const T *p = std::get_if<T>(&dv.cache);
  if (expected && !p) {
    assert(false && "get_variant: dynamic value is not a T");
  }
  return p;
}

// Helper to apply dimension values based on parsed unit
bool applyDimension(
    YGNodeRef node,                       //
    const DimensionResolved &dim,         //
    void (*setPercent)(YGNodeRef, float), //
    void (*setPoint)(YGNodeRef, float),   //
    void (*setAuto)(YGNodeRef) // can be nullptr if Yoga does not support auto
                               // for this dimension
) {                            //
  switch (dim.unit) {

  case DimensionResolved::Percent:
    setPercent(node, dim.value);
    return true;
  case DimensionResolved::Point:
    setPoint(node, dim.value);
    return true;
  case DimensionResolved::Auto:
    if (!setAuto) {
      assert(false && "applyDimension: auto value set not supported for this "
                      "dimension type");
      return false;
    }
    setAuto(node);
    return true;
  }

  return false;
}

bool applyDimension(YGNodeRef node, const DynamicValue &dv,
                    void (*setPercent)(YGNodeRef, float),
                    void (*setPoint)(YGNodeRef, float),
                    void (*setAuto)(YGNodeRef)) {
  const DimensionResolved *p = get_variant<DimensionResolved>(dv);
  if (p == nullptr)
    return false;

  return applyDimension(node, *p, setPercent, setPoint, setAuto);
}

// Helper to apply dimension values to edge properties
bool applyEdgeDimension(YGNodeRef node, const DynamicValue &dv, YGEdge edge,
                        void (*setPercentEdge)(YGNodeRef, YGEdge, float),
                        void (*setPointEdge)(YGNodeRef, YGEdge, float)) {
  const DimensionResolved *p = get_variant<DimensionResolved>(dv);

  // If already resolved in cache
  if (p) {
    if (p->unit == DimensionResolved::Percent) {
      setPercentEdge(node, edge, p->value);
    } else {
      setPointEdge(node, edge, p->value);
    }
    return true;
  }

  return false;
}

bool applyAlign(YGNodeRef node, const DynamicValue &dv,
                void (*setter)(YGNodeRef, YGAlign), YGAlign defaultValue) {
  const YGAlign *p = get_variant<YGAlign>(dv);
  setter(node, p ? *p : defaultValue);
  return true;
}

bool applyGapDimension(YGNodeRef node, const DynamicValue &dv,
                       YGGutter gutter) {
  if (const DimensionResolved *p = get_variant<DimensionResolved>(dv); p) {
    YGNodeStyleSetGap(node, gutter, p->value);
    return true;
  }
  return false;
}

// Helper to apply float value
bool applyFloat(YGNodeRef node, const DynamicValue &dv,
                void (*setter)(YGNodeRef, float)) {
  if (const float *p = get_variant<float>(dv, false); p) {
    setter(node, *p);
    return true;
  }

  if (const int64_t *p = get_variant<int64_t>(dv, false); p) {
    setter(node, static_cast<float>(*p));
    return true;
  }

  // todo: support parsing from other formats?
  return false;
}
bool applyDimensionEdge(YGNodeRef node, const DynamicValue &dv, YGEdge edge,
                        void (*set)(YGNodeRef, YGEdge, float),
                        void (*setPercent)(YGNodeRef, YGEdge, float) = nullptr,
                        void (*setAuto)(YGNodeRef, YGEdge) = nullptr) {
  const DimensionResolved *p = get_variant<DimensionResolved>(dv);
  if (p == nullptr)
    return false;

  switch (p->unit) {
  case DimensionResolved::Percent:
    if (setPercent)
      setPercent(node, edge, p->value);
    else
      MOTR_LOG(
          "error: percent unit is not supported for attribute \"{}\": \"{}\"",
          dv.key.sv(), dv.expr.sv());
    break;
  case DimensionResolved::Point:
    if (set)
      set(node, edge, p->value);
    else
      MOTR_LOG(
          "error: point unit is not supported for attribute \"{}\": \"{}\"",
          dv.key.sv(), dv.expr.sv());
    break;
  case DimensionResolved::Auto:
    if (setAuto)
      setAuto(node, edge);
    else
      MOTR_LOG("error: auto unit is not supported for attribute \"{}\": \"{}\"",
               dv.key.sv(), dv.expr.sv());
    break;
  }
  return true;
}

bool applyDynamicValue(LayoutNode &node, MString key) {
  auto &attrs = node.attrs;
  auto &ygnode = node.node;

  if (!ygnode)
    return false;

  auto it = attrs.find(key);
  if (it == attrs.end())
    return false;

  const DynamicValue &dv = it->second;

  DVKind keyKind = Attribute::DVKindFromMString(key);
  assert(keyKind != DVKind::COUNT);
  assert(dv.kind == keyKind);

  switch (keyKind) {
  // Early exit Non-Yoga properties
  case DVKind::Color:
  case DVKind::HorizontalAlign:
  case DVKind::VerticalAlign:
  case DVKind::Name:
  case DVKind::BackgroundColor:
    return false;

  case DVKind::Position: {
    assert(dv.kind == DVKind::Position);
    YGNodeStyleSetPositionType(ygnode, std::get<YGPositionType>(dv.cache));
    return true;
  }
  case DVKind::FlexDirection: {
    auto *p = std::get_if<YGFlexDirection>(&dv.cache);
    assert(p && "applyDynamicValue: flexDirection is not a YGFlexDirection");
    if (*p)
      YGNodeStyleSetFlexDirection(ygnode, *p);
    return true;
  }
  case DVKind::Width: {
    return applyDimension(ygnode, dv, YGNodeStyleSetWidthPercent,
                          YGNodeStyleSetWidth, YGNodeStyleSetWidthAuto);
  }
  case DVKind::Height: {
    return applyDimension(ygnode, dv, YGNodeStyleSetHeightPercent,
                          YGNodeStyleSetHeight, YGNodeStyleSetHeightAuto);
  }
  case DVKind::MinWidth: {
    return applyDimension(ygnode, dv, YGNodeStyleSetMinWidthPercent,
                          YGNodeStyleSetMinWidth, nullptr);
  }
  case DVKind::MaxWidth: {
    return applyDimension(ygnode, dv, YGNodeStyleSetMaxWidthPercent,
                          YGNodeStyleSetMaxWidth, nullptr);
  }
  case DVKind::MinHeight: {
    return applyDimension(ygnode, dv, YGNodeStyleSetMinHeightPercent,
                          YGNodeStyleSetMinHeight, nullptr);
  }
  case DVKind::MaxHeight: {
    return applyDimension(ygnode, dv, YGNodeStyleSetMaxHeightPercent,
                          YGNodeStyleSetMaxHeight, nullptr);
  }
  case DVKind::Top: {
    // todo handle auto
    return applyEdgeDimension(ygnode, dv, YGEdgeTop,
                              YGNodeStyleSetPositionPercent,
                              YGNodeStyleSetPosition);
  }
  case DVKind::Right: {
    // todo handle auto
    return applyEdgeDimension(ygnode, dv, YGEdgeRight,
                              YGNodeStyleSetPositionPercent,
                              YGNodeStyleSetPosition);
  }
  case DVKind::Bottom: {
    // todo handle auto
    return applyEdgeDimension(ygnode, dv, YGEdgeBottom,
                              YGNodeStyleSetPositionPercent,
                              YGNodeStyleSetPosition);
  }
  case DVKind::Left: {
    // todo handle auto
    return applyEdgeDimension(ygnode, dv, YGEdgeLeft,
                              YGNodeStyleSetPositionPercent,
                              YGNodeStyleSetPosition);
  }
  case DVKind::FlexGrow: {
    return applyFloat(ygnode, dv, YGNodeStyleSetFlexGrow);
  }
  case DVKind::FlexShrink: {
    return applyFloat(ygnode, dv, YGNodeStyleSetFlexShrink);
  }
  case DVKind::FlexBasis: {
    return applyDimension(ygnode, dv, YGNodeStyleSetFlexBasisPercent,
                          YGNodeStyleSetFlexBasis, YGNodeStyleSetFlexBasisAuto);
  }
  case DVKind::Flex: {
    auto *p = get_variant<FlexResolved>(dv);
    if (!p)
      return false;

    YGNodeStyleSetFlexGrow(ygnode, p->grow);
    YGNodeStyleSetFlexShrink(ygnode, p->shrink);
    applyDimension(ygnode, p->basis, YGNodeStyleSetFlexBasisPercent,
                   YGNodeStyleSetFlexBasis, YGNodeStyleSetFlexBasisAuto);
    return true;
  }
  case DVKind::JustifyContent: {
    auto *p = get_variant<YGJustify>(dv);
    if (!p)
      return false;
    YGNodeStyleSetJustifyContent(ygnode, *p);
    return true;
  }
  case DVKind::Gap: {
    return applyGapDimension(ygnode, dv, YGGutterAll);
  }
  case DVKind::RowGap: {
    return applyGapDimension(ygnode, dv, YGGutterRow);
  }
  case DVKind::ColumnGap: {
    return applyGapDimension(ygnode, dv, YGGutterColumn);
  }
  case DVKind::AspectRatio: {
    return applyFloat(ygnode, dv, YGNodeStyleSetAspectRatio);
  }
  case DVKind::AlignSelf: {
    return applyAlign(ygnode, dv, YGNodeStyleSetAlignSelf, YGAlignAuto);
  }
  case DVKind::AlignContent: {
    return applyAlign(ygnode, dv, YGNodeStyleSetAlignContent, YGAlignStretch);
  }
  case DVKind::AlignItems: {
    return applyAlign(ygnode, dv, YGNodeStyleSetAlignItems, YGAlignStretch);
  }
  case DVKind::Overflow: {
    auto *p = get_variant<YGOverflow>(dv);
    if (!p)
      return false;
    YGNodeStyleSetOverflow(ygnode, *p);
    return true;
  }
  case DVKind::Direction: {
    auto *p = get_variant<YGDirection>(dv);
    if (!p)
      return false;
    YGNodeStyleSetDirection(ygnode, *p);
    return true;
  }
  // padding, padding-left, padding-right, padding-top, padding-bottom,
  // padding-start, padding-end, padding-horizontal, padding-vertical
  case DVKind::PaddingAll:
    return applyDimensionEdge(ygnode, dv, YGEdgeAll, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingLeft:
    return applyDimensionEdge(ygnode, dv, YGEdgeLeft, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingRight:
    return applyDimensionEdge(ygnode, dv, YGEdgeRight, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingTop:
    return applyDimensionEdge(ygnode, dv, YGEdgeTop, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingBottom:
    return applyDimensionEdge(ygnode, dv, YGEdgeBottom, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingStart:
    return applyDimensionEdge(ygnode, dv, YGEdgeStart, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingEnd:
    return applyDimensionEdge(ygnode, dv, YGEdgeEnd, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingHorizontal:
    return applyDimensionEdge(ygnode, dv, YGEdgeHorizontal,
                              YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);
  case DVKind::PaddingVertical:
    return applyDimensionEdge(ygnode, dv, YGEdgeVertical, YGNodeStyleSetPadding,
                              YGNodeStyleSetPaddingPercent);

  // border, border-left, border-right, border-top, border-bottom, border-start,
  // border-end, border-horizontal, border-vertical
  case DVKind::BorderAll:
    return applyDimensionEdge(ygnode, dv, YGEdgeAll, YGNodeStyleSetBorder);
  case DVKind::BorderLeft:
    return applyDimensionEdge(ygnode, dv, YGEdgeLeft, YGNodeStyleSetBorder);
  case DVKind::BorderRight:
    return applyDimensionEdge(ygnode, dv, YGEdgeRight, YGNodeStyleSetBorder);
  case DVKind::BorderTop:
    return applyDimensionEdge(ygnode, dv, YGEdgeTop, YGNodeStyleSetBorder);
  case DVKind::BorderBottom:
    return applyDimensionEdge(ygnode, dv, YGEdgeBottom, YGNodeStyleSetBorder);
  case DVKind::BorderStart:
    return applyDimensionEdge(ygnode, dv, YGEdgeStart, YGNodeStyleSetBorder);
  case DVKind::BorderEnd:
    return applyDimensionEdge(ygnode, dv, YGEdgeEnd, YGNodeStyleSetBorder);
  case DVKind::BorderHorizontal:
    return applyDimensionEdge(ygnode, dv, YGEdgeHorizontal,
                              YGNodeStyleSetBorder);
  case DVKind::BorderVertical:
    return applyDimensionEdge(ygnode, dv, YGEdgeVertical, YGNodeStyleSetBorder);

  // margin, margin-left, margin-right, margin-top, margin-bottom, margin-start,
  // margin-end, margin-horizontal, margin-vertical
  case DVKind::MarginAll:
    return applyDimensionEdge(ygnode, dv, YGEdgeAll, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginLeft:
    return applyDimensionEdge(ygnode, dv, YGEdgeLeft, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginRight:
    return applyDimensionEdge(ygnode, dv, YGEdgeRight, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginTop:
    return applyDimensionEdge(ygnode, dv, YGEdgeTop, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginBottom:
    return applyDimensionEdge(ygnode, dv, YGEdgeBottom, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginStart:
    return applyDimensionEdge(ygnode, dv, YGEdgeStart, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginEnd:
    return applyDimensionEdge(ygnode, dv, YGEdgeEnd, YGNodeStyleSetMargin,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginHorizontal:
    return applyDimensionEdge(ygnode, dv, YGEdgeHorizontal,
                              YGNodeStyleSetMargin, YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMarginAuto);
  case DVKind::MarginVertical:
    return applyDimensionEdge(ygnode, dv, YGEdgeVertical,
                              YGNodeStyleSetMarginPercent,
                              YGNodeStyleSetMargin);

  case DVKind::COUNT:
    assert(false && "applyDynamicValue: invalid DVKind::COUNT");
    return false;
  }
}

/*
static const std::unordered_map<std::string, YGPositionType> positionTypeMap = {
    {"absolute", YGPositionTypeAbsolute},
    {"relative", YGPositionTypeRelative},
};

static const std::unordered_map<std::string, YGOverflow> overflowMap = {
    {"visible", YGOverflowVisible},
    {"hidden", YGOverflowHidden},
    {"scroll", YGOverflowScroll},
};

static const std::unordered_map<std::string, YGWrap> wrapMap = {
    {"nowrap", YGWrapNoWrap},
    {"wrap", YGWrapWrap},
    {"wrap-reverse", YGWrapWrapReverse},
};

static const std::unordered_map<std::string, YGDirection> directionMap = {
    {"ltr", YGDirectionLTR},
    {"rtl", YGDirectionRTL},
    {"inherit", YGDirectionInherit},
};

// Helper to look up a string value and convert to enum
template <typename T>
bool applyEnum(YGNodeRef node, const DynamicValue &dv,
               const std::unordered_map<std::string, T> &map,
               void (*setter)(YGNodeRef, T), T defaultValue) {
  // If we have a string value, try to look it up in the map
  if (auto *strVal = std::get_if<std::string>(&dv.cache)) {
    auto it = map.find(*strVal);
    if (it != map.end()) {
      setter(node, it->second);
      return true;
    }
    // If not found, use default
    setter(node, defaultValue);
    return true;
  }

  // If not a string, try to use as direct value
  if (auto *intVal = std::get_if<int64_t>(&dv.cache)) {
    setter(node, static_cast<T>(*intVal));
    return true;
  }

  // Otherwise use default
  setter(node, defaultValue);
  return false;
}

static const std::unordered_map<std::string, YGJustify> justifyContentMap = {
    {"flex-start", YGJustifyFlexStart},
    {"center", YGJustifyCenter},
    {"flex-end", YGJustifyFlexEnd},
    {"space-between", YGJustifySpaceBetween},
    {"space-around", YGJustifySpaceAround},
    {"space-evenly", YGJustifySpaceEvenly},
};

void set(YGNodeRef node, M::motr::Hash::Value key, const DynamicValue &dv) {
  if (!node)
    return;

  // Dispatch based on attribute key
  if (key.v == Keys::width().v) {
    applyDimension(node, dv, YGNodeStyleSetWidthPercent, YGNodeStyleSetWidth);
  } else if (key.v == Keys::height().v) {
    applyDimension(node, dv, YGNodeStyleSetHeightPercent, YGNodeStyleSetHeight);
  } else if (key.v == Keys::minWidth().v) {
    applyDimension(node, dv, YGNodeStyleSetMinWidthPercent,
                   YGNodeStyleSetMinWidth);
  } else if (key.v == Keys::maxWidth().v) {
    applyDimension(node, dv, YGNodeStyleSetMaxWidthPercent,
                   YGNodeStyleSetMaxWidth);
  } else if (key.v == Keys::minHeight().v) {
    applyDimension(node, dv, YGNodeStyleSetMinHeightPercent,
                   YGNodeStyleSetMinHeight);
  } else if (key.v == Keys::maxHeight().v) {
    applyDimension(node, dv, YGNodeStyleSetMaxHeightPercent,
                   YGNodeStyleSetMaxHeight);
  }
  // Position-related
  else if (key.v == Keys::top().v) {
    applyEdgeDimension(node, dv, YGEdgeTop, YGNodeStyleSetPositionPercent,
                       YGNodeStyleSetPosition);
  } else if (key.v == Keys::right().v) {
    applyEdgeDimension(node, dv, YGEdgeRight, YGNodeStyleSetPositionPercent,
                       YGNodeStyleSetPosition);
  } else if (key.v == Keys::bottom().v) {
    applyEdgeDimension(node, dv, YGEdgeBottom, YGNodeStyleSetPositionPercent,
                       YGNodeStyleSetPosition);
  } else if (key.v == Keys::left().v) {
    applyEdgeDimension(node, dv, YGEdgeLeft, YGNodeStyleSetPositionPercent,
                       YGNodeStyleSetPosition);
  }
  // Padding, margin, border
  else if (key.v == Keys::margin().v) {
    applyEdgeDimension(node, dv, YGEdgeAll, YGNodeStyleSetMarginPercent,
                       YGNodeStyleSetMargin);
  } else if (key.v == Keys::padding().v) {
    applyEdgeDimension(node, dv, YGEdgeAll, YGNodeStyleSetPaddingPercent,
                       YGNodeStyleSetPadding);
  } else if (key.v == Keys::border().v) {
    if (auto *val = std::get_if<float>(&dv.cache)) {
      YGNodeStyleSetBorder(node, YGEdgeAll, *val);
    } else if (auto *val = std::get_if<int64_t>(&dv.cache)) {
      YGNodeStyleSetBorder(node, YGEdgeAll, static_cast<float>(*val));
    }
  }
  // Flex properties
  else if (key.v == Keys::flexDirection().v) {
    applyEnum(node, dv, flexDirectionMap, YGNodeStyleSetFlexDirection,
              YGFlexDirectionColumn);
  } else if (key.v == Keys::flexWrap().v) {
    applyEnum(node, dv, wrapMap, YGNodeStyleSetFlexWrap, YGWrapNoWrap);
  } else if (key.v == Keys::flexGrow().v) {
    applyFloat(node, dv, YGNodeStyleSetFlexGrow);
  } else if (key.v == Keys::flexShrink().v) {
    applyFloat(node, dv, YGNodeStyleSetFlexShrink);
  } else if (key.v == Keys::flexBasis().v) {
    applyDimension(node, dv, YGNodeStyleSetFlexBasisPercent,
                   YGNodeStyleSetFlexBasis);
  } else if (key.v == Keys::flex().v) {
    applyFloat(node, dv, YGNodeStyleSetFlex);
  }
  // Alignment
  else if (key.v == Keys::justifyContent().v) {
    applyEnum(node, dv, justifyContentMap, YGNodeStyleSetJustifyContent,
              YGJustifyFlexStart);
  } else if (key.v == Keys::alignItems().v) {
    applyEnum(node, dv, alignMap, YGNodeStyleSetAlignItems, YGAlignStretch);
  } else if (key.v == Keys::alignSelf().v) {
    applyEnum(node, dv, alignMap, YGNodeStyleSetAlignSelf, YGAlignAuto);
  } else if (key.v == Keys::alignContent().v) {
    applyEnum(node, dv, alignMap, YGNodeStyleSetAlignContent, YGAlignFlexStart);
  }
  // Gap, overflow, direction
  else if (key.v == Keys::gap().v) {
    applyFloat(node, dv, [](YGNodeRef node, float value) {
      YGNodeStyleSetGap(node, YGGutterAll, value);
    });
  } else if (key.v == Keys::rowGap().v) {
    applyFloat(node, dv, [](YGNodeRef node, float value) {
      YGNodeStyleSetGap(node, YGGutterRow, value);
    });
  } else if (key.v == Keys::columnGap().v) {
    applyFloat(node, dv, [](YGNodeRef node, float value) {
      YGNodeStyleSetGap(node, YGGutterColumn, value);
    });
  } else if (key.v == Keys::overflow().v) {
    applyEnum(node, dv, overflowMap, YGNodeStyleSetOverflow, YGOverflowVisible);
  } else if (key.v == Keys::direction().v) {
    applyEnum(node, dv, directionMap, YGNodeStyleSetDirection,
              YGDirectionInherit);
  } else if (key.v == Keys::aspectRatio().v) {
    applyFloat(node, dv, YGNodeStyleSetAspectRatio);
  }
  // Other properties can be added as needed
}
*/

} // namespace M::motr::Gui::YogaApply
