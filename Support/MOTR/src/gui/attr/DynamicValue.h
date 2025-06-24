//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_DYNAMICVALUE_H
#define MOTR_DYNAMICVALUE_H

#include "../Color.h"
#include "motr/Hash.h"
#include "motr/MString.h"
#include "yoga/YGEnums.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

/* The DynamicVariable is stored as attributes in LayoutNode
 * Steps to add an attribute to layoutnode:
 * 1. Find the underling C++ type represented in DynamicVariant below, if you
 * don't find, add it.
 * 2. Add a new DVKind enum, this corresponds to a unique attribute across all
 * LayoutNode types
 * 3. Add a new entry for the attribute string representation in DVKindsKeys
 * 4. Add a new entry in MStringFromDVKind
 * 5. Add a new entry in DVKindFromMString
 */

namespace M::motr::Gui::Attribute {

enum class HorizontalAlign : int { Left, Center, Right };
enum class VerticalAlign : int { Top, Center, Bottom };

// Struct for storing width/height values with their unit
struct DimensionResolved {
  enum Unit { Point, Percent, Auto };
  Unit unit = Unit::Point;
  float value = 0.0f;
};

struct FlexResolved {
  float grow = 0.0f;
  float shrink = 0.0f;
  DimensionResolved basis{DimensionResolved::Unit::Auto};
};

// Special case for DynamicValue that needs to be resolved
struct UnresolvedVariant {};

using DynamicVariant = std::variant< // using std for better or worse.
    UnresolvedVariant,               //
    std::string,       // String, todo: store StringLibrary hash instead
    int64_t,           // Int
    float,             // Float
    DimensionResolved, // Width/Height
    Color::RGBA32,     // Color
    int,               // HAlign, VAlign, or other enum types
    bool,              // Bool
    HorizontalAlign,   // HAlign
    VerticalAlign,     // VAlign
    YGPositionType,    // Yoga position type
    YGFlexDirection,   // Yoga flex direction
    FlexResolved,      // Flex resolved
    YGJustify,         // Yoga justify
    YGAlign,           // Yoga align
    YGOverflow,        // Yoga overflow
    YGDirection        // Yoga direction

    >;

// Attribute value kinds
enum class DVKind : uint8_t {
  Name = 0,        // Name of the node
  Width,           // Width/height with unit (%, pt, or none)
  Height,          // Height with same semantics as Width
  MinWidth,        // Minimum width with same semantics as Width
  MaxWidth,        // Maximum width with same semantics as Width
  MinHeight,       // Minimum height with same semantics as Height
  MaxHeight,       // Maximum height with same semantics as Height
  Top,             // Top with same semantics as Width
  Left,            // Left with same semantics as Width
  Bottom,          // Bottom with same semantics as Width
  Right,           // Right with same semantics as Width
  Position,        // Yoga position type
  FlexDirection,   // Flex direction attribute
  FlexGrow,        // Flex grow attribute
  FlexShrink,      // Flex shrink attribute
  FlexBasis,       // Flex basis attribute
  Flex,            // Flex attribute
  JustifyContent,  // Justify content attribute
  Gap,             // Gap attribute
  RowGap,          // Row gap attribute
  ColumnGap,       // Column gap attribute
  Color,           // RGBA32 color
  BackgroundColor, // Background color attribute
  AspectRatio,     // Aspect ratio attribute
  HorizontalAlign, // Horizontal alignment enum
  VerticalAlign,   // Vertical alignment enum
  AlignSelf,
  AlignContent,
  AlignItems,
  Overflow,
  Direction,
  PaddingAll,
  PaddingLeft,
  PaddingTop,
  PaddingRight,
  PaddingBottom,
  PaddingStart,
  PaddingEnd,
  PaddingHorizontal,
  PaddingVertical,
  BorderAll,
  BorderLeft,
  BorderTop,
  BorderRight,
  BorderBottom,
  BorderStart,
  BorderEnd,
  BorderHorizontal,
  BorderVertical,
  MarginAll,
  MarginLeft,
  MarginTop,
  MarginRight,
  MarginBottom,
  MarginStart,
  MarginEnd,
  MarginHorizontal,
  MarginVertical,
  // Add more types as needed before COUNT
  COUNT,
};

constexpr std::array<std::string_view, static_cast<size_t>(DVKind::COUNT)>
    DVKindsKeys = {
        "name",            //
        "width",           //
        "height",          //
        "minWidth",        //
        "maxWidth",        //
        "minHeight",       //
        "maxHeight",       //
        "top",             //
        "left",            //
        "bottom",          //
        "right",           //
        "position",        //
        "flexDirection",   //
        "flexGrow",        //
        "flexShrink",      //
        "flexBasis",       //
        "flex",            //
        "justifyContent",  //
        "gap",             //
        "rowGap",          //
        "columnGap",       //
        "color",           //
        "backgroundColor", //
        "aspectRatio",     //
        "horizontalAlign", //
        "verticalAlign",   //
        "alignSelf",
        "alignContent",
        "alignItems",
        "overflow",
        "direction",
        "padding",
        "padding-left",
        "padding-top",
        "padding-right",
        "padding-bottom",
        "padding-start",
        "padding-end",
        "padding-horizontal",
        "padding-vertical",
        "border",
        "border-left",
        "border-top",
        "border-right",
        "border-bottom",
        "border-start",
        "border-end",
        "border-horizontal",
        "border-vertical",
        "margin",
        "margin-left",
        "margin-top",
        "margin-right",
        "margin-bottom",
        "margin-start",
        "margin-end",
        "margin-horizontal",
        "margin-vertical",
};

static_assert(DVKindsKeys[static_cast<size_t>(DVKind::Gap)] ==
                  std::string_view("gap"),
              "DVKindsKeys spot check mismatch");
static_assert(DVKindsKeys[static_cast<size_t>(DVKind::MarginAll)] ==
                  std::string_view("margin"),
              "DVKindsKeys spot check mismatch");
static_assert(DVKindsKeys[static_cast<size_t>(DVKind::PaddingAll)] ==
                  std::string_view("padding"),
              "DVKindsKeys spot check mismatch");
static_assert(DVKindsKeys[static_cast<size_t>(DVKind::BorderAll)] ==
                  std::string_view("border"),
              "DVKindsKeys spot check mismatch");
static_assert(DVKindsKeys[static_cast<size_t>(DVKind::MarginVertical)] ==
                  std::string_view("margin-vertical"),
              "DVKindsKeys spot check mismatch");

static MString MStringFromDVKind(DVKind kind) {
  static const std::vector<MString> DVKindsKeysMstr = ([]() {
    std::vector<MString> ret;
    ret.reserve(DVKindsKeys.size());
    for (auto key : DVKindsKeys) {
      ret.push_back(MString(key));
    }
    return ret;
  }());
  assert(kind >= static_cast<DVKind>(0) && kind < DVKind::COUNT);
  size_t index = static_cast<size_t>(kind) % DVKindsKeysMstr.size();
  return DVKindsKeysMstr[index];
}

static DVKind DVKindFromMString(const MString &mstr) {
  static const std::unordered_map<MString, DVKind> MStrToDVKind = ([]() {
    std::unordered_map<MString, DVKind> ret;
    for (size_t i = 0; i < static_cast<size_t>(DVKind::COUNT); ++i) {
      ret[{DVKindsKeys[i]}] = static_cast<DVKind>(i);
    }
    return ret;
  }());
  auto it = MStrToDVKind.find(mstr);
  if (it != MStrToDVKind.end()) {
    return it->second;
  }
  return DVKind::COUNT;
}
// Struct representing a single dynamic attribute value
struct DynamicValue {

  // The key of the attribute
  MString key;

  // The expression of the attribute
  // Raw value as a hash (from string library)
  MString expr;

  // The semantic kind of attribute this represents
  DVKind kind;

  // Cached resolved value - using std::variant to hold different types
  // Note: This will be replaced with TypeID + valueBits in the final
  // implementation
  DynamicVariant cache;

  // Dependencies - variables this value depends on
  std::vector<MString> deps;

  // Version numbers for each dependency (for change tracking)
  std::vector<uint64_t> versions;

  DynamicValue(const DynamicValue &) = default;
  DynamicValue &operator=(const DynamicValue &) = default;
  DynamicValue(DynamicValue &&) = default;
  DynamicValue &operator=(DynamicValue &&) = default;

private:
  DynamicValue() = default;

public:
  static DynamicValue make(const MString &key, const MString &expr,
                           DVKind kind) {
    DynamicValue dv;
    dv.key = key;
    dv.expr = expr;
    dv.kind = kind;
    return dv;
  }
};

} // namespace M::motr::Gui::Attribute

#endif // MOTR_DYNAMICVALUE_H
