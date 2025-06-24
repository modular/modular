//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Convert.h"
#include "DynamicValue.h"
#include "Parse.h"
#include "motr/TagLibrary.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <unordered_map>

using namespace M::motr::Gui::Attribute;

// Implementation at bottom of file, calls appropriate convert function
// based on the DVKind DynamicValue::kind member
static bool dispatchConvertValue(std::string_view val_sv, DynamicValue &dv);

using ConvertFunc = bool (*)(std::string_view, DynamicValue &);
using ConvertFuncTable = std::vector<ConvertFunc>;
static ConvertFuncTable &getConvertFuncTable();

// Convert a string value to a DynamicValue
// Returns true if the conversion was successful
// Returns false if the conversion was not successful
bool M::motr::Gui::Attribute::convertValue(std::string_view val_sv,
                                           DynamicValue &dv) {
  ConvertFuncTable &convertFuncTable = getConvertFuncTable();
  ConvertFunc convertFunc = convertFuncTable[size_t(dv.kind)];
  if (convertFunc && convertFunc(val_sv, dv))
    return true;
  dv.cache = UnresolvedVariant{};
  return false;
}

namespace {
template <typename EnumType>
const EnumType *
lookup(std::string_view key,
       const std::unordered_map<std::string_view, EnumType> &map) {

  auto it = map.find(key);
  if (it != map.end())
    return &it->second;
  return nullptr;
}

template <typename T>
inline bool setValue(DynamicValue &value, T &&val) {
  value.cache = std::forward<T>(val);
  return true;
}

template <typename T>
inline bool
lookupAndSetDynamicValue(std::string_view key, DynamicValue &value,
                         const std::unordered_map<std::string_view, T> &map) {
  if (auto *p = lookup(key, map); p)
    return setValue(value, *p);
  return false;
}

bool convertToString(std::string_view str, DynamicValue &value) {
  return setValue(value, std::string(str));
}

bool convertToInt64(std::string_view str, DynamicValue &value) {
  int64_t result = 0;
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);
  if (ec == std::errc())
    return setValue(value, result);
  return false;
}

bool convertToFloat(std::string_view str, DynamicValue &value) {
  char *end;
  float result = strtof(std::string(str).c_str(), &end);
  if (end != str.data())
    return setValue(value, result);
  return false;
}

bool convertToBool(std::string_view str, DynamicValue &value) {
  std::string lower(str);
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "true" || lower == "1" || lower == "yes" || lower == "on")
    return setValue(value, true);
  else if (lower == "false" || lower == "0" || lower == "no" || lower == "off")
    return setValue(value, false);
  return false;
}

bool convertToDimensionResolved(std::string_view str, DynamicValue &value) {
  DimensionResolved result;
  if (Parse::parseDimensionValue(str, result))
    return setValue(value, result);
  return setValue(value, DimensionResolved{DimensionResolved::Point, 0.0f});
}

bool convertToHorizontalAlign(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, HorizontalAlign> hAlignMap =
      {{"left", HorizontalAlign::Left},
       {"center", HorizontalAlign::Center},
       {"middle", HorizontalAlign::Center},
       {"right", HorizontalAlign::Right}};

  return lookupAndSetDynamicValue(str, value, hAlignMap);
}

bool convertToVerticalAlign(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, VerticalAlign> vAlignMap = {
      {"top", VerticalAlign::Top},
      {"center", VerticalAlign::Center},
      {"middle", VerticalAlign::Center},
      {"bottom", VerticalAlign::Bottom}};

  return lookupAndSetDynamicValue(str, value, vAlignMap);
}

bool convertToColorRGBA32(std::string_view str, DynamicValue &value) {
  M::motr::Gui::Color::RGBA32 result;

  if (Parse::parseColor(str, result))
    return setValue(value, result);
  return false;
}

bool convertToYGPositionType(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGPositionType>
      positionTypeMap = {
          {"absolute", YGPositionTypeAbsolute}, //
          {"relative", YGPositionTypeRelative}, //
      };
  return lookupAndSetDynamicValue(str, value, positionTypeMap);
}

bool convertToYGFlexDirection(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGFlexDirection>
      flexDirectionMap = {
          {"row", YGFlexDirectionRow},
          {"row-reverse", YGFlexDirectionRowReverse},
          {"column", YGFlexDirectionColumn},
          {"column-reverse", YGFlexDirectionColumnReverse},
      };
  return lookupAndSetDynamicValue(str, value, flexDirectionMap);
}

bool convertToFlexResolved(std::string_view str, DynamicValue &value) {
  FlexResolved result;
  if (Parse::parseFlexResolved(str, result))
    return setValue(value, result);
  return false;
}

bool convertToYGJustify(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGJustify> justifyMap = {
      {"flex-start", YGJustifyFlexStart},
      {"center", YGJustifyCenter},
      {"flex-end", YGJustifyFlexEnd},
      {"space-between", YGJustifySpaceBetween},
      {"space-around", YGJustifySpaceAround},
      {"space-evenly", YGJustifySpaceEvenly},
  };
  return lookupAndSetDynamicValue(str, value, justifyMap);
}

bool convertToYGAlign(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGAlign> alignMap = {
      {"auto", YGAlignAuto},
      {"flex-start", YGAlignFlexStart},
      {"center", YGAlignCenter},
      {"flex-end", YGAlignFlexEnd},
      {"stretch", YGAlignStretch},
      {"baseline", YGAlignBaseline},
      {"space-between", YGAlignSpaceBetween},
      {"space-around", YGAlignSpaceAround},
      {"space-evenly", YGAlignSpaceEvenly},
  };
  return lookupAndSetDynamicValue(str, value, alignMap);
}

bool convertToYGOverflow(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGOverflow> overflowMap = {
      {"visible", YGOverflowVisible},
      {"hidden", YGOverflowHidden},
      {"scroll", YGOverflowScroll},
  };
  return lookupAndSetDynamicValue(str, value, overflowMap);
}

bool convertToYGDirection(std::string_view str, DynamicValue &value) {
  static const std::unordered_map<std::string_view, YGDirection> directionMap =
      {
          {"ltr", YGDirectionLTR},
          {"rtl", YGDirectionRTL},
          {"inherit", YGDirectionInherit},
      };
  return lookupAndSetDynamicValue(str, value, directionMap);
}

} // namespace

static ConvertFuncTable &getConvertFuncTable() {
  static ConvertFuncTable convertFuncTable;
  static bool initialized = false;
  if (initialized)
    return convertFuncTable;

  convertFuncTable.resize(size_t(DVKind::COUNT) - 1, nullptr);
  convertFuncTable[size_t(DVKind::Name)] = convertToString;
  convertFuncTable[size_t(DVKind::Width)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Height)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MinWidth)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MaxWidth)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MinHeight)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MaxHeight)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Top)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Left)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Bottom)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Right)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Position)] = convertToYGPositionType;
  convertFuncTable[size_t(DVKind::FlexDirection)] = convertToYGFlexDirection;
  convertFuncTable[size_t(DVKind::FlexGrow)] = convertToFloat;
  convertFuncTable[size_t(DVKind::FlexShrink)] = convertToFloat;
  convertFuncTable[size_t(DVKind::FlexBasis)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Flex)] = convertToFlexResolved;
  convertFuncTable[size_t(DVKind::JustifyContent)] = convertToYGJustify;
  convertFuncTable[size_t(DVKind::Gap)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::RowGap)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::ColumnGap)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::Color)] = convertToColorRGBA32;
  convertFuncTable[size_t(DVKind::BackgroundColor)] = convertToColorRGBA32;
  convertFuncTable[size_t(DVKind::AspectRatio)] = convertToFloat;
  convertFuncTable[size_t(DVKind::HorizontalAlign)] = convertToHorizontalAlign;
  convertFuncTable[size_t(DVKind::VerticalAlign)] = convertToVerticalAlign;
  convertFuncTable[size_t(DVKind::AlignSelf)] = convertToYGAlign;
  convertFuncTable[size_t(DVKind::AlignContent)] = convertToYGAlign;
  convertFuncTable[size_t(DVKind::AlignItems)] = convertToYGAlign;
  convertFuncTable[size_t(DVKind::Overflow)] = convertToYGOverflow;
  convertFuncTable[size_t(DVKind::Direction)] = convertToYGDirection;
  convertFuncTable[size_t(DVKind::PaddingAll)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingLeft)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingTop)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingRight)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingBottom)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingStart)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingEnd)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingHorizontal)] =
      convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::PaddingVertical)] =
      convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderAll)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderLeft)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderTop)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderRight)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderBottom)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderStart)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderEnd)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderHorizontal)] =
      convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::BorderVertical)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginAll)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginLeft)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginTop)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginRight)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginBottom)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginStart)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginEnd)] = convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginHorizontal)] =
      convertToDimensionResolved;
  convertFuncTable[size_t(DVKind::MarginVertical)] = convertToDimensionResolved;

  for (size_t i = 0; i < convertFuncTable.size(); i++) {
    if (convertFuncTable[i] == nullptr) {
      MOTR_LOG("Error: convertFuncTable[DVKind({})={}] = nullptr", i,
               MStringFromDVKind(DVKind(i)).sv());
    }
  }
  initialized = true;

  return convertFuncTable;
}
