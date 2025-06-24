//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Parse.h"
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>

namespace M::motr::Gui::Attribute::Parse {

// Parse width/height values with unit detection
bool parseDimensionValue(std::string_view str, DimensionResolved &result) {
  if (str.empty())
    return false;

  // Check for % suffix
  if (str.back() == '%') {
    std::string numStr(str.substr(0, str.size() - 1));
    char *end;
    result.value = strtof(numStr.c_str(), &end);
    if (end != numStr.c_str()) {
      result.unit = DimensionResolved::Percent;
      return true;
    }
    return false;
  }

  // Check for pt suffix
  if (str.size() > 2 && str.substr(str.size() - 2) == "pt") {
    std::string numStr(str.substr(0, str.size() - 2));
    char *end;
    result.value = strtof(numStr.c_str(), &end);
    if (end != numStr.c_str()) {
      result.unit = DimensionResolved::Point;
      return true;
    }
    return false;
  }

  // Otherwise treat as points
  char *end;
  result.value = strtof(std::string(str).c_str(), &end);
  if (end != str.data()) {
    result.unit = DimensionResolved::Point;
    return true;
  }
  return false;
}

// Hex character to integer
static inline int hexCharToInt(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return 0;
}

static inline bool allHexChars(std::string_view str) {
  for (char c : str) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F')))
      return false;
  }
  return true;
}

static bool parseHexRGB(std::string_view colorStr, Color::RGBA32 &result) {
  if (colorStr.length() != 4)
    return false;

  if (colorStr[0] != '#')
    return false;

  if (!allHexChars(colorStr.substr(1)))
    return false;

  uint8_t r = hexCharToInt(colorStr[1]);
  uint8_t g = hexCharToInt(colorStr[2]);
  uint8_t b = hexCharToInt(colorStr[3]);
  r = (r << 4) | r;
  g = (g << 4) | g;
  b = (b << 4) | b;
  result = {r, g, b, 255};
  return true;
}

static bool parseHexRRGGBB(std::string_view colorStr, Color::RGBA32 &result) {
  if (colorStr.length() != 7)
    return false;

  if (colorStr[0] != '#')
    return false;

  if (!allHexChars(colorStr.substr(1)))
    return false;

  uint8_t r = (hexCharToInt(colorStr[1]) << 4) | hexCharToInt(colorStr[2]);
  uint8_t g = (hexCharToInt(colorStr[3]) << 4) | hexCharToInt(colorStr[4]);
  uint8_t b = (hexCharToInt(colorStr[5]) << 4) | hexCharToInt(colorStr[6]);
  result = {r, g, b, 255};
  return true;
}

bool parseHexRRGGBBAA(std::string_view colorStr, Color::RGBA32 &result) {
  if (colorStr.length() != 9)
    return false;

  if (colorStr[0] != '#')
    return false;

  if (!allHexChars(colorStr.substr(1)))
    return false;

  uint8_t r = (hexCharToInt(colorStr[1]) << 4) | hexCharToInt(colorStr[2]);
  uint8_t g = (hexCharToInt(colorStr[3]) << 4) | hexCharToInt(colorStr[4]);
  uint8_t b = (hexCharToInt(colorStr[5]) << 4) | hexCharToInt(colorStr[6]);
  uint8_t a = (hexCharToInt(colorStr[7]) << 4) | hexCharToInt(colorStr[8]);
  result = {r, g, b, a};
  return true;
}

static std::vector<std::string_view> split(std::string_view str, char delimiter,
                                           size_t maxSplits = 0) {
  std::vector<std::string_view> result;
  auto pos = str.find(delimiter);
  while (pos != std::string_view::npos) {
    result.push_back(str.substr(0, pos));
    str = str.substr(pos + 1);
    if (maxSplits > 0 && result.size() >= maxSplits)
      break;
  }
  result.push_back(str);
  return result;
}

static bool parseFloat(std::string_view str, float &result) {
  char *end;
  result = strtof(std::string(str).c_str(), &end);
  if (end != str.data())
    return true;
  return false;
}

static bool parseRGBAfunc(std::string_view colorStr, Color::RGBA32 &result) {
  if (colorStr.substr(0, 5) != "rgba(" && colorStr.back() != ')')
    return false;

  auto params = colorStr.substr(5, colorStr.length() - 6);
  auto parts = split(params, ',', 3);
  if (parts.size() != 4)
    return false;

  float rf = 0.0f;
  float gf = 0.0f;
  float bf = 0.0f;
  float af = 0.0f;
  if (!parseFloat(parts[0], rf))
    return false;
  if (!parseFloat(parts[1], gf))
    return false;
  if (!parseFloat(parts[2], bf))
    return false;
  if (!parseFloat(parts[3], af))
    return false;

  /*
    rf = rf * 255.0f;
    gf = gf * 255.0f;
    bf = bf * 255.0f;
    af = af * 255.0f;
    */
  af = af * 255.0f;

  rf = rf < 0.0f ? 0.0f : rf > 255.0f ? 255.0f : rf;
  gf = gf < 0.0f ? 0.0f : gf > 255.0f ? 255.0f : gf;
  bf = bf < 0.0f ? 0.0f : bf > 255.0f ? 255.0f : bf;
  af = af < 0.0f ? 0.0f : af > 255.0f ? 255.0f : af;

  result = {static_cast<uint8_t>(rf), static_cast<uint8_t>(gf),
            static_cast<uint8_t>(bf), static_cast<uint8_t>(af)};
  return true;
}

static bool parseColorName(std::string_view colorStr, Color::RGBA32 &result) {
  using namespace std::string_view_literals;
  if (colorStr.empty())
    return false;

  static const std::unordered_map<std::string_view, Color::RGBA32> colorNames =
      {
          // clang-format off
    {"black", {0, 0, 0, 255}},
    {"white", {255, 255, 255, 255}},
    {"transparent", {0, 0, 0, 0}},

    {"blue", {0, 0, 255, 255}},
    {"brown", {165, 42, 42, 255}},
    {"gray", {128, 128, 128, 255}},
    {"green", {0, 255, 0, 255}},
    {"orange", {255, 165, 0, 255}},
    {"pink", {255, 192, 203, 255}},
    {"purple", {128, 0, 128, 255}},
    {"red", {255, 0, 0, 255}},
    {"yellow", {255, 255, 0, 255}},

    {"darkblue", {0, 0, 139, 255}},
    {"darkbrown", {139, 35, 35, 255}},
    {"darkgray", {169, 169, 169, 255}},
    {"darkgreen", {0, 100, 0, 255}},
    {"darkorange", {255, 127, 0, 255}},
    {"darkpink", {255, 105, 180, 255}},
    {"darkpurple", {139, 0, 139, 255}},
    {"darkred", {139, 0, 0, 255}},
    {"darkyellow", {139, 119, 101, 255}},

    {"lightblue", {173, 216, 230, 255}},
    {"lightbrown", {165, 42, 42, 255}},
    {"lightgray", {211, 211, 211, 255}},
    {"lightgreen", {144, 238, 144, 255}},
    {"lightorange", {255, 165, 0, 255}},
    {"lightpink", {255, 192, 203, 255}},
    {"lightpurple", {155, 105, 180, 255}},
    {"lightred", {255, 105, 180, 255}},
    {"lightyellow", {255, 255, 224, 255}},
          // clang-format on
      };

  auto it = colorNames.find(colorStr);
  if (it != colorNames.end()) {
    result = it->second;
    return true;
  }

  return false;
}

bool parseColor(std::string_view colorStr, Color::RGBA32 &result) {
  if (parseHexRGB(colorStr, result))
    return true;
  if (parseHexRRGGBB(colorStr, result))
    return true;
  if (parseHexRRGGBBAA(colorStr, result))
    return true;
  if (parseRGBAfunc(colorStr, result))
    return true;
  if (parseColorName(colorStr, result))
    return true;
  return false;
}

bool parseFlexResolved(std::string_view str, FlexResolved &result) {
  if (str.empty())
    return false;

  auto parts = split(str, ' ', 2);
  if (parts.size() >= 1) {
    if (!parseFloat(parts[0], result.grow))
      result.grow = 0;
  }

  if (parts.size() >= 2) {
    if (!parseFloat(parts[1], result.shrink))
      result.shrink = 0;
  }

  if (parts.size() >= 3) {
    if (!parseDimensionValue(parts[2], result.basis))
      result.basis = DimensionResolved{DimensionResolved::Auto, 0};
  }

  return true;
}
} // namespace M::motr::Gui::Attribute::Parse
