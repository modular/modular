//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef COLOR_H
#define COLOR_H

#include "motr/Log.h"
#include "motr/Macros.h"
#include <array>
#include <cmath>
#include <cstdint>

struct Color3 {
  float color[3];
};

enum class ColorNames : uint8_t {
  RED,
  GREEN,
  BLUE,
  YELLOW,
  PURPLE,
  ORANGE,
  PINK,
  GRAY,
  WHITE,
  BLACK,
  CYAN,
  MAGENTA,
  LIME,
  MAROON,
  NAVY,
  OLIVE,
};

constexpr Color3 getColor(ColorNames colorName) {
  switch (colorName) {
  case ColorNames::RED:
    return {1.0f, 0.0f, 0.0f};
  case ColorNames::GREEN:
    return {0.0f, 1.0f, 0.0f};
  case ColorNames::BLUE:
    return {0.0f, 0.0f, 1.0f};
  case ColorNames::YELLOW:
    return {1.0f, 1.0f, 0.0f};
  case ColorNames::PURPLE:
    return {1.0f, 0.0f, 1.0f};
  case ColorNames::ORANGE:
    return {1.0f, 0.5f, 0.0f};
  case ColorNames::PINK:
    return {1.0f, 0.0f, 0.5f};
  case ColorNames::GRAY:
    return {0.5f, 0.5f, 0.5f};
  case ColorNames::WHITE:
    return {1.0f, 1.0f, 1.0f};
  case ColorNames::BLACK:
    return {0.0f, 0.0f, 0.0f};
  case ColorNames::CYAN:
    return {0.0f, 1.0f, 1.0f};
  case ColorNames::MAGENTA:
    return {1.0f, 0.0f, 1.0f};
  case ColorNames::LIME:
    return {0.0f, 1.0f, 0.0f};
  case ColorNames::MAROON:
    return {1.0f, 0.0f, 0.0f};
  case ColorNames::NAVY:
    return {0.0f, 0.0f, 1.0f};
  case ColorNames::OLIVE:
    return {0.5f, 0.5f, 0.0f};
  }
  return {0.0f, 0.0f, 0.0f};
}

namespace M::motr::Gui::Color {
struct HSVA {
  float h;
  float s;
  float v;
  float a;
};

struct RGBA {
  float r;
  float g;
  float b;
  float a;
};
MOTR_ALWAYS_INLINE RGBA RGBA32toRGBA(uint32_t color) {
  const uint32_t c = static_cast<uint32_t>(color);
  const uint8_t r = (c >> 24) & 0xFF;
  const uint8_t g = (c >> 16) & 0xFF;
  const uint8_t b = (c >> 8) & 0xFF;
  const uint8_t a = c & 0xFF;
  const float f = 1.0f / 255.0f;
  return {r * f, g * f, b * f, a * f};
}

MOTR_ALWAYS_INLINE uint32_t RGBA32toARGB32(uint32_t color) {
  const uint32_t c = static_cast<uint32_t>(color);
  return (c & 0xFF) << 24 | ((c >> 8) & 0x00FFFFFF);
}

namespace Palette {
// https://iamkate.com/data/12-bit-rainbow/
enum class TwelveBitRainbow : uint32_t {
  DarkMagenta = 0x881177FF,
  DarkPink = 0xaa3355FF,
  LightRed = 0xcc6666FF,
  LightOrange = 0xee9944FF,
  BrightYellow = 0xeedd00FF,
  LightGreen = 0x99dd55FF,
  MediumGreen = 0x44dd88FF,
  Teal = 0x22ccbbFF,
  Cyan = 0x00bbccFF,
  MediumBlue = 0x0099ccFF,
  RoyalBlue = 0x3366bbFF,
  DarkPurple = 0x663399FF
};

constexpr std::array<TwelveBitRainbow, 12> twelveBitRainbowArray = {
    TwelveBitRainbow::DarkMagenta,  TwelveBitRainbow::DarkPink,
    TwelveBitRainbow::LightRed,     TwelveBitRainbow::LightOrange,
    TwelveBitRainbow::BrightYellow, TwelveBitRainbow::LightGreen,
    TwelveBitRainbow::MediumGreen,  TwelveBitRainbow::Teal,
    TwelveBitRainbow::Cyan,         TwelveBitRainbow::MediumBlue,
    TwelveBitRainbow::RoyalBlue,    TwelveBitRainbow::DarkPurple};

template <typename T>
RGBA getColorAt(int index);

template <>
inline RGBA getColorAt<TwelveBitRainbow>(int index) {
  int size = twelveBitRainbowArray.size();
  index = (index % size + size) % size; // Wrap around using modulo
  return RGBA32toRGBA(uint32_t(twelveBitRainbowArray[index]));
}

} // namespace Palette

enum class Names : uint8_t {
  RED,
  GREEN,
  BLUE,
  YELLOW,
  PURPLE,
  ORANGE,
  PINK,
  GRAY,
  WHITE,
  BLACK,
  CYAN,
  MAGENTA,
  LIME,
  MAROON,
  NAVY,
  OLIVE,
};

constexpr RGBA getRGBA(Names colorName) {
  switch (colorName) {
  case Names::RED:
    return {1.0f, 0.0f, 0.0f, 1.0f};
  case Names::GREEN:
    return {0.0f, 1.0f, 0.0f, 1.0f};
  case Names::BLUE:
    return {0.0f, 0.0f, 1.0f, 1.0f};
  case Names::YELLOW:
    return {1.0f, 1.0f, 0.0f, 1.0f};
  case Names::PURPLE:
    return {1.0f, 0.0f, 1.0f, 1.0f};
  case Names::ORANGE:
    return {1.0f, 0.5f, 0.0f, 1.0f};
  case Names::PINK:
    return {1.0f, 0.0f, 0.5f, 1.0f};
  case Names::GRAY:
    return {0.5f, 0.5f, 0.5f, 1.0f};
  case Names::WHITE:
    return {1.0f, 1.0f, 1.0f, 1.0f};
  case Names::BLACK:
    return {0.0f, 0.0f, 0.0f, 1.0f};
  case Names::CYAN:
    return {0.0f, 1.0f, 1.0f, 1.0f};
  case Names::MAGENTA:
    return {1.0f, 0.0f, 1.0f, 1.0f};
  case Names::LIME:
    return {0.0f, 1.0f, 0.0f, 1.0f};
  case Names::MAROON:
    return {1.0f, 0.0f, 0.0f, 1.0f};
  case Names::NAVY:
    return {0.0f, 0.0f, 1.0f, 1.0f};
  case Names::OLIVE:
    return {0.5f, 0.5f, 0.0f, 1.0f};
  }
  return {0.0f, 0.0f, 0.0f, 1.0f};
}

constexpr RGBA getFlameGraphColor(int level) {
  constexpr int numColors = 10;
  constexpr RGBA colors[numColors] = {
      {1.0, 0.0, 0.0, 1.0}, //[0] blue
      {1.0, 0.5, 0.0, 1.0}, //[1] light blue
      {1.0, 1.0, 0.0, 1.0}, //[2] yellow
      {0.0, 1.0, 0.0, 1.0}, //[3] yellow-green
      {0.0, 1.0, 0.5, 1.0}, //[4] yellow
      {0.0, 1.0, 1.0, 1.0}, //[5] green
      {0.0, 0.5, 1.0, 1.0}, //[6] cyan
      {0.0, 0.0, 1.0, 1.0}, //[7] blue
      {0.5, 0.0, 1.0, 1.0}, //[8] purple
      {1.0, 0.0, 1.0, 1.0}, //[9] red
  };

  if (level < 0) {
    level = 0;
  } else if (level >= numColors) {
    level = numColors - 1;
  }

  return colors[level];
}

constexpr HSVA RGBAtoHSVA(RGBA rgba) {
  float r = rgba.r;
  float g = rgba.g;
  float b = rgba.b;

  int max_idx = r > g ? r > b ? 0 : 2 : g > b ? 1 : 2;
  int min_idx = r < g ? r < b ? 0 : 2 : g < b ? 1 : 2;

  float max = max_idx == 0 ? r : max_idx == 1 ? g : b;
  float min = min_idx == 0 ? r : min_idx == 1 ? g : b;
  float delta = max - min;

  const float epsilon = 1 / 255.0f;

  float h = 0;
  if (delta < epsilon) {
    h = 0;
  } else if (max_idx == 0) {
    h = 60 * (int((g - b) / delta) % 6);
  } else if (max_idx == 1) {
    h = 60 * (((b - r) / delta) + 2);
  } else {
    h = 60 * (((r - g) / delta) + 4);
  }

  float s = max < epsilon ? 0 : delta / max;

  float v = max;

  return {h, s, v, rgba.a};
}

constexpr RGBA HSVAtoRGBA(HSVA hsva) {
  float h = hsva.h;
  float s = hsva.s;
  float v = hsva.v;
  float a = hsva.a;

  float c = v * s;
  float x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
  float m = v - c;

  float r = 0;
  float g = 0;
  float b = 0;

  if (h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (h < 300) {
    r = x;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x;
  }

  return {r + m, g + m, b + m, a};
}

inline RGBA cycleSaturation(RGBA rgba, float factor, float amplitude = 1.0) {
  auto hsva = RGBAtoHSVA(rgba);
  constexpr float pi = 3.14159265358979323846;
  float sfactor = sin(factor * 2.0 * pi) * amplitude / 2.0;
  float vfactor = cos(factor * 2.0 * pi) * amplitude / 2.0;
  // MOTR_LOG("s: {}+{}={}, v: {}+{}={}", hsva.s, sfactor, hsva.s + sfactor,
  //          hsva.v, vfactor, hsva.v + vfactor);

  hsva.s = hsva.s + sfactor;
  hsva.v = hsva.v + vfactor;
  hsva.s = hsva.s < 0.0 ? 0.0 : hsva.s > 1.0 ? 1.0 : hsva.s;
  hsva.v = hsva.v < 0.0 ? 0.0 : hsva.v > 1.0 ? 1.0 : hsva.v;
  return HSVAtoRGBA(hsva);
}

struct RGBA32 {
  uint8_t r = 0x00;
  uint8_t g = 0x00;
  uint8_t b = 0x00;
  uint8_t a = 0x00;
};

constexpr ::Color3 toColor3(RGBA rgba) { return {rgba.r, rgba.g, rgba.b}; }

} // namespace M::motr::Gui::Color

#endif // COLOR_H
