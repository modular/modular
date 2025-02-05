//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares `Float16` and `BFloat16` which are 16-bit floating point
// representations. `BFloat16` is often used on NVidia GPUs and other
// accelerators: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format.
// `Float16` is the IEEE "half" precision floating point representation:
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format.
//
// `BFloat16` and `Float16` are available on some targets as `__bf16` and
// `_Float16` in Clang as a non-standard extension. They are being added to
// C++23 as `std::bfloat16_t` and `std::float16_t`, respectively, in the new
// `<stdfloat>` library. Until support is standardized, we maintain this for
// conversion to/from other types.
//
// The `bfloat16` format is quite simple: it's a truncated IEEE single-width
// floating point number. The sign and exponent (8 bits) are the same, but
// instead of a 23-bit mantissa, `bfloat16` has 7 bits in its mantissa. The
// result is a much less precise number that can represent the same "range" of
// numbers as a normal 32-bit `float`.
//
// The IEE `fp16` format has a 5-bit exponent and 10-bit mantissa.
//
// Visually:
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp16
//              |S|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + bfloat16
//              |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp32
//              |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// Note that this `BFloat16` implementation _truncates_ when converting from
// float32. This means that compared to torch, we will round some numbers
// towards zero when converting from higher precisions.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_FLOAT16_H
#define SUPPORT_ML_FLOAT16_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/bit.h"

namespace M {
namespace {

constexpr bool is_little_endian =
    llvm::endianness::native == llvm::endianness::little;

} // namespace

namespace BFloat {

struct bfloat16_t {
  // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
  // Minimum negative value found by enabling sign bit on maximum value
  static constexpr uint16_t MIN_BITS = 0xFF7F;
  static constexpr uint16_t MAX_BITS = 0x7F7F;

  bfloat16_t(float v) : bits(floatToBf16Bits(v)) {}
  explicit bfloat16_t(uint16_t rawBits) : bits(rawBits) {}

  static bfloat16_t min() { return bfloat16_t(MIN_BITS); }
  static bfloat16_t max() { return bfloat16_t(MAX_BITS); }

  operator float() {
    constexpr size_t index = is_little_endian ? 1 : 0;
    float result = 0.;
    auto shorts = reinterpret_cast<uint16_t *>(&result);
    shorts[index] = bits;
    return result;
  }

private:
  static inline uint16_t floatToBf16Bits(float v) {
    constexpr size_t index = is_little_endian ? 1 : 0;
    return reinterpret_cast<uint16_t *>(&v)[index];
  }

  uint16_t bits;
};

} // namespace BFloat

namespace Float16 {
struct float16_t {
  static constexpr uint16_t MIN_BITS = 0xFBFF;
  static constexpr uint16_t MAX_BITS = 0x7BFF;

  float16_t(float v) : bits(floatToF16Bits(v)) {}
  explicit float16_t(uint16_t rawBits) : bits(rawBits) {}

  static float16_t min() { return float16_t(MIN_BITS); }
  static float16_t max() { return float16_t(MAX_BITS); }

  operator float() {
    // If the system is big-endian, reverse the byte order of the bits.
    if constexpr (!is_little_endian)
      bits = (bits >> 8) | (bits << 8);

    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(16, bits);
    llvm::APFloat apFloat(llvm::APFloat::IEEEhalf(), apInt);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &_);
    return apFloat.convertToFloat();
  }

private:
  static inline uint16_t floatToF16Bits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEhalf(),
                    llvm::APFloat::rmNearestTiesToEven, &_);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const uint64_t *rawData = apFloat.bitcastToAPInt().getRawData();
    constexpr size_t index = is_little_endian ? 0 : 3;
    return reinterpret_cast<const uint16_t *>(rawData)[index];
  }

  uint16_t bits;
};

} // namespace Float16

} // namespace M

#endif // SUPPORT_ML_FLOAT16_H
