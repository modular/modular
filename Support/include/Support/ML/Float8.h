//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_FLOAT8_H
#define SUPPORT_ML_FLOAT8_H

#include "Support/ML/Float16.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/bit.h"

namespace M::Float8 {

struct float8_e4m3_t {
  // https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
  // Minimum negative value found by enabling sign bit on maximum value
  static constexpr uint8_t MIN_BITS = 0xFE;
  static constexpr uint8_t MAX_BITS = 0x7E;

  float8_e4m3_t(float v) : bits(floatToE4M3Bits(v)) {}
  explicit float8_e4m3_t(uint8_t rawBits) : bits(rawBits) {}

  static float8_e4m3_t min() { return float8_e4m3_t(MIN_BITS); }
  static float8_e4m3_t max() { return float8_e4m3_t(MAX_BITS); }

  operator float() {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(8, bits);
    llvm::APFloat apFloat(llvm::APFloat::Float8E4M3(), apInt);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &_);
    return apFloat.convertToFloat();
  }

  operator uint8_t() const { return bits; }

private:
  static inline uint8_t floatToE4M3Bits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool _;
    apFloat.convert(llvm::APFloat::Float8E4M3(),
                    llvm::APFloat::rmNearestTiesToEven, &_);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const uint64_t *rawData = apFloat.bitcastToAPInt().getRawData();
    constexpr size_t index = M::is_little_endian ? 0 : 7;
    return reinterpret_cast<const uint8_t *>(rawData)[index];
  }

  uint8_t bits;
};

struct float8_e5m2_t {
  static constexpr uint8_t MIN_BITS = 0xFB;
  static constexpr uint8_t MAX_BITS = 0x7B;

  float8_e5m2_t(float v) : bits(floatToE5M2Bits(v)) {}
  explicit float8_e5m2_t(uint8_t rawBits) : bits(rawBits) {}

  static float8_e5m2_t min() { return float8_e5m2_t(MIN_BITS); }
  static float8_e5m2_t max() { return float8_e5m2_t(MAX_BITS); }

  operator float() {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(8, bits);
    llvm::APFloat apFloat(llvm::APFloat::Float8E5M2(), apInt);
    bool _;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &_);
    return apFloat.convertToFloat();
  }

  operator uint8_t() const { return bits; }

private:
  static inline uint8_t floatToE5M2Bits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool _;
    apFloat.convert(llvm::APFloat::Float8E5M2(),
                    llvm::APFloat::rmNearestTiesToEven, &_);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const uint64_t *rawData = apFloat.bitcastToAPInt().getRawData();
    constexpr size_t index = M::is_little_endian ? 0 : 7;
    return reinterpret_cast<const uint8_t *>(rawData)[index];
  }

  uint8_t bits;
};

} // namespace M::Float8

#endif // SUPPORT_ML_FLOAT8_H
