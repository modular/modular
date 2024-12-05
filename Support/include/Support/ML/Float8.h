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

namespace Detail {
template <llvm::APFloat::Semantics Semantics>
struct float8_generic_t {
  float8_generic_t(float v) : bits(toBits(v)) {}
  explicit float8_generic_t(uint8_t rawBits) : bits(rawBits) {}

  operator float() {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APInt apInt(8, bits);
    llvm::APFloat apFloat(llvm::APFloat::EnumToSemantics(Semantics), apInt);
    bool ignore;
    apFloat.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &ignore);
    return apFloat.convertToFloat();
  }

  operator uint8_t() const { return bits; }

private:
  static inline uint8_t toBits(float v) {
    // We use APFloat to do the heavy lifting here. This is probably not the
    // most efficient way, but it should be battle tested.
    llvm::APFloat apFloat(v);
    bool ignore;
    apFloat.convert(llvm::APFloat::EnumToSemantics(Semantics),
                    llvm::APFloat::rmNearestTiesToEven, &ignore);

    // APInt will store a uint64_t array, which in this case should be
    // singleton. We will index into this, taking endianness into account.
    const uint64_t *rawData = apFloat.bitcastToAPInt().getRawData();
    constexpr size_t index = M::is_little_endian ? 0 : 7;
    return reinterpret_cast<const uint8_t *>(rawData)[index];
  }

  uint8_t bits;
};
} // namespace Detail

struct float8_e4m3_t : Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3> {};
struct float8_e4m3fnuz_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E4M3FNUZ> {};
struct float8_e5m2_t : Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2> {};
struct float8_e5m2fnuz_t
    : Detail::float8_generic_t<llvm::APFloat::S_Float8E5M2FNUZ> {};

} // namespace M::Float8

#endif // SUPPORT_ML_FLOAT8_H
