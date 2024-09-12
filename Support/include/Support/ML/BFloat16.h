//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares `BFloat16` which is a 16-bit floating point representation
// used often on NVidia GPUs.
// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
//
// BFloat16 is available on some targets as __bf16 in Clang as a non-standard
// extension. It is being added to C++23 as std::bfloat16_t in the new
// <stdfloat> library. Until support is standardized, we maintain this for
// conversion to/from other types.
//
//
// The format is quite simple: it's a truncated IEEE single-width floating point
// number. The sign and exponent (8 bits) are the same, but instead of a 23 bit
// mantissa, bfloat16 has 7 bits in its mantissa. The result is a much less
// precise number that can represent the same "range" of numbers as a normal 32
// bit float.
//
// Visually:
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp16 |S|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + bfloat16  |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M| | | | | | | | | | | | | | | |
//  |
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  + IEEE fp32
//  |S|E|E|E|E|E|E|E|E|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|
//  +-----------+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
// Note that this implementation _truncates_ when converting from float32. This
// means that compared to torch. we will round some numbers towards zero when
// converting from higher precisions.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_BFLOAT16_H
#define SUPPORT_ML_BFLOAT16_H

#include "llvm/include/llvm/ADT/bit.h"

namespace M {
namespace {

const bool is_little_endian =
    llvm::endianness::native == llvm::endianness::little;

uint16_t float2bf16bits(float v) {
  const size_t index = is_little_endian ? 1 : 0;
  return reinterpret_cast<uint16_t *>(&v)[index];
}

float bf16bits2float(uint16_t bits) {
  const size_t index = is_little_endian ? 1 : 0;
  float result = 0.;
  auto shorts = reinterpret_cast<uint16_t *>(&result);
  shorts[index] = bits;
  return result;
}

} // namespace

namespace bfloat {

struct bfloat16_t {
  bfloat16_t(float v) : bits(float2bf16bits(v)) {}
  operator float() { return bf16bits2float(bits); }

private:
  uint16_t bits;
};

} // namespace bfloat

} // namespace M

#endif // SUPPORT_ML_BFLOAT16_H
