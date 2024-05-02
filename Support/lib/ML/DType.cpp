//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ML/DType.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
using namespace M;

/// Return the in-memory size for an array of the specified type with the
/// specified number of elements, or -1 for non-numeric types or too large
/// values.  This supports densely packed sub-byte types like i1, i2, i4.
ssize_t DType::getSizeInBytes(size_t numElements) const {
  // Handle complex separately from per-element types below.
  if (isComplex()) {
    ssize_t size = stripComplex().getSizeInBytes(numElements) * 2;
    // If the element type was negative, or the multiply by two overflows,
    // return -1.
    return size >= 0 ? size : -1;
  }

  // This switch handles special cases inline, or determines the logarithmic
  // size of each element and breaks for the overflow check.
  size_t widthShift;
  switch (getValue()) {
  default: {
    // For integers, we just return the bit-width turned into bytes.  We treat
    // i1/i2/i4 types as being a single byte.
    if (!isInt())
      return -1;

    widthShift = getIntegerWidthInLogBits();
    // i1,i2,i4 values are packed densely in memory.
    if (widthShift < 3) {
      // We're going to do a truncating division (with a shift right) by the
      // element size, so make sure we round up to the next byte.
      return llvm::divideCeil(numElements << widthShift, CHAR_BIT);
    }

    // Otherwise, we're growing this convert shift amount to byte shift amount.
    widthShift -= 3;
    break;
  }

    // Handle other types.
  case DType::f8:
  case DType::kBool:
    widthShift = 0;
    break;
  case DType::f16:
  case DType::bf16:
    widthShift = 1;
    break;
  case DType::f24: {
    ssize_t result = numElements * 3;
    if (result / 3 != ssize_t(numElements))
      return -1;
    return result;
  }
  case DType::f32:
  case DType::tf32: // tf32 has 19bits, store as 4 bytes.
    widthShift = 2;
    break;
  case DType::f64:
    widthShift = 3;
    break;
  case DType::f80: {
    ssize_t result = numElements * 10;
    if (result / 10 != ssize_t(numElements))
      return -1;
    return result;
  }
  case DType::f128:
    widthShift = 4;
    break;
  }

  // Check that the result doesn't overflow.
  ssize_t result = numElements << widthShift;
  if (result >> widthShift != ssize_t(numElements))
    return -1;
  return result;
}

/// Return a complex type if it is valid, otherwise fail.
FailureOr<DType> DType::getComplexChecked(DType eltType) {
  if (eltType.getWidthInBits() < 8 || eltType.isComplex())
    return failure();
  return getComplex(eltType);
}

/// This turns the printed form of a dtype back into a DType or
/// returns None if it is an unrecognized name.
FailureOr<DType> DType::getFromString(StringRef str) {
  if (str.empty())
    return failure();
  switch (str[0]) {
  case 'f':
    if (str == "f32")
      return DType(f32);
    if (str == "f64")
      return DType(f64);
    if (str == "f16")
      return DType(f16);
    if (str == "f80")
      return DType(f80);
    if (str == "f24")
      return DType(f24);
    if (str == "f8")
      return DType(f8);
    if (str == "f128")
      return DType(f128);
    return failure();
  case 'u':
  case 's':
    if (str.size() >= 3 && str[1] == 'i') {
      unsigned width = 0;
      if (str.drop_front(2).getAsInteger(10, width))
        return failure();
      return getInt(width, /*isSigned=*/str[0] == 's');
    }
    return failure();

  case 'b':
    if (str == "bool")
      return DType(kBool);
    if (str == "bf16")
      return DType(bf16);
    return failure();
  case 'c':
    if (str.starts_with("complex<") && str.back() == '>') {
      auto elt = getFromString(str.drop_front(8).drop_back());
      if (failed(elt))
        return failure();
      return getComplexChecked(*elt);
    }
    return failure();
  case 't':
    if (str == "tf32")
      return DType(tf32);
    return failure();
  case 'i':
    if (str == "invalid")
      return DType(invalid);
    return failure();
  default:
    // TODO: Could handle the eltType<unknown42> syntax if we wanted to.
    return failure();
  }
}

/// Return a string form of this eltType suitable for printing and error
/// messages.
std::string DType::getAsString() const {
  if (isComplex())
    return "complex<" + stripComplex().getAsString() + ">";
  if (isUInt())
    return "ui" + llvm::utostr(getIntegerWidthInBits());
  if (isSInt())
    return "si" + llvm::utostr(getIntegerWidthInBits());

  switch (getValue()) {
  case f8:
    return "f8";
  case f16:
    return "f16";
  case f32:
    return "f32";
  case f64:
    return "f64";
  case f128:
    return "f128";
  case bf16:
    return "bf16";
  case f24:
    return "f24";
  case f80:
    return "f80";
  case tf32:
    return "tf32";
  case kBool:
    return "bool";
  case invalid:
    return "invalid";
  default:
    return "eltType<unknown" + llvm::utostr(getValue()) + ">";
  }
}

void DType::print(raw_ostream &os) const { os << getAsString(); }
void DType::dump() const { print(llvm::errs()); }

ErrorOr<std::pair<int32_t, int32_t>> DType::getMaxAndMinValue() const {
  return dispatch<ErrorOr<std::pair<int32_t, int32_t>>>()
      .when<DType::si32>([&]() {
        return std::pair(std::numeric_limits<int32_t>::max(),
                         std::numeric_limits<int32_t>::min());
      })
      .when<DType::si16>([&]() {
        return std::pair(std::numeric_limits<int16_t>::max(),
                         std::numeric_limits<int16_t>::min());
      })
      .when<DType::ui16>([&]() {
        return std::pair(std::numeric_limits<uint16_t>::max(),
                         std::numeric_limits<uint16_t>::min());
      })
      .when<DType::si8>([&]() {
        return std::pair(std::numeric_limits<int8_t>::max(),
                         std::numeric_limits<int8_t>::min());
      })
      .when<DType::ui8>([&]() {
        return std::pair(std::numeric_limits<uint8_t>::max(),
                         std::numeric_limits<uint8_t>::min());
      })
      .otherwise([&]() {
        return Error("Unsupported quantization dtype " + getAsString());
      });
}
