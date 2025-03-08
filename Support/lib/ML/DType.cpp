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

  // Handle floating point types.
#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, CXX_TYPE,      \
                      BITCOUNT, ...)                                           \
  case DType::SHORT_NAME: {                                                    \
    constexpr ssize_t BYTECOUNT = BITCOUNT / CHAR_BIT;                         \
    ssize_t result = numElements * BYTECOUNT;                                  \
    if (result / BYTECOUNT != ssize_t(numElements))                            \
      return -1;                                                               \
    return result;                                                             \
  };
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT

  // Handle other types.
  case DType::kBool:
    widthShift = 0;
    break;
  case DType::tf32: // tf32 has 19bits, store as 4 bytes.
    widthShift = 2;
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
#define DECLARE_FLOAT(SHORT_NAME, ...)                                         \
  if (str == #SHORT_NAME)                                                      \
    return DType(DType::SHORT_NAME);
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
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
    // Handle the bf16 special case, since it's a floating point type which does
    // not start with the letter 'f'.
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
#define DECLARE_FLOAT(SHORT_NAME, ...)                                         \
  case DType::SHORT_NAME:                                                      \
    return #SHORT_NAME;
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
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

/// This method returns the LLVM floating point semantics for the given DType,
/// or nullptr if the DType is not a floating point type LLVM knows about
/// (e.g. TF32).
const llvm::fltSemantics *DType::getFPSemantics() const {
  switch (getValue()) {
  default:
    return nullptr;

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, CXX_TYPE,      \
                      BITCOUNT, APFLOAT_TYPE, ...)                             \
  case DType::SHORT_NAME:                                                      \
    return &APFLOAT_TYPE();
#include "Support/ML/FloatTypes.def"
#undef DECLARE_FLOAT
  }
}
