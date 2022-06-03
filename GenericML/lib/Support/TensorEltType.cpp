//===- TensorEltType.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/TensorEltType.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace M;

/// Return the in-memory size for an array of the specified type with the
/// specified number of elements, or -1 for non-numeric types or too large
/// values.  This supports densely packed sub-byte types like i1, i2, i4.
ssize_t TensorEltType::getSizeInBytes(size_t numElements) const {
  // Handle complex separately from per-element types below.
  if (isComplex()) {
    ssize_t size = stripComplex().getSizeInBytes(numElements) * 2;
    // If the element type was negative, or the multiply by two overflows,
    // return -1.
    return size >= 0 ? size : -1;
  }

  // This switch handles special cases inline, or determines the logrithmic size
  // of each element and breaks for the overflow check.
  size_t widthShift;
  switch (getValue()) {
  default: {
    // For integers, we just return the bitwidth turned into bytes.  We treat
    /// i1/i2/i4 types as being a single byte.
    if (!isInt())
      return -1;

    widthShift = getIntegerWidthInLogBits();
    // i1,i2,i4 values are packed densely in memory.
    if (widthShift < 3) {
      // We're going to do a truncating division (with a shift right) by the
      // element size, so add 7 to make sure we round up to the next byte.
      size_t numElementsRoundedUp = numElements + 7;
      return numElementsRoundedUp >> (3 - widthShift);
    }

    // Otherwise, we're growing this convert shift amount to byte shift amount.
    widthShift -= 3;
    break;
  }

    // Handle other types.
  case TensorEltType::f8:
  case TensorEltType::kBool:
    widthShift = 0;
    break;
  case TensorEltType::f16:
  case TensorEltType::bf16:
    widthShift = 1;
    break;
  case TensorEltType::f32:
  case TensorEltType::tf32:
    widthShift = 2;
    break;
  case TensorEltType::f64:
    widthShift = 3;
    break;
  case TensorEltType::f80: {
    ssize_t result = numElements * 10;
    if (result / 10 != ssize_t(numElements))
      return -1;
    return result;
  }
  }

  // Check that the result doesn't overflow.
  ssize_t result = numElements << widthShift;
  if (result >> widthShift != ssize_t(numElements))
    return -1;
  return result;
}

/// Return a string form of this eltType suitable for printing and error
/// messages.
std::string TensorEltType::getAsString() const {
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
  default:
    return "eltType<unknown" + llvm::utostr(getValue()) + ">";
  }
}

void TensorEltType::print(raw_ostream &os) const { os << getAsString(); }
void TensorEltType::dump() const { print(llvm::errs()); }