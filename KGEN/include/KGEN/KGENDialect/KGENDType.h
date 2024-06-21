//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the KGEN dtype constants and helpers
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENDTYPE_H
#define KGEN_KGENDIALECT_KGENDTYPE_H

#include "Support/ForwardDecls.h"
#include "Support/ML/DType.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace M::KGEN {

/// KGEN dtype is always compatible with GenericML's DType but has some
/// additional types.
class KGENDType : public DType {
public:
  using DType::DType;

  enum ExtraCases : uint8_t {
    // Represents an address (e.g. a pointer). The size of the address is not
    // specified.
    address = kFirstExtendedOption,
    // Represents a signless integer that has the same size as a pointer.
    index
  };

  KGENDType(DType dtype) : DType(dtype) {}
  KGENDType(ExtraCases type) : DType(type) {}

  constexpr bool isAddress() const { return getValue() == ExtraCases::address; }
  constexpr bool isIndex() const { return getValue() == ExtraCases::index; }

  /// Returns true if the underlying dtype is arithmetic.
  constexpr bool isArithmetic() const {
    return isIndex() || DType::isArithmetic();
  }

  /// Returns true if the underlying dtype is an integer and is signed. The
  /// index dtype is signed.
  constexpr bool isSInt() const { return isIndex() || DType::isSInt(); }

  /// Returns true if the type is any valid integer representation.
  constexpr bool isIntLike() const {
    return isIndex() || isInt() || isBool() || isAddress();
  }

  /// Return the element type for it's string representation.
  static FailureOr<KGENDType> getFromString(StringRef str) {
    if (str == "address")
      return KGENDType(ExtraCases::address);
    if (str == "index")
      return KGENDType(ExtraCases::index);
    auto dtype = DType::getFromString(str);
    if (succeeded(dtype))
      return KGENDType(*dtype);
    return failure();
  }

  /// Return a string form of this eltType suitable for printing and error
  /// messages.
  std::string getAsString() const {
    if (isAddress())
      return "address";
    if (isIndex())
      return "index";
    return DType::getAsString();
  }
};

inline raw_ostream &operator<<(raw_ostream &os, KGENDType value) {
  return os << value.getAsString();
}
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENDTYPE_H
