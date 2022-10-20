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
  };

  KGENDType(DType dtype) : DType(dtype){};
  KGENDType(ExtraCases type) : DType(type){};

  constexpr bool isAddress() const { return getValue() == ExtraCases::address; }

  /// Return the element type for it's string representation.
  static FailureOr<KGENDType> getFromString(StringRef str) {
    if (str == "address")
      return KGENDType(ExtraCases::address);
    auto dtype = DType::getFromString(str);
    if (succeeded(dtype))
      return KGENDType(dtype.value());
    return failure();
  }

  /// Return a string form of this eltType suitable for printing and error
  /// messages.
  std::string getAsString() const {
    if (isAddress())
      return "address";
    return DType::getAsString();
  }
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENDTYPE_H
