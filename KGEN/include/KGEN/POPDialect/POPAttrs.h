//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_H
#define KGEN_POPDIALECT_POPATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/POPDialect/POPEnums.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/APSInt.h"

//===----------------------------------------------------------------------===//
// DTypeValue
//===----------------------------------------------------------------------===//

namespace M::KGEN::POP {
/// This class stores a value of a particular dtype. It supports containing
/// integer, index, float, and bool dtype values only. Index values are treated
/// as signed.
class DTypeValue {
public:
  /// Returns true if this is a supported float dtype.
  static bool isValidFloatDType(KGENDType dtype);

  /// Get the floating semantics for the given dtype.
  static const llvm::fltSemantics &getFloatSemantics(KGENDType dtype);

  /// Get an integer value.
  DTypeValue(APSInt value, KGENDType dtype);

  /// Get a floating point value.
  DTypeValue(APFloat value, KGENDType dtype);

  /// Get a bool value.
  DTypeValue(bool value, KGENDType dtype);

  /// Get an index value.
  DTypeValue(int64_t value, KGENDType dtype);

  /// Raw data constructor.
  DTypeValue(APInt data, KGENDType dtype);

  /// Compare two dtype values.
  bool operator==(const DTypeValue &rhs) const {
    return std::tie(dtype, data) == std::tie(rhs.dtype, rhs.data);
  }

  /// Get the underlying data.
  const APInt &getData() const { return data; }

  /// Get the dtype.
  KGENDType getDType() const { return dtype; }

  /// Get the value as an integer.
  APSInt getIntVal() const;

  /// Get the value as a float.
  APFloat getFloatVal() const;

  /// Get the value as a bool
  bool getBoolVal() const;

  /// Get the value as an index.
  int64_t getIndexVal() const;

private:
  /// Default constructor accessible only by the attribute storage class.
  DTypeValue() {}
  friend class llvm::OwningArrayRef<DTypeValue>;

  /// All values are stored as `APInt`s.
  APInt data;

  /// The dtype of the value. This indicates how to interpret `data`.
  KGENDType dtype;
};

namespace detail {
struct SIMDAttrStorage;
} // namespace detail
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.h.inc"

#endif // GEN_POPDIALECT_POPATTRS_H
