//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_H
#define KGEN_POPDIALECT_POPATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "llvm/ADT/APSInt.h"

//===----------------------------------------------------------------------===//
// DTypeValue
//===----------------------------------------------------------------------===//

namespace M::KGEN::POP {
/// This class stores a value of a particular dtype. It supports containing
/// integer, float, and bool dtype values only.
class DTypeValue {
public:
  /// Get an integer value.
  DTypeValue(APSInt value, DType dtype);

  /// Get a floating point value.
  DTypeValue(APFloat value, DType dtype);

  /// Get a bool value.
  DTypeValue(bool value, DType dtype);

  /// Compare two dtype values.
  bool operator==(const DTypeValue &rhs) const {
    return std::tie(dtype, data) == std::tie(rhs.dtype, rhs.data);
  };

  /// Get the underlying data.
  const APInt &getData() const { return data; }

  /// Get the dtype.
  DType getDType() const { return dtype; }

  /// Get the value as an integer.
  APSInt getIntVal() const;

  /// Get the value as a float.
  APFloat getFloatVal() const;

  /// Get the value as a bool
  bool getBoolVal() const;

private:
  /// Private constructor.
  DTypeValue(APInt data, DType dtype) : data(std::move(data)), dtype(dtype) {}

  /// All values are stored as `APInt`s.
  APInt data;

  /// The dtype of the value. This indicates how to interpret `data`.
  DType dtype;
};
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.h.inc"

#endif // GEN_POPDIALECT_POPATTRS_H
