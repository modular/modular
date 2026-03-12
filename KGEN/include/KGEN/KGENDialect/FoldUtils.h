//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for expressing folds over SSA values and typed attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_FOLDUTILS_H
#define KGEN_KGENDIALECT_FOLDUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace M::KGEN {

/// A foldable value that can carry an SSA value, a typed attribute, or both.
/// When both are present, they represent the same logical operand/result.
class FoldValue {
public:
  /// A default-constructed FoldValue is falsey. It can only exist as a return
  /// value from a fold operation, and means that the fold operation did not
  /// produce a value.
  FoldValue() = default;

  /// Create a FoldValue from an SSA value and an optional typed attribute. For
  /// use during op-based folding.
  FoldValue(Value value, TypedAttr attr = {}) : value(value), attr(attr) {}

  /// Create a FoldValue from a typed attribute. For use during attribute-based
  /// folding.
  FoldValue(TypedAttr attr) : attr(attr) {}

  explicit operator bool() const { return value || attr; }

  Value getValue() const { return value; }

  template <typename AttrT = TypedAttr>
  AttrT getAttr() const {
    return dyn_cast_or_null<AttrT>(attr);
  }

  Type getType() const {
    assert(*this && "expected fold value");
    if (value)
      return value.getType();
    return attr.getType();
  }

  OpFoldResult asOpFoldResult() const {
    assert(*this && "expected fold value");
    if (attr)
      return attr;
    return value;
  }

  bool operator==(const FoldValue &other) const {
    if (value && other.value)
      return value == other.value;
    if (attr && other.attr)
      return attr == other.attr;
    return false;
  }

private:
  Value value;
  TypedAttr attr;
};

/// A lightweight view over fold operands carried as attributes plus optional
/// parallel SSA values.
class FoldValues {
public:
  FoldValues(ArrayRef<Attribute> attrs, ValueRange values = {})
      : attrs(attrs), values(values) {
    assert((values.empty() || values.size() == attrs.size()) &&
           "expected one value per attribute");
  }

  size_t size() const { return attrs.size(); }

  ArrayRef<Attribute> getAttrs() const { return attrs; }

  template <typename AttrT = TypedAttr>
  AttrT getAttr(size_t index) const {
    assert(index < size() && "operand index out of bounds");
    return dyn_cast_or_null<AttrT>(attrs[index]);
  }

  Value getValue(size_t index) const {
    assert(index < size() && "operand index out of bounds");
    if (values.empty())
      return {};
    return values[index];
  }

  FoldValue operator[](size_t index) const {
    return FoldValue(getValue(index), getAttr(index));
  }

private:
  ArrayRef<Attribute> attrs;
  ValueRange values;
};

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_FOLDUTILS_H
