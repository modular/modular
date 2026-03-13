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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace M::KGEN {

class ParameterEvaluationContext;

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

//===----------------------------------------------------------------------===//
// Fold helpers
//
// These reduce the boilerplate for connecting ops and attributes to a shared
// fold function.  Each fold function has the canonical signature:
//
//   FoldValue foldFoo(FoldValues operands, TargetInfoAttr target);
//
// The helpers below adapt that signature for the four standard hooks:
//   - Attr::get            (fold-on-construction, no target)
//   - evaluateWithContext  (contextual eval, target from context)
//   - Op::fold             (op folding, target from module)
//   - Op::interpret        (interpreter, target from state)
//===----------------------------------------------------------------------===//

/// The canonical fold function signature accepted by all fold helpers.
using TargetAwareFoldFn = function_ref<FoldValue(FoldValues, TargetInfoAttr)>;

/// Try to fold an attribute during construction (no target info available).
/// A null TargetInfoAttr is passed, relying on the fold function to treat it
/// as "unknown target" and only fold when safe.
inline TypedAttr tryFoldAttr(ArrayRef<Attribute> operands,
                             TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), {})) {
    assert(result.getAttr() && "attribute fold should produce an attribute");
    return result.getAttr();
  }
  return {};
}

/// Evaluate an attribute with context using a target-aware fold function.
/// Returns the folded attribute if target info is available and the fold
/// succeeds, or failure() otherwise.
FailureOr<TypedAttr> foldAttrWithTarget(ParameterEvaluationContext &context,
                                        ArrayRef<Attribute> operands,
                                        TargetAwareFoldFn fold);

/// Fold an op using a target-aware fold function.
inline OpFoldResult foldOpWithTarget(FoldValues operands, TargetInfoAttr target,
                                     TargetAwareFoldFn fold) {
  if (auto result = fold(operands, target))
    return result.asOpFoldResult();
  return {};
}

/// Interpret an op using a target-aware fold function.
ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       InterpreterState &state,
                                       TargetAwareFoldFn fold);
ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state,
                                       TargetAwareFoldFn fold);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_FOLDUTILS_H
