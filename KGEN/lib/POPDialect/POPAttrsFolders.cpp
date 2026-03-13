//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains evaluation/folding implementations for POP attributes.
// These methods implement
// ContextuallyEvaluatedAttrInterface::evaluateWithContext.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPUtils.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// SIMDCmpAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDCmpAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  // The default getter already performs target-agnostic folding.
  // This contextual evaluator only handles cases where the target is known,
  // enabling index-type comparisons that require a concrete index bit width.
  auto target = context.getTargetInfo();
  if (!target)
    return failure();

  auto outType = cast<SIMDType>(getType());
  Attribute operands[] = {getLhs(), getRhs()};
  if (auto fold = foldSIMDCmp(toCmpPredicate(getCc()), FoldValues(operands),
                              outType, target)) {
    assert(fold.getAttr() && "attribute fold should produce an attribute");
    return fold.getAttr();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SIMDAbsAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDAbsAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  auto target = context.getTargetInfo();
  if (!target)
    return failure();

  Attribute operands[] = {getOperand()};
  if (auto fold = foldSIMDAbs(FoldValues(operands), target)) {
    assert(fold.getAttr() && "attribute fold should produce an attribute");
    return fold.getAttr();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SIMDShlAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShlAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  auto target = context.getTargetInfo();
  if (!target)
    return failure();

  Attribute operands[] = {getVal(), getShft()};
  if (auto fold = foldSIMDShl(FoldValues(operands), target)) {
    assert(fold.getAttr() && "attribute fold should produce an attribute");
    return fold.getAttr();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SIMDShrAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShrAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  auto target = context.getTargetInfo();
  if (!target)
    return failure();

  Attribute operands[] = {getVal(), getShft()};
  if (auto fold = foldSIMDShr(FoldValues(operands), target)) {
    assert(fold.getAttr() && "attribute fold should produce an attribute");
    return fold.getAttr();
  }

  return failure();
}
