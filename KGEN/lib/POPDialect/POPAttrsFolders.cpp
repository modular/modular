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
  llvm_unreachable("should have been folded to kgen.param.expr");
}

//===----------------------------------------------------------------------===//
// SIMDAbsAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDAbsAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getOperand()};
  return foldAttrWithTarget(context, operands, foldSIMDAbs);
}

//===----------------------------------------------------------------------===//
// SIMDDivAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDDivAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getLhs(), getRhs()};
  return foldAttrWithTarget(context, operands, foldSIMDDiv);
}

//===----------------------------------------------------------------------===//
// SIMDFloorDivAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> SIMDFloorDivAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  Attribute operands[] = {getLhs(), getRhs()};
  return foldAttrWithTarget(context, operands, foldSIMDFloorDiv);
}

//===----------------------------------------------------------------------===//
// SIMDShlAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShlAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getVal(), getShft()}; // spellchecker:disable-line
  return foldAttrWithTarget(context, operands, foldSIMDShl);
}

//===----------------------------------------------------------------------===//
// SIMDShrAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDShrAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getVal(), getShft()}; // spellchecker:disable-line
  return foldAttrWithTarget(context, operands, foldSIMDShr);
}
