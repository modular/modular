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
// SIMDAbsAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
SIMDAbsAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  Attribute operands[] = {getOperand()};
  return foldAttrWithTarget(context, operands, foldSIMDAbs);
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
