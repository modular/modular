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
  auto outDType = outType.getResolvedDType();
  if (!outDType || !outDType->isBool())
    return failure();

  if (auto fold = foldSIMDCmp(toCmpPredicate(getCc()), {getLhs(), getRhs()},
                              *outDType, target.resolveIndexBitWidth()))
    return TypedAttr(fold);

  return failure();
}
