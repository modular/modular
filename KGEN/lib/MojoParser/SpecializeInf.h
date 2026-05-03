//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Specialization inference helpers for closure conformance.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H
#define KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H

#include "InferenceState.h"
#include "OverloadSet.h"

namespace M::KGEN::LIT {

class ExprNode;

class SpecializeInf : public InferenceState {
public:
  SpecializeInf(ASTDecl &declScope, const ExprNode *expr,
                ArrayRef<Type> declaredParamTypes,
                PogListAttr declaredParamPogs, SMLoc defaultLoc,
                bool discardError);

  LogicalResult setInitialInferredValue(size_t paramIdx, TypedAttr paramVal) {
    return setInferredValue(paramIdx, paramVal);
  }

  FailureOr<SmallVector<TypedAttr>>
  inferSpecialization(FnTypeGeneratorType target, FnOp actualFn);

private:
  bool isExplicitlyUnbound(size_t) const override { return false; }

  LogicalResult matchArgument(Type actualType, ArgConvention actualConvention,
                              size_t argIdx, ASTType expectedType,
                              ArgConvention expectedConvention,
                              PogListAttr argPogs);
  LogicalResult matchValueType(ASTType actualType, size_t argIdx,
                               ASTType expectedType, PogListAttr argPogs);

  const ExprNode *expr;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SPECIALIZEINFERENCE_H
