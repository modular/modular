//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "InferenceState.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/SharedState.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

InferenceState::InferenceState(
    ASTDecl &declScope, SharedState &shared, ArrayRef<Type> declaredParamTypes,
    PogListAttr declaredParamPogs,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag)
    : declScope(declScope), shared(shared),
      evaluator(shared.getParameterEvaluator()),
      declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs), getDiag(std::move(getDiag)) {
  for (size_t i = 0; i != declaredParamTypes.size(); ++i)
    evaluator.appendIndexBinding(TypedAttr());
}

LogicalResult InferenceState::setInferredValue(size_t paramIdx,
                                               TypedAttr paramVal) {
  paramVal = evaluator.getReboundAttribute(paramVal);
  ASTType targetType = evaluator.getReboundType(declaredParamTypes[paramIdx]);
  // Type must be equal
  assert(targetType.isEqualCanon(paramVal.getType()));

  // now align sugar
  if (paramVal.getType() != targetType)
    paramVal = ParamOperatorAttr::getRebind(paramVal, targetType);

  evaluator.overwriteIndexBinding(paramIdx, paramVal);

  if (isa<UnboundAttr>(paramVal))
    return success();

  ArrayRef<ConstraintAttr> constraints =
      declaredParamPogs.getPogs()[paramIdx].getConstraints();
  if (constraints.empty())
    return success();

  // Verify all constraints are satisfied, collecting unprovable constraints.
  ConstraintResult result = checkConstraints(
      declScope, declaredParamPogs, constraints, /*origConstraints=*/{},
      getDiag, &unprovableConstraints, &evaluator);

  // TODO: how about we just emitting unprovable error here right away?
  return success(result == ConstraintResult::Satisfied);
}
