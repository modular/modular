//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "InferenceState.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/lib/MojoParser/ExprNodes.h"
#include "KGEN/lib/MojoParser/IREmitter.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

OptionalDiag::OptionalDiag(SharedState &shared, SMLoc defaultLoc,
                           bool discardError)
    : discardError(discardError), diag(std::nullopt) {
  getDiagClosure = [=, &shared,
                    this](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    this->diag = shared.emitError(loc ? *loc : defaultLoc);
    return *this->diag;
  };
}

llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc>)>
OptionalDiag::getDiag() {
  return getDiagClosure;
}

InferenceState::InferenceState(ASTDecl &declScope,
                               ArrayRef<Type> declaredParamTypes,
                               PogListAttr declaredParamPogs, SMLoc defaultLoc,
                               bool discardError)
    : declScope(declScope), shared(declScope.getShared()),
      evaluator(shared.getParameterEvaluator()),
      declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs),
      diag(shared, defaultLoc, discardError) {
  for (size_t i = 0; i != declaredParamTypes.size(); ++i)
    evaluator.appendIndexBinding(TypedAttr());
}

LogicalResult InferenceState::setInferredValue(size_t paramIdx,
                                               TypedAttr paramVal,
                                               bool isDefaulted) {
  paramVal = evaluator.getReboundAttribute(paramVal);
  ASTType targetType = evaluator.getReboundType(declaredParamTypes[paramIdx]);

  if (!isDefaulted && LIT::isTypeExpr(paramVal)) {
    IREmitter emitter(declScope, EC_TypeParamValue);
    if (auto nmTarget = ASTType(paramVal).getNonmaterializableTarget(shared)) {
      TypedAttr nmTargetAttr = PValue(nmTarget).get();
      FailureOr<bool> typeUpCastable = IREmitter::canMetaTypeUpCastTo(
          shared, declScope.getLoc(), nmTargetAttr.getType(), targetType,
          &declScope);
      // If the nonmaterializable type can be upcast to the target type, then
      // make sure we infer to the nonmaterializable type:
      //    def example[T: TrivialRegisterPassable](a: T): ...
      //    example(1) # T should be Int, not IntLiteral.
      if (succeeded(typeUpCastable) && typeUpCastable.value()) {
        SyntheticNode expr(declScope.getLoc());
        paramVal = emitter.emitPValue({nmTargetAttr, &expr}, EC_TypeParamValue,
                                      targetType);
        assert(paramVal && "must be convertible");
      }
    }
  }

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
      diag.getDiag(), &unprovableConstraints, &evaluator);

  // TODO: how about we just emitting unprovable error here right away?
  return success(result == ConstraintResult::Satisfied);
}
