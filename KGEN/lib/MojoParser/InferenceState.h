//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared state for parameter inference implementations.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_INFERENCESTATE_H
#define KGEN_MOJOPARSER_INFERENCESTATE_H

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/MojoParser/Constraints.h"
#include "ParserEvaluationContext.h"

#include <cstddef>

namespace M::KGEN::LIT {

class ASTDecl;
class ParamMatcher;
class SharedState;

class InferenceState {
public:
  InferenceState(
      ASTDecl &declScope, SharedState &shared,
      ArrayRef<Type> declaredParamTypes, PogListAttr declaredParamPogs,
      llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag);
  virtual ~InferenceState() = default;

  ASTDecl &getDeclScope() const { return declScope; }
  SharedState &getShared() const { return shared; }

  LogicalResult setInferredValue(size_t paramIdx, TypedAttr paramVal);
  virtual bool isExplicitlyUnbound(size_t paramIdx) const = 0;

  SmallVector<ConstraintAttr> unprovableConstraints;

protected:
  friend class ParamMatcher;

  ASTDecl &declScope;
  SharedState &shared;
  ParameterEvaluator evaluator;
  ParamIndexRefAttrFinder paramFinder;
  ArrayRef<Type> declaredParamTypes;
  PogListAttr declaredParamPogs;
  llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_INFERENCESTATE_H
