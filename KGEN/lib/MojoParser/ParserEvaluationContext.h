//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H
#define KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
class SharedState;

/// An evaluation context that uses the parser's SharedState to evaluate
/// expressions.
class ParserEvaluationContext : public ParameterEvaluationContext {
public:
  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) override;

  /// Simplifying getters for evaluatable attributes. These should be used
  /// instead of directly creating attributes so that immediately simplifiable
  /// attributes do not need to be created in the first place.
  TypedAttr getBindParamsAttr(TypedAttr generator,
                              ArrayRef<TypedAttr> paramValues);

private:
  friend class SharedState;
  ParserEvaluationContext(SharedState &shared) : shared(shared) {}

  FailureOr<TypedAttr> evaluateBindParams(TypedAttr generator,
                                          ArrayRef<TypedAttr> paramValues);

  SharedState &shared;
};

/// A convenience ParameterEvaluator that uses ParserEvaluationContext to
/// evaluate expressions.
class ParserParameterEvaluator : public ParameterEvaluator {
public:
  ParserParameterEvaluator(SharedState &shared) {
    setEvaluationContext(&shared.getEvaluationContext());
  }
  ParserParameterEvaluator(SharedState &shared,
                           ArrayRef<ParamDeclAttr> paramDecls,
                           ArrayRef<TypedAttr> paramValues)
      : ParameterEvaluator(paramDecls, paramValues) {
    setEvaluationContext(&shared.getEvaluationContext());
  }
  ParserParameterEvaluator(SharedState &shared, ArrayRef<TypedAttr> paramValues)
      : ParameterEvaluator(paramValues) {
    setEvaluationContext(&shared.getEvaluationContext());
  }
};
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERPARAMETEREVALUATOR_H
