//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a custom subclass of the KGEN ParameterEvaluator class that
// knows how to look up and fold 'apply' parameter expressions to alwaysinline
// functions when invoked with simple constants.  This is important to provide
// type canonicalization in the face of zero cost abstractions like `Int`.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_PARAMETER_EVALUATOR_H
#define LIT_PARAMETER_EVALUATOR_H

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Interpreter/InterpreterInterface.h"

namespace M::KGEN::LIT {
class DeclResolver;
class LitSharedState;

class LitParameterEvaluator : public ParameterEvaluator,
                              public InterpreterState {
public:
  LitParameterEvaluator(LitSharedState &shared);
  LitParameterEvaluator(ArrayRef<ParamBindAttr> paramValues,
                        LitSharedState &shared);

  /// Attempt to evaluate 'apply' expressions.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Lookup the body of the referenced function using the DeclResolver.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

private:
  DeclResolver &resolver;
};

} // namespace M::KGEN::LIT

#endif // LIT_PARAMETER_EVALUATOR_H
