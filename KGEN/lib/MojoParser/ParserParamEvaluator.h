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

#ifndef PARSER_PARAM_EVALUATOR_H
#define PARSER_PARAM_EVALUATOR_H

#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"

namespace M::KGEN::LIT {
class DeclResolver;

class ParserParamEvaluator : public ParameterEvaluator,
                             public InterpreterState {
public:
  ParserParamEvaluator(DeclResolver &resolver,
                       ArrayRef<ParamBindAttr> paramValues = {});

  /// Attempt to evaluate 'apply' expressions.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Attempt to evaluate a function call in a parameter context.
  FailureOr<TypedAttr> evaluateFunctionCall(SymbolRefAttr symbol,
                                            ArrayRef<Attribute> arguments);

  /// Lookup the body of the referenced function using the DeclResolver.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

  /// Evaluate all constant 'apply' expressions within a type.
  Type refineType(Type type);

private:
  DeclResolver &resolver;
};

} // namespace M::KGEN::LIT

#endif // PARSER_PARAM_EVALUATOR_H
