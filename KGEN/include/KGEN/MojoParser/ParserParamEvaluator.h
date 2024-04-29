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

#ifndef KGEN_MOJOPARSER_PARSERPARAMEVALUATOR_H
#define KGEN_MOJOPARSER_PARSERPARAMEVALUATOR_H

#include "KGEN/KGENDialect/ParameterEvaluator.h"

namespace M::KGEN::LIT {
class DeclResolver;

class ParserParamEvaluator : public ParameterEvaluator {
public:
  ParserParamEvaluator(DeclResolver &resolver,
                       ArrayRef<ParamDeclAttr> paramDecls,
                       ArrayRef<TypedAttr> paramValues);
  ParserParamEvaluator(DeclResolver &resolver,
                       ArrayRef<TypedAttr> paramValues = {});

  /// Attempt to evaluate 'apply' expressions.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Attempt to evaluate a function call in a parameter context.
  FailureOr<TypedAttr> evaluateFunctionCall(SymbolRefAttr symbol,
                                            ArrayRef<Attribute> arguments);

  /// Evaluate all constant 'apply' expressions within a type.
  Type refineType(Type type);

private:
  DeclResolver &resolver;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERPARAMEVALUATOR_H
