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

  /// Try to dig out a direct callee, seeing through rebinds that are emitted
  /// due to lifetimes.
  static SymbolConstantAttr findDirectCallee(TypedAttr callee);

  /// Attempt to evaluate 'apply' expressions.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Attempt to evaluate a function call in a parameter context.
  FailureOr<TypedAttr> evaluateFunctionCall(SymbolRefAttr symbol,
                                            ArrayRef<Attribute> arguments);

  /// Evaluate all constant 'apply' expressions within a type.
  Type refine(Type type);
  /// Evaluate all constant 'apply' expressions within an attribute.
  Attribute refine(Attribute attr);

private:
  template <typename T>
  T refineImpl(T arg);

  DeclResolver &resolver;
  /// Cache intermediate refine results.
  DenseMap<const void *, const void *> refineCache;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERPARAMEVALUATOR_H
