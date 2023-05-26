//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_IREVALUATOR_H
#define KGEN_ELABORATOR_IREVALUATOR_H

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Interpreter/InterpreterInterface.h"

namespace M::KGEN {
class Elaborator;
class FuncOp;

/// This IR evaluator is a parameter evaluator that can work during elaboration
/// to concretize parameter expressions and compute symbolic parameter
/// expressions, such as `apply` on a symbol constant or `get_sizeof` and
/// `get_alignof` a decl type.
class IREvaluator : public ParameterEvaluator, public InterpreterState {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(Elaborator &elaborator,
              DenseMap<StringAttr, Attribute> paramValues =
                  DenseMap<StringAttr, Attribute>());

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available). If `allowUnknown`
  /// is set, only unevaluated parameter operators are rejected.
  ErrorTreeOr<Attribute> concretizeParameterExpr(Location loc, Attribute expr,
                                                 bool allowUnknown = false);
  ErrorTreeOr<Type> concretizeParameterExpr(Location loc, Type expr);

  /// Lookup the body of the referenced function. Ensure the function is
  /// inflated as well.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

  /// Evaluate the function with the provided constant inputs.
  ErrorTreeOr<TypedAttr> evaluateFunction(FuncOp func,
                                          ArrayRef<TypedAttr> inputs);

private:
  Attribute getReboundAttribute(Attribute attr) {
    return ParameterEvaluator::getReboundAttribute(attr);
  }
  Type getReboundType(Type type) {
    return ParameterEvaluator::getReboundType(type);
  }

  /// A reference to the elaborator instance. The elaborator is invoked to
  /// concretize symbol constants prior to interpreting them.
  Elaborator *elaborator;

  /// The contextual location of an error.
  std::optional<Location> errorLoc;
  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;
};

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
std::optional<ErrorTree>
evaluateConstraints(ArrayRef<ConstraintAttr> constraints,
                    IREvaluator &evaluator);

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATOR_H
