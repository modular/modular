//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
#define KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ForwardDecls.h"

namespace M {
class Error;
} // namespace M

namespace M::KGEN {
class KGENDeclInterface;

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

// NOTE: None of these are particularly efficient, because they walk the whole
// IR tree without caching.

/// Given a parameter expression, walk it and return any references to named
/// parameters.  This fails if an invalid parameter expression exists.
LogicalResult
collectParameterReferences(TypedAttr expr,
                           SmallVector<ParamDeclRefAttr> &results);

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.  This fails if an invalid parameter
/// expression exists.
LogicalResult
collectParameterReferences(Type type, SmallVector<ParamDeclRefAttr> &results);

/// Return true if the attribute is a valid parameter expression.
bool isValidParameterExpr(TypedAttr value);

/// Return true if the specified type contains parameter references, e.g.
/// `!pop.scalar<dt>` returns true, but `!pop.scalar<f32>` returns false.
bool isParameterizedType(Type type);

//===----------------------------------------------------------------------===//
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This typedef represents a generator declaration + a set of input
/// parameters that provide a complete binding for something that can be
/// resolved.
using DeclAndInputParamsPair = std::pair<KGENDeclInterface, ArrayAttr>;

/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify parameter expressions based on those values.
class ParameterEvaluator {
public:
  ParameterEvaluator(SymbolTable *symtab = nullptr) : symtab(symtab) {}

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(StringAttr name, Attribute value) {
    assert(!paramValues.count(name) && "parameter already declared!");
    paramValues[name] = value;
  }
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    setParameterValue(decl.getName(), value);
  }

  /// Set or overwrite the value of a parameter.
  void setOrOverwriteParameterValue(ParamDeclAttr decl, Attribute value) {
    paramValues[decl.getName()] = value;
  }

  /// Iterate over the current parameter values.
  const DenseMap<StringAttr, Attribute> &getParameterValues() const {
    return paramValues;
  }

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant.  This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available).
  ErrorOr<Attribute> concretizeParameterExpr(Attribute expr);

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type);

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.  The substituted attributes are not necessarily fully folded:
  /// for that use concretizeParameterExpr.
  Attribute getReboundAttribute(Attribute attr);

  void dump() const;

private:
  /// Evaluate a potentially symbolic expression.
  Optional<TypedAttr> evaluateSymbolicExpression(ParamOperatorAttr op);

  /// A symbol table to lookup type declarations.
  SymbolTable *symtab;

  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;

  /// This caches attributes and Types with parameter references rebound, and
  /// remembers complex attributes that don't have parameter subexprs (noted as
  /// being rebound to themselves).
  DenseMap<Attribute, Attribute> rewrittenAttrs;
  DenseMap<Type, Type> rewrittenTypes;
};

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
LogicalResult
evaluateConstraints(ConstraintArrayAttr constraints,
                    ParameterEvaluator &evaluator,
                    function_ref<LogicalResult(Location, Error)> emitError);

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
LogicalResult
evaluateConstraints(KGENDeclInterface decl,
                    ArrayRef<Attribute> inputParamValues,
                    function_ref<LogicalResult(Location, Error)> emitError,
                    SymbolTable &symtab);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
