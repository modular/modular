//===- KGEN/KGENDialect/ParameterEvaluator.h ------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
#define KGEN_KGENDIALECT_PARAMETEREVALUATOR_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/ForwardDecls.h"

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
/// `!meta.scalar<dt>` returns true, but `!meta.scalar<f32>` returns false.
bool isParameterizedType(Type type);

//===----------------------------------------------------------------------===//
// ParameterEvaluator
//===----------------------------------------------------------------------===//

/// This typedef represents a kernel/generator declaration + a set of input
/// parameters that provide a complete binding for something that can be
/// resolved.
using DeclAndInputParamsPair = std::pair<KGENDeclInterface, ArrayAttr>;

/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify parameter expressions based on those values.
class ParameterEvaluator {
public:
  ParameterEvaluator() = default;
  ParameterEvaluator(ParameterEvaluator &&) = default;
  ParameterEvaluator(const ParameterEvaluator &) = default;

  /// Given a generator or interface declaration operation, evaluate any
  /// constraints against inputParamValues.  If the constraints are met, return
  /// success, otherwise return why they aren't.
  static ErrorOrSuccess
  evaluateConstraints(DeclAndInputParamsPair declAndInputParams);

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    assert(!paramValues.count(decl.getName()) && "parameter already declared!");
    paramValues[decl.getName()] = value;
  }

  /// Given a generic parameter expression, simplify it by folding the
  /// expression according to known parameter values.  This returns an error if
  /// the expression cannot be folded for one reason or another.
  ErrorOr<TypedAttr> simplifyParameterExpr(TypedAttr expr);

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.  This can fail with incompatible IR (not due to expansion
  /// errors).  In that case, an error is emitted at the specified location and
  /// the attribute is returned unmodified.
  Attribute getReboundAttribute(Attribute attr);

  /// Get the specified type with any nested parameter expressions rewritten.
  /// This can fail with incompatible IR (not due to expansion errors).  In that
  /// case, an error is emitted at the specified location and the type is
  /// returned unmodified.
  Type getReboundType(Type type);

private:
  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;

  /// This caches attributes and Types with parameter references rebound, and
  /// remembers complex attributes that don't have parameter subexprs (noted as
  /// being rebound to themselves).
  DenseMap<Attribute, Attribute> rewrittenAttrs;
  DenseMap<Type, Type> rewrittenTypes;
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PARAMETEREVALUATOR_H
