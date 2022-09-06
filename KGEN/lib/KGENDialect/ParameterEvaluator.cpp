//===- ParameterEvaluator.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

/// Given a parameter expression, walk it and return any references to named
/// parameters.  This fails if an unknown parameter expression exists.
///
/// NOTE: This must be kept in sync with
/// ParameterEvaluator::simplifyParameterExpr.
LogicalResult
KGEN::collectParameterReferences(TypedAttr expr,
                                 SmallVector<ParamDeclRefAttr> &results) {
  // Simple constants don't have parameter references.
  if (isSimpleConstant(expr))
    return success();

  // We can directly substitute declaration references given our known table of
  // bindings.
  if (auto declRef = expr.dyn_cast<ParamDeclRefAttr>()) {
    results.push_back(declRef);
    return success();
  }

  // Dig references out of expressions.
  if (auto oper = expr.dyn_cast<ParamOperatorAttr>()) {
    for (auto value : oper.getOperands()) {
      if (failed(collectParameterReferences(value, results)))
        return failure();
    }
    return success();
  }

  // Dig parameters out of parameterized types.
  if (auto typeConstant = expr.dyn_cast<ParameterizedTypeConstantAttr>())
    return collectParameterReferences(typeConstant.getValue(), results);

  // Otherwise, we don't know how to walk this attribute.
  return failure();
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.  This fails if an invalid parameter
/// expression exists.
LogicalResult
KGEN::collectParameterReferences(Type type,
                                 SmallVector<ParamDeclRefAttr> &results) {
  // These are known leaf types that don't participate with
  // SubElementTypeInterface and have no attributes or types within them.
  if (type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>())
    return success();

  auto itf = type.dyn_cast<mlir::SubElementTypeInterface>();
  if (!itf)
    return success();

  LogicalResult result = success();
  itf.walkImmediateSubElements(
      [&](Attribute attr) {
        // Skip ConcreteTypeConstantAttr's since we know they can never have
        // parameters.
        if (succeeded(result) && !attr.isa<ConcreteTypeConstantAttr>())
          result = collectParameterReferences(attr, results);
      },
      [&](Type type) {
        if (succeeded(result))
          result = collectParameterReferences(type, results);
      });
  return result;
}

/// Return true if the attribute is a valid parameter expression.
///
bool KGEN::isValidParameterExpr(TypedAttr value) {
  SmallVector<ParamDeclRefAttr> paramDecls;
  return succeeded(collectParameterReferences(value, paramDecls));
}

/// Return true if the specified type contains parameter references, e.g.
/// `!meta.scalar<dt>` returns true, but `!meta.scalar<f32>` returns false.
///
/// NOTE: This must be kept in sync with ParameterEvaluator::getReboundType.
///
/// TODO: This isn't an efficient method, it walks the entire type graph without
/// caching.
bool KGEN::isParameterizedType(Type type) {
  SmallVector<ParamDeclRefAttr> paramDecls;
  (void)collectParameterReferences(type, paramDecls);
  return !paramDecls.empty();
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator implementation.
//===----------------------------------------------------------------------===//

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
///
/// Note: this must be kept in sync with KGEN::isValidParameterExpr.
ErrorOr<TypedAttr> ParameterEvaluator::simplifyParameterExpr(TypedAttr expr) {
  // Simple constants don't need simplification.
  if (isSimpleConstant(expr))
    return expr;

  // We can directly substitute declaration references given our known table of
  // bindings.
  if (auto declRef = expr.dyn_cast<ParamDeclRefAttr>()) {
    auto value = paramValues[declRef.getName()];
    assert(value && "Verifier should check that all parameters are defined");
    return value;
  }

  // Simplify operators by recursively simplifying their operands, then
  // refolding the expression.
  if (auto oper = expr.dyn_cast<ParamOperatorAttr>()) {
    SmallVector<TypedAttr> simplifiedOperands;
    for (auto value : oper.getOperands()) {
      auto simplified = simplifyParameterExpr(value);
      if (simplified.isError())
        return simplified;
      simplifiedOperands.push_back(simplified.takeValue());
    }

    // FIXME: 'index' folding should require target information to simplify
    // things like div.
    auto result = ParamOperatorAttr::get(oper.getOpcode(), simplifiedOperands);
    if (!isSimpleConstant(result))
      return Error("could not simplify operator " + getParamAsString(result));
    return result;
  }

  // Simplify the types contained within a parameterized MLIR type constant,
  // folding to a ConcreteTypeConstantAttr.
  if (auto typeConstant = expr.dyn_cast<ParameterizedTypeConstantAttr>())
    return ConcreteTypeConstantAttr::get(
        getReboundType(typeConstant.getValue()));

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return Error("unknown expression to fold: " + getParamAsString(expr));
}

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
ErrorOrSuccess ParameterEvaluator::evaluateConstraints(
    DeclAndInputParamsPair declAndInputParams) {
  KGENDeclInterface decl = declAndInputParams.first;

  // If there are no constraints, we are trivially done.
  ArrayRef<ConstraintAttr> constraints = decl.getConstraints();
  if (constraints.empty())
    return success();

  // Otherwise, we have constraints to evaluate.  Bind each of the input
  // parameter names.
  ParameterEvaluator evaluator;
  auto inputParamDecls = getParamDecls(decl);
  ArrayRef<Attribute> inputParamValues = declAndInputParams.second.getValue();
  assert(inputParamDecls.size() == inputParamValues.size() &&
         "incorrect number of input parameters");
  for (auto [paramDecl, value] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setParameterValue(paramDecl.cast<ParamDeclAttr>(), value);

  // Each constraint must be foldable, and must fold to true.
  for (ConstraintAttr constraint : constraints) {
    auto result = evaluator.simplifyParameterExpr(constraint.getExpr());
    if (failed(result))
      return Error("constraint evaluation failure: " +
                   Twine(result.getError()));
    auto resultInt = (*result).dyn_cast<IntegerAttr>();
    if (!resultInt || resultInt.getValue().getBitWidth() != 1)
      return Error("constraint evaluation didn't return true or false");

    // If this failed, indicate why.
    if (resultInt.getValue().isZero())
      return Error("constraint failed: " + constraint.getMessage().getValue());
  }

  // If we made it this far, then everything folded to true.
  return success();
}

/// Get the specified attribute with any nested parameter expressions
/// rewritten.  This can fail with incompatible IR (not due to expansion
/// errors).
Attribute ParameterEvaluator::getReboundAttribute(Attribute attr) {
  // These are common leaf attributes that we know are never parameterized.
  if (!attr || attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr,
                        DTypeConstantAttr>())
    return attr;

  // If we've already processed this attribute, just reuse the memoized result.
  auto iter = rewrittenAttrs.find(attr);
  if (iter != rewrittenAttrs.end())
    return iter->second;

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (attr.isa<ParamDeclRefAttr, ParamOperatorAttr>()) {
    auto newVal = simplifyParameterExpr(attr);
    if (!newVal.isError())
      result = newVal.takeValue();
  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    // TODO: Capture types using SubElementAttrInterface.
    auto newType = getReboundType(typeAttr.getValue());
    if (newType != typeAttr.getValue())
      result = TypeAttr::get(newType);
  } else if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;
    itf.walkImmediateSubElements(
        [&](Attribute attr) { newAttrs.push_back(getReboundAttribute(attr)); },
        [&](Type type) { newTypes.push_back(getReboundType(type)); });
    result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  }

  return rewrittenAttrs[attr] = result;
}

/// Get the specified type with any nested parameter expressions rewritten. This
/// can fail with incompatible IR (not due to expansion errors).  In that case,
/// an error is emitted at the specified location and the type is returned
/// unmodified.
///
/// NOTE: This must be kept in sync with KGEN::isParameterizedType.
///
Type ParameterEvaluator::getReboundType(Type type) {
  // These are known leaf types that don't participate with
  // SubElementTypeInterface and have no attributes or types within them.
  if (type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType,
               MLIRTypeType>())
    return type;

  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewrittenTypes.find(type);
  if (iter != rewrittenTypes.end())
    return iter->second;

  Type result = type;

  // Rebind types in aggregates that implement SubElementTypeInterface.
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;

    itf.walkImmediateSubElements(
        [&](Attribute attr) { newAttrs.push_back(getReboundAttribute(attr)); },
        [&](Type type) { newTypes.push_back(getReboundType(type)); });
    result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  }

  return rewrittenTypes[type] = result;
}
