//===- ParameterEvaluator.cpp ---------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

static std::string getString(Attribute attr) {
  std::string str;
  llvm::raw_string_ostream(str) << attr;
  return str;
}

LogicalResult ParameterEvaluator::collectParameterReferences(
    Attribute expr, SmallVector<ParamDeclRefAttr> &results) {
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
  // Otherwise, we don't know how to walk this attribute.
  return failure();
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
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
      return Error("could not simplify operator " + getString(expr));
    return result;
  }

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return Error("unknown expression to fold: " + getString(expr));
}

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
ErrorOrSuccess ParameterEvaluator::evaluateConstraints(
    GeneratorAndInputParamsPair declAndInputParams) {
  Operation *decl = declAndInputParams.first;

  // If there are no constraints, we are trivially done.
  auto constraints = getDeclConstraints(decl);
  if (constraints.empty())
    return success();

  // Otherwise, we have constraints to evaluate.  Bind each of the input
  // parameter names.
  ParameterEvaluator evaluator;
  auto inputParamDecls = getDeclParameterInfo(decl).first;
  ArrayRef<Attribute> inputParamValues = declAndInputParams.second.getValue();
  assert(inputParamDecls.size() == inputParamValues.size() &&
         "incorrect number of input parameters");
  for (auto [decl, value] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setParameterValue(decl.cast<ParamDeclAttr>(), value);

  // Each constraint must be foldable, and must fold to true.
  for (auto [expr, message] : constraints) {
    auto result = evaluator.simplifyParameterExpr(expr);
    if (failed(result))
      return Error("constraint evaluation failure: " +
                   Twine(result.getError()));
    auto resultInt = (*result).dyn_cast<IntegerAttr>();
    if (!resultInt || resultInt.getValue().getBitWidth() != 1)
      return Error("constraint evaluation didn't return true or false");

    // If this failed, indicate why
    if (resultInt.getValue().isZero())
      return Error("constraint failed: " + message.getValue());
  }

  // If we made it this far, then everything folded to true.
  return success();
}

/// Get the specified attribute with any nested parameter expressions
/// rewritten.  This can fail with incompatible IR (not due to expansion
/// errors).  In that case, an error is emitted at the specified location and
/// the attribute is returned unmodified.
Attribute ParameterEvaluator::getReboundAttribute(Attribute attr,
                                                  Location loc) {
  // These are common leaf attributes that we know are never parameterized.
  if (!attr || attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr,
                        DTypeConstantAttr>())
    return attr;

  // If we've already processed this attribute, just reuse the memoized result.
  auto iter = rewrittenAttrs.find(attr);
  if (iter != rewrittenAttrs.end())
    return iter->second;

  // Otherwise, check the type of typed attributes.
  if (auto typedAttr = attr.dyn_cast<TypedAttr>()) {
    if (getReboundType(typedAttr.getType(), loc) != typedAttr.getType()) {
      emitError(loc, "unsupported parameterized type in attribute ") << attr;
      return rewrittenAttrs[attr] = attr;
    }
  }

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (attr.isa<ParamDeclRefAttr, ParamOperatorAttr>()) {
    auto newVal = simplifyParameterExpr(attr);
    if (!newVal.isError())
      result = newVal.takeValue();

  } else if (auto typeAttr = attr.dyn_cast<TypeAttr>()) {
    // TODO: Capture types using SubElementAttrInterface.
    Type newType = getReboundType(typeAttr.getValue(), loc);
    if (newType != typeAttr.getValue())
      result = TypeAttr::get(newType);
  } else if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          newAttrs.push_back(getReboundAttribute(attr, loc));
        },
        [&](Type type) { newTypes.push_back(getReboundType(type, loc)); });
    result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  } else {
    emitError(loc, "unknown attribute in parameterized operation ") << attr;
  }

  return rewrittenAttrs[attr] = result;
}

/// Get the specified type with any nested parameter expressions rewritten. This
/// can fail with incompatible IR (not due to expansion errors).  In that case,
/// an error is emitted at the specified location and the type is returned
/// unmodified.
Type ParameterEvaluator::getReboundType(Type type, Location loc) {
  // These are known leaf types that don't participate with
  // SubElementTypeInterface and have no attributes or types within them.
  if (type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>())
    return type;

  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewrittenTypes.find(type);
  if (iter != rewrittenTypes.end())
    return iter->second;

  Type result = type;

  // Rebind types in aggregates that we know, but otherwise just scan the types
  // that we don't.
  if (auto fnType = type.dyn_cast<FunctionType>()) {
    // Function types show up in kernel signatures.
    // TODO: Capture function subtypes using SubElementAttrInterface.
    SmallVector<Type> inputs, results;
    bool changed = false;
    for (Type input : fnType.getInputs()) {
      inputs.push_back(getReboundType(input, loc));
      changed |= inputs.back() != input;
    }
    for (Type result : fnType.getResults()) {
      results.push_back(getReboundType(result, loc));
      changed |= results.back() != result;
    }

    if (changed)
      result = FunctionType::get(type.getContext(), inputs, results);

  } else if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          newAttrs.push_back(getReboundAttribute(attr, loc));
        },
        [&](Type type) { newTypes.push_back(getReboundType(type, loc)); });
    result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  } else {
    emitError(loc, "unknown type in parameterized operation ") << type;
  }

  return rewrittenTypes[type] = result;
}
