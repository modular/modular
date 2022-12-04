//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/KGENDialect/KGENUtils.h"
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
void KGEN::collectParameterReferences(Attribute attr,
                                      SmallVector<ParamDeclRefAttr> &results) {
  // We know that simple constants (including concrete type constants) don't
  // have parameter references in them. Walk over them.
  if (!attr || isSimpleConstant(attr))
    return;

  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    results.push_back(paramRef);
    return;
  }

  auto itf = dyn_cast<mlir::SubElementAttrInterface>(attr);
  if (!itf)
    return;
  itf.walkImmediateSubElements(
      [&](Attribute attr) { collectParameterReferences(attr, results); },
      [&](Type type) { collectParameterReferences(type, results); });
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.
void KGEN::collectParameterReferences(Type type,
                                      SmallVector<ParamDeclRefAttr> &results) {
  auto itf = dyn_cast_or_null<mlir::SubElementTypeInterface>(type);
  if (!itf)
    return;
  itf.walkImmediateSubElements(
      [&](Attribute attr) { collectParameterReferences(attr, results); },
      [&](Type type) { collectParameterReferences(type, results); });
}

/// Return true if the specified type contains parameter references, e.g.
/// `!pop.scalar<dt>` returns true, but `!pop.scalar<f32>` returns false.
///
/// NOTE: This must be kept in sync with ParameterEvaluator::getReboundType.
///
/// TODO: This isn't an efficient method, it walks the entire type graph without
/// caching.
bool KGEN::isParameterizedType(Type type) {
  SmallVector<ParamDeclRefAttr> paramDecls;
  collectParameterReferences(type, paramDecls);
  return !paramDecls.empty();
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator core implementation.
//===----------------------------------------------------------------------===//

/// Get the specified attribute with any nested parameter expressions rewritten.
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
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    result = paramValues[declRef.getName()];
    assert(result && "Verifier should check that all parameters are defined");
  } else if (isa<ExprFuncAttr>(attr)) {
    // Expression functions are isolated from above, so don't collect from them.
  } else if (auto itf = dyn_cast<mlir::SubElementAttrInterface>(attr)) {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;
    itf.walkImmediateSubElements(
        [&](Attribute attr) { newAttrs.push_back(getReboundAttribute(attr)); },
        [&](Type type) { newTypes.push_back(getReboundType(type)); });
    result = itf.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // If an operator persisted, try to simplify it with the symbol table.
  if (symtab && isa<ParamOperatorAttr>(result))
    if (Optional<TypedAttr> expr =
            evaluateSymbolicExpression(cast<ParamOperatorAttr>(result)))
      result = *expr;

  return rewrittenAttrs[attr] = result;
}

/// Get the specified type with any nested parameter expressions rewritten.
Type ParameterEvaluator::getReboundType(Type type) {
  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewrittenTypes.find(type);
  if (iter != rewrittenTypes.end())
    return iter->second;

  Type result = type;

  // Rebind types in aggregates that implement SubElementTypeInterface.
  auto signature = dyn_cast<SignatureType>(type);
  // Signature types with input parameters are special because they are
  // "isolated from above" with respect to their contexts, so we don't rebind
  // within them.
  if (!signature || signature.getInputParams().empty()) {
    if (auto itf = dyn_cast<mlir::SubElementTypeInterface>(type)) {
      SmallVector<Attribute> newAttrs;
      SmallVector<Type> newTypes;

      itf.walkImmediateSubElements(
          [&](Attribute attr) {
            newAttrs.push_back(getReboundAttribute(attr));
          },
          [&](Type type) { newTypes.push_back(getReboundType(type)); });
      result = itf.replaceImmediateSubElements(newAttrs, newTypes);
    }
  }

  return rewrittenTypes[type] = result;
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorOr<Attribute> ParameterEvaluator::concretizeParameterExpr(Attribute expr) {
  assert(symtab && "must have a symbol table to concretize expressions");

  // If we can fold this to a simple constant result, do.
  auto result = getReboundAttribute(expr);
  if (isSimpleConstant(result))
    return result;

  // If this was an unfoldable operator expression, error.  This can happen for
  // things like 'index' arithmetic that has target-specific results.
  if (auto oper = dyn_cast<ParamOperatorAttr>(result))
    return Error("could not simplify operator " + getParamAsString(result));

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return Error("unknown expression to fold: " + getParamAsString(result));
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator debugging support.
//===----------------------------------------------------------------------===//

// Note: this dumps out in non-stable hash table order, only use for debugging
// purposes!
void ParameterEvaluator::dump() const {
  auto &os = llvm::errs();
  os << "ParameterEvaluator: \n";
  for (auto [name, value] : paramValues)
    os << "  " << name << " = " << value << "\n";
}

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
LogicalResult KGEN::evaluateConstraints(
    ConstraintArrayAttr constraints, ParameterEvaluator &evaluator,
    function_ref<LogicalResult(Location, Error)> emitError) {
  // Each constraint must be foldable, and must fold to true.
  for (ConstraintAttr constraint : constraints) {
    ErrorOr<Attribute> result =
        evaluator.concretizeParameterExpr(constraint.getExpr());
    if (failed(result)) {
      return emitError(constraint.getLoc(), "constraint evaluation failure: " +
                                                Twine(result.getError()));
    }

    auto resultInt = dyn_cast<IntegerAttr>(result.takeValue());
    if (!resultInt || resultInt.getValue().getBitWidth() != 1) {
      return emitError(constraint.getLoc(),
                       "constraint evaluation didn't return true or false");
    }

    // If this failed, indicate why.
    if (resultInt.getValue().isZero()) {
      return emitError(constraint.getLoc(),
                       "constraint failed: " +
                           constraint.getMessage().getValue());
    }
  }

  // If we made it this far, then everything folded to true.
  return success();
}

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
LogicalResult KGEN::evaluateConstraints(
    KGENDeclInterface decl, ArrayRef<Attribute> inputParamValues,
    function_ref<LogicalResult(Location, Error)> emitError,
    SymbolTable &symtab) {
  // If there are no constraints, we are trivially done.
  ConstraintArrayAttr constraints = decl.getConstraintsAttr();
  if (!constraints || constraints.empty())
    return success();

  // Otherwise, we have constraints to evaluate.  Bind each of the input
  // parameter names.
  ParameterEvaluator evaluator(&symtab);
  auto inputParamDecls = decl.getInputParamDeclsAttr();
  assert(inputParamDecls.size() == inputParamValues.size() &&
         "incorrect number of input parameters");
  for (auto [paramDecl, value] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setParameterValue(paramDecl, value);

  return evaluateConstraints(constraints, evaluator, std::move(emitError));
}
