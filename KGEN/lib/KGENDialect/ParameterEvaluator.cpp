//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
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
  if (!attr || ParameterAttr::isSimpleConstant(attr))
    return;

  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    results.push_back(paramRef);
    return;
  }

  attr.walkImmediateSubElements(
      [&](Attribute attr) { collectParameterReferences(attr, results); },
      [&](Type type) { collectParameterReferences(type, results); });
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.
void KGEN::collectParameterReferences(Type type,
                                      SmallVector<ParamDeclRefAttr> &results) {
  type.walkImmediateSubElements(
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

ParameterEvaluator::ParameterEvaluator(ArrayRef<ParamBindAttr> paramValues) {
  for (ParamBindAttr bind : paramValues)
    setParameterValue(bind.getName(), bind.getValue());
}

// NOTE: This is out of line to provide a home for the ParameterEvaluator
// vtable.
FailureOr<TypedAttr>
ParameterEvaluator::evaluateExpression(ParamOperatorAttr op) {
  return failure();
}

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
  } else if (isa<MLIROpAttr>(attr)) {
    // Expression functions and MLIR operation expressions are isolated from
    // above, so don't collect from them.
  } else {
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;
    attr.walkImmediateSubElements(
        [&](Attribute attr) { newAttrs.push_back(getReboundAttribute(attr)); },
        [&](Type type) { newTypes.push_back(getReboundType(type)); });
    result = attr.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // If an operator persisted, try to simplify it with the symbol table.
  if (auto op = dyn_cast<ParamOperatorAttr>(result))
    if (FailureOr<TypedAttr> expr = evaluateExpression(op); succeeded(expr))
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
    SmallVector<Attribute> newAttrs;
    SmallVector<Type> newTypes;

    type.walkImmediateSubElements(
        [&](Attribute attr) { newAttrs.push_back(getReboundAttribute(attr)); },
        [&](Type type) { newTypes.push_back(getReboundType(type)); });
    result = type.replaceImmediateSubElements(newAttrs, newTypes);
  }

  return rewrittenTypes[type] = result;
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
