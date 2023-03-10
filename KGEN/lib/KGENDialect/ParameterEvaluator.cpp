//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/ErrorOr.h"
#include "Support/TimeProfiler.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

/// Given a parameter expression, walk it and return any references to named
/// parameters.  This fails if an unknown parameter expression exists.
void KGEN::collectParameterReferences(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &results,
    bool &hasConstExpr) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  c.collectUsesFromAttr(attr, results, hasConstExpr);
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.
void KGEN::collectParameterReferences(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &results, bool &hasConstExpr) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  c.collectUsesFromType(type, results, hasConstExpr);
}

/// Return true if the specified type contains parameter references, e.g.
/// `!pop.scalar<dt>` returns true, but `!pop.scalar<f32>` returns false.
///
/// TODO: This isn't an efficient method, it walks the entire type graph without
/// caching.
bool KGEN::isParameterizedType(Type type) {
  SmallVector<ParamDeclRefAttr> paramDecls;
  bool hasConstExpr = false;
  collectParameterReferences(type, paramDecls, hasConstExpr);
  return !paramDecls.empty() || hasConstExpr;
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
  // FIXME: Symbol references are always excepted because the elaborator wants
  // to consider them as constants even when they are parametric.
  if (ParameterAttr::isSimpleConstant(attr) && !isa<SymbolConstantAttr>(attr))
    return attr;

  // If we've already processed this attribute, just reuse the memoized result.
  auto iter = rewritten.find(attr.getAsOpaquePointer());
  if (iter != rewritten.end())
    return Attribute::getFromOpaquePointer(iter->second);

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    result = paramValues[declRef.getName()];
    assert(result && "Verifier should check that all parameters are defined");
  } else if (isa<MLIROpAttr>(attr)) {
    // Expression functions and MLIR operation expressions are isolated from
    // above, so don't collect from them.
  } else {
    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    attr.walkImmediateSubElements(
        [&](Attribute attr) {
          Attribute newAttr = getReboundAttribute(attr);
          changed |= newAttr != attr;
          newAttrs.push_back(newAttr);
        },
        [&](Type type) {
          Type newType = getReboundType(type);
          changed |= newType != type;
          newTypes.push_back(newType);
        });
    if (changed)
      result = attr.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // If an operator persisted, try to simplify it with the symbol table.
  if (auto op = dyn_cast<ParamOperatorAttr>(result))
    if (FailureOr<TypedAttr> expr = evaluateExpression(op); succeeded(expr))
      result = *expr;

  rewritten.try_emplace(attr.getAsOpaquePointer(), result.getAsOpaquePointer());
  return result;
}

/// Get the specified type with any nested parameter expressions rewritten.
Type ParameterEvaluator::getReboundType(Type type) {
  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewritten.find(type.getAsOpaquePointer());
  if (iter != rewritten.end())
    return Type::getFromOpaquePointer(iter->second);

  Type result = type;

  // Rebind types in aggregates that implement SubElementTypeInterface.
  auto signature = dyn_cast<SignatureType>(type);
  // Signature types with input parameters are special because they are
  // "isolated from above" with respect to their contexts, so we don't rebind
  // within them.
  if (!signature || signature.getInputParams().empty()) {
    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    type.walkImmediateSubElements(
        [&](Attribute attr) {
          Attribute newAttr = getReboundAttribute(attr);
          changed |= newAttr != attr;
          newAttrs.push_back(newAttr);
        },
        [&](Type type) {
          Type newType = getReboundType(type);
          changed |= newType != type;
          newTypes.push_back(newType);
        });
    if (changed)
      result = type.replaceImmediateSubElements(newAttrs, newTypes);
  }

  rewritten.try_emplace(type.getAsOpaquePointer(), result.getAsOpaquePointer());
  return result;
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
