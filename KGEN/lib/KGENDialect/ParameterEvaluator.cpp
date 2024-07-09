//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/Support/CompilerProfiling.h"
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
void KGEN::collectParameterReferences(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &results,
    bool &hasConstExpr) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  VerboseCompilerTimeTraceScope traceScope("collectParameters");
  c.collectUsesFromAttr(attr, results, hasConstExpr);
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.
void KGEN::collectParameterReferences(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &results, bool &hasConstExpr) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  VerboseCompilerTimeTraceScope traceScope("collectParameters");
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

ParameterEvaluator::ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                                       ArrayRef<TypedAttr> paramValues) {
  for (auto [decl, value] : llvm::zip(paramDecls, paramValues))
    setParameterValue(decl, value);
}

ParameterEvaluator::ParameterEvaluator(ArrayRef<TypedAttr> paramValues) {
  for (TypedAttr param : paramValues)
    addInputValue(param);
}

// NOTE: This is out of line to provide a home for the ParameterEvaluator
// vtable.
FailureOr<TypedAttr>
ParameterEvaluator::evaluateExpression(ParamOperatorAttr op) {
  return failure();
}

std::pair<IntegerAttr, bool>
ParameterEvaluator::narrowCondOp(Attribute attr, size_t rootDepth) {
  if (auto op = dyn_cast<ParamOperatorAttr>(attr);
      op && op.getOpcode() == POC::Cond) {
    Attribute cond = replaceImpl(op.getOperands().front(), rootDepth);
    if (!cond)
      return {nullptr, true};
    return {dyn_cast<IntegerAttr>(cond), false};
  }
  return {nullptr, false};
}

Attribute ParameterEvaluator::doReplace(Attribute attr, size_t rootDepth) {
  // If a parameter got rebound to an index reference, we need to increase its
  // depth based on the current signature.
  // FIXME: Is there a better way around this? This previously manifested as
  // unintentional name shadowing problems, but walking here is inefficient.
  auto upbindValue = [&](Attribute value) {
    if (rootDepth + inputDepth == 0)
      return value;
    IndexDepthAdjuster adjuster(/*adjustDepth=*/rootDepth + inputDepth);
    return adjuster.replace(value);
  };

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    // If the referenced parameter is not bound, forward the reference.
    if (auto it = paramValues.find(declRef.getName()); it != paramValues.end())
      result = upbindValue(it->second);
    else
      result = declRef;
  } else if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr);
             indexRef && indexRef.getDepth() == rootDepth) {
    auto values = indexRef.getIsResult() ? ArrayRef(resultParamValues)
                                         : ArrayRef(inputParamValues);
    assert(indexRef.getIndex() < values.size() &&
           "parameter index out of range");
    result = upbindValue(values[indexRef.getIndex()]);
  } else if (isa<MLIROpAttr>(attr)) {
    // Expression functions and MLIR operation expressions are isolated from
    // above, so don't collect from them.
  } else if (auto [condVal, skip] = narrowCondOp(attr, rootDepth);
             condVal || skip) {
    if (skip)
      return nullptr;
    // If condition is a constant rebind only one of the clauses.
    auto op = cast<ParamOperatorAttr>(attr);
    if (condVal.getValue().isZero())
      result = replaceImpl(op.getOperands()[2], rootDepth);
    else
      result = replaceImpl(op.getOperands()[1], rootDepth);
    if (!result)
      return nullptr;
  } else {
    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    // Stop walking and propagate failures when they occur.
    bool changed = false;
    bool failed = false;
    attr.walkImmediateSubElements(
        [&](Attribute attr) {
          if (failed)
            return;
          Attribute newAttr = replaceImpl(attr, rootDepth);
          if (!newAttr)
            failed = true;
          changed |= newAttr != attr;
          newAttrs.push_back(newAttr);
        },
        [&](Type type) {
          if (failed)
            return;
          Type newType = replaceImpl(type, rootDepth);
          if (!newType)
            failed = true;
          changed |= newType != type;
          newTypes.push_back(newType);
        });
    if (failed)
      return nullptr;
    if (changed)
      result = attr.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // If an operator persisted, try to simplify it with the symbol table.
  if (auto op = dyn_cast<ParamOperatorAttr>(result))
    if (FailureOr<TypedAttr> expr = evaluateExpression(op); succeeded(expr))
      result = *expr;

  return result;
}

Type ParameterEvaluator::doReplace(Type type, size_t rootDepth) {
  Type result = type;

  if (isa<ParameterScopeTypeInterface>(type))
    ++rootDepth;

  // Rebind types in aggregates that implement SubElementTypeInterface.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  bool changed = false;
  // Stop walking and propagate failures when they occur.
  bool failed = false;
  type.walkImmediateSubElements(
      [&](Attribute attr) {
        if (failed)
          return;
        Attribute newAttr = replaceImpl(attr, rootDepth);
        if (!newAttr)
          failed = true;
        changed |= newAttr != attr;
        newAttrs.push_back(newAttr);
      },
      [&](Type type) {
        if (failed)
          return;
        Type newType = replaceImpl(type, rootDepth);
        if (!newType)
          failed = true;
        changed |= newType != type;
        newTypes.push_back(newType);
      });
  if (failed)
    return nullptr;
  if (changed)
    result = type.replaceImmediateSubElements(newAttrs, newTypes);
  return result;
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator debugging support.
//===----------------------------------------------------------------------===/r

// Note: this dumps out in non-stable hash table order, only use for debugging
// purposes!
void ParameterEvaluator::dump() const {
  auto &os = llvm::errs();
  os << "ParameterEvaluator: \n";
  for (auto [name, value] : paramValues)
    os << "  " << name << " = " << value << "\n";
  for (auto [idx, value] : llvm::enumerate(inputParamValues))
    os << "  *(0," << idx << ") = " << value << "\n";
  for (auto [idx, value] : llvm::enumerate(resultParamValues))
    os << "  *(0," << idx << ")* = " << value << "\n";
}
