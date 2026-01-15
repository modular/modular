//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

static Type tryReplaceVariadicSplat(Type type) {
  if (!isa<mlir::LLVM::LLVMStructType, StructType>(type)) {
    return nullptr;
  }

  auto processVariadicSplatType = [&](ArrayRef<Type> types) {
    SmallVector<Type> newTypes;
    bool changed = false;
    for (Type type : types) {
      auto splatType = dyn_cast<VariadicSplatType>(type);
      if (!splatType) {
        newTypes.push_back(type);
        continue;
      }
      std::optional<uint64_t> count = splatType.getResolvedCount();
      if (!count) {
        newTypes.push_back(type);
        continue;
      }
      changed = true;
      for (unsigned i = 0, e = *count; i < e; ++i)
        newTypes.push_back(splatType.getElementType());
    }
    return changed ? newTypes : SmallVector<Type>();
  };

  MLIRContext *context = type.getContext();

  // Handle `!kgen.struct`.
  if (auto structType = dyn_cast<StructType>(type)) {
    SmallVector<Type> newTypes =
        processVariadicSplatType(structType.getElementTypes());
    if (newTypes.empty())
      return type;
    return StructType::get(context, newTypes);
  }

  // Handle `!llvm.struct`.
  if (auto llvmStructType = dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    SmallVector<Type> newTypes =
        processVariadicSplatType(llvmStructType.getBody());
    if (newTypes.empty())
      return type;
    return mlir::LLVM::LLVMStructType::getLiteral(context, newTypes);
  }
  llvm_unreachable("unhandled type");
}

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
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

ResolvedStructHandle
ParameterEvaluationContext::resolveStructOp(TypedAttr /*typeValue*/) {
  return {};
}

Operation *ParameterEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle /*resolved*/, StringAttr /*traitName*/) {
  return nullptr;
}

void ParameterEvaluationContext::withEvaluator(
    ArrayRef<ParamDeclAttr> /*paramDecls*/, ArrayRef<TypedAttr> /*paramValues*/,
    llvm::function_ref<void(ParameterEvaluator &)> /*callback*/) {}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface /*attr*/) {
  // Default implementation - no context-specific handling.
  return failure();
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
  // Let derived classes handle first. If they return success (including null
  // for retry), use that. Only fall through to base class if they return
  // failure (meaning they don't handle it).
  FailureOr<TypedAttr> contextResult = evaluateContextSpecific(attr);
  if (succeeded(contextResult))
    return contextResult;

  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle GetWitnessAttr using struct resolution.
  if (auto getWitness = sugarDynCast<GetWitnessAttr>(typedAttr))
    return evaluateGetWitness(getWitness);

  // Handle struct field reflection attributes.
  if (auto structFieldTypes = sugarDynCast<StructFieldTypesAttr>(typedAttr))
    return evaluateStructFieldTypes(structFieldTypes);

  if (auto structFieldNames = sugarDynCast<StructFieldNamesAttr>(typedAttr))
    return evaluateStructFieldNames(structFieldNames);

  if (auto structFieldIndexByName =
          sugarDynCast<StructFieldIndexByNameAttr>(typedAttr))
    return evaluateStructFieldIndexByName(structFieldIndexByName);

  if (auto structFieldTypeByName =
          sugarDynCast<StructFieldTypeByNameAttr>(typedAttr))
    return evaluateStructFieldTypeByName(structFieldTypeByName);

  // Handle VariadicReduceAttr - common across contexts.
  if (auto variadicReduce = sugarDynCast<VariadicReduceAttr>(typedAttr))
    return variadicReduce.evaluateWith(this);

  return failure();
}

FailureOr<TypedAttr>
ParameterEvaluationContext::evaluateGetWitness(GetWitnessAttr getWitness) {
  ResolvedStructHandle resolved = resolveStructOp(getWitness.getTypeValue());
  if (!resolved)
    return failure();
  Operation *conformanceOp =
      resolveConformanceForStruct(resolved, getWitness.getTraitName());
  if (!conformanceOp)
    return failure();

  auto conformance = cast<ConformanceOp>(conformanceOp);
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  FailureOr<TypedAttr> simplified =
                      getWitness.simplify(conformance, &evaluator);
                  if (succeeded(simplified) && simplified.value())
                    result = simplified.value();
                });
  return result;
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldTypes(
    StructFieldTypesAttr attr) {
  ResolvedStructHandle resolved = resolveStructOp(attr.getTypeValue());
  if (!resolved)
    return failure();
  SmallVector<TypedAttr> fieldTypes;
  resolved.decl.getFieldTypes(fieldTypes);

  FailureOr<TypedAttr> result = failure();
  withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        SmallVector<TypedAttr> resultAttrs;
        for (TypedAttr fieldType : fieldTypes)
          resultAttrs.push_back(evaluator.getReboundAttribute(fieldType));

        result =
            cast<TypedAttr>(VariadicAttr::get(resultAttrs, attr.getType()));
      });
  return result;
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldNames(
    StructFieldNamesAttr attr) {
  ResolvedStructHandle resolved = resolveStructOp(attr.getTypeValue());
  if (!resolved)
    return failure();
  SmallVector<StringAttr> fieldNames;
  resolved.decl.getFieldNames(fieldNames);

  SmallVector<TypedAttr> resultAttrs;
  MLIRContext *ctx = attr.getContext();
  for (StringAttr name : fieldNames)
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));

  return cast<TypedAttr>(VariadicAttr::get(resultAttrs, attr.getType()));
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldIndexByName(
    StructFieldIndexByNameAttr attr) {
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr)
    return failure();

  ResolvedStructHandle resolved = resolveStructOp(attr.getTypeValue());
  if (!resolved)
    return failure();
  auto index = resolved.decl.findFieldIndex(fieldNameAttr.getValue());
  if (!index)
    return failure();
  return cast<TypedAttr>(
      Builder(attr.getType().getContext()).getIndexAttr(*index));
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldTypeByName(
    StructFieldTypeByNameAttr attr) {
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr)
    return failure();

  ResolvedStructHandle resolved = resolveStructOp(attr.getTypeValue());
  if (!resolved)
    return failure();
  TypedAttr fieldType = resolved.decl.getFieldType(fieldNameAttr.getValue());
  if (!fieldType)
    return failure();
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  result = evaluator.getReboundAttribute(fieldType);
                });
  return result;
}

//===----------------------------------------------------------------------===//
// SymTabEvaluationContext
//===----------------------------------------------------------------------===//

ResolvedStructHandle
SymTabEvaluationContext::resolveStructOp(TypedAttr typeValue) {
  auto genRef = sugarDynCastIfPresent<TypeGeneratorRefAttr>(
      getTypeRefForTypeValueIfResolved(typeValue));
  if (!genRef)
    return {};

  auto structDecl =
      symtab.lookupSymbolIn<StructGeneratorOp>(module, genRef.getSymbol());
  if (!structDecl)
    return {};

  return {cast<StructDeclInterface>(structDecl.getOperation()),
          genRef.getParamValues(), nullptr};
}

void SymTabEvaluationContext::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParameterEvaluator evaluator(paramDecls, paramValues);
  evaluator.setEvaluationContext(this);
  callback(evaluator);
}

Operation *SymTabEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, StringAttr traitName) {
  return symtab.lookupSymbolIn<ConformanceOp>(resolved.decl, traitName);
}

Operation *SymTabEvaluationContext::getStructInstIfResolved(TypedAttr typeVal) {
  auto genRef = dyn_cast_if_present<TypeGeneratorRefAttr>(typeVal);
  if (!genRef)
    return nullptr;

  // Find the struct decl for the instance.
  return symtab.lookupSymbolIn<StructGeneratorOp>(module, genRef.getSymbol());
}

FailureOr<TypedAttr> SymTabEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle TypeConformsToTraitAttr using struct resolution.
  if (auto conformsTo = sugarDynCast<TypeConformsToTraitAttr>(typedAttr)) {
    if (auto decl = getStructInstIfResolved(conformsTo.getTypeRefIfResolved()))
      return conformsTo.simplify(SymbolTable(decl));
  }

  // Handle inlined apply operations.
  if (auto pocAttr = sugarDynCast<ParamOperatorAttr>(typedAttr))
    return inlineApply(pocAttr);

  return failure();
}

FailureOr<TypedAttr>
SymTabEvaluationContext::inlineApply(ParamOperatorAttr apply) {
  // if there is any generator that is marked to have an inlined form, we inline
  // it to reach the canonicalized form.
  if (apply.getOpcode() != POC::Apply &&
      apply.getOpcode() != POC::ApplyResultSlot)
    return failure();

  auto cst = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
  if (!cst)
    return failure();

  Operation *op =
      symtab.lookupSymbolIn(module, cst.getSymbol().getLeafReference());

  // TODO(MOCO-2656): At the moment, only generator will be annotated by
  //`ApplyInliner`, this can be generalized to handle indirect calls to
  //"always_inline("builtin")" (e.g., via trait method) function.
  if (auto func = dyn_cast_or_null<GeneratorOp>(op);
      func && func.getInlinedFormAttr()) {
    TypedAttr inlinedExpr = func.getInlinedFormAttr();
    // Drop the symbol to get the parameter binding;
    ArrayRef<TypedAttr> paramBinding = apply.getOperands().drop_front();

    // Rebind first
    ParameterEvaluator evaluator(func.getInputParams(), cst.getParamValues());
    evaluator.setEvaluationContext(this);
    inlinedExpr = cast<TypedAttr>(evaluator.getReboundAttribute(inlinedExpr));
    if (!paramBinding.empty()) {
      // If generator takes input, we need to further evaluate the inlined
      // expression with the parameter binding provided by the apply.
      inlinedExpr = cast<GeneratorAttr>(inlinedExpr)
                        .getSpecializedGenerator(paramBinding, this)
                        .getInstantiatedValue();
    }

    return inlinedExpr;
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator core implementation.
//===----------------------------------------------------------------------===//

ParameterEvaluator::ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                                       ArrayRef<TypedAttr> declBindings) {
  for (auto [decl, value] : llvm::zip(paramDecls, declBindings))
    setDeclBinding(decl, value);
}

ParameterEvaluator::ParameterEvaluator(ArrayRef<TypedAttr> declBindings) {
  for (TypedAttr param : declBindings)
    appendIndexBinding(param);
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
  if (isa<ParameterScopeAttrInterface>(attr))
    ++rootDepth;

  // If a parameter got rebound to an index reference, we need to increase its
  // depth based on the current signature, per STCHDDDOS.
  // FIXME: Is there a better way around this? This previously manifested as
  // unintentional name shadowing problems, but walking here is inefficient.
  auto upbindValue = [&](TypedAttr value) -> TypedAttr {
    if (rootDepth + inputDepth == 0)
      return value;
    IndexDepthAdjuster adjuster(/*adjustDepth=*/rootDepth + inputDepth);
    return adjuster.replace(value);
  };

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    // If the referenced parameter is not bound, forward the reference.
    auto declRefType = doReplace(declRef.getType(), rootDepth);
    if (auto it = declBindings.find(declRef.getName());
        it != declBindings.end()) {
      auto resultV = upbindValue(it->second);
      // If we are mapping between a sugared and non-sugared version of the
      // parameter, make sure to keep a consistent type.  This enables us to
      // substitute values into parameter expressions that have sugared and
      // canonical forms.
      if (resultV.getType() != declRefType &&
          isEqualCanon(resultV.getType(), declRefType))
        resultV = ParamOperatorAttr::getRebind(resultV, declRefType);
      result = resultV;
    } else {
      result = ParamDeclRefAttr::get(declRef.getName(), declRefType);
    }
  } else if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr);
             indexRef && indexRef.getDepth() == rootDepth) {
    assert(indexRef.getIndex() < indexBindings.size() &&
           "parameter index out of range");
    auto indexRefType = doReplace(indexRef.getType(), rootDepth);
    TypedAttr resultV = indexBindings[indexRef.getIndex()];
    if (resultV) {
      resultV = cast<TypedAttr>(upbindValue(resultV));
      // If we are mapping between a sugared and non-sugared version of the
      // parameter, make sure to keep a consistent type.  This enables us to
      // substitute values into parameter expressions that have sugared and
      // canonical forms.
      if (resultV.getType() != indexRefType &&
          isEqualCanon(resultV.getType(), indexRefType))
        resultV = ParamOperatorAttr::getRebind(resultV, indexRefType);
    } else if (indexRefType == indexRef.getType()) {
      resultV = indexRef; // Reuse the IndexRef if the type matches.
    } else {              // Otherwise rebuild it.
      resultV = ParamIndexRefAttr::get(indexRef.getDepth(), indexRef.getIndex(),
                                       indexRefType);
    }
    result = resultV;
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
  } else if (auto bindParams = dyn_cast<BindParamsAttr>(attr)) {
    bool changed = false;
    // BindParamsAttr must always be re-created using an Evaluation Context.
    SmallVector<TypedAttr> newParamValues;
    for (auto param : bindParams.getParamValues()) {
      auto newParam = replaceImpl(param, rootDepth);
      if (!newParam)
        return nullptr;
      changed |= newParam != param;
      newParamValues.push_back(cast<TypedAttr>(newParam));
    }

    Attribute newGenerator = replaceImpl(bindParams.getGenerator(), rootDepth);
    if (!newGenerator)
      return nullptr;
    changed |= newGenerator != bindParams.getGenerator();

    Type newType = replaceImpl(bindParams.getType(), rootDepth);
    if (!newType)
      return nullptr;
    changed |= newType != bindParams.getType();

    if (changed)
      return BindParamsAttr::get(bindParams.getContext(),
                                 cast<TypedAttr>(newGenerator), newParamValues,
                                 newType, getEvaluationContext());
    return bindParams;
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

  // If an evaluatable parameter persisted, try to simplify it with additional
  // context.
  if (evaluationContext)
    if (auto attr = dyn_cast<ContextuallyEvaluatedAttrInterface>(result))
      if (FailureOr<TypedAttr> expr =
              evaluationContext->evaluateExpression(attr);
          succeeded(expr))
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
  if (auto newType = tryReplaceVariadicSplat(result))
    result = newType;
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
  for (auto [name, value] : declBindings)
    os << "  " << name << " = " << value << "\n";
  for (auto [idx, value] : llvm::enumerate(indexBindings))
    os << "  *(0," << idx << ") = " << value << "\n";
}

//===----------------------------------------------------------------------===//
// Helper methods involving parameter evaluation.
//===----------------------------------------------------------------------===//

std::optional<PartiallySpecializedInputParams>
PartiallySpecializedInputParams::from(
    ArrayRef<Type> paramTypes, ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn) {
  // Verify the number of input parameters.
  if (paramBindings.size() != paramTypes.size()) {
    assert(emitErrorFn && "unexpected invalid bindings");
    emitErrorFn() << "generator type expects " << paramTypes.size()
                  << " parameters but got bindings for "
                  << paramBindings.size();
    return std::nullopt;
  }

  PartiallySpecializedInputParams result;
  ParameterEvaluator &evaluator = result.evaluator;
  SmallVector<Type, 16> &unboundParamTypes = result.unboundParamTypes;
  llvm::BitVector &boundParams = result.boundParams;
  boundParams.resize(paramTypes.size());

  evaluator.setEvaluationContext(evaluationContext);
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  for (auto [paramNo, valueX, type] :
       llvm::enumerate(paramBindings, paramTypes)) {
    auto value = valueX;
    // Bound parameters are allowed to refine the type of subsequent
    // parameters, e.g. in `<ty: type, fn: () -> !kgen.param<ty>>`, the
    // expected type of the second parameter will be refined when the first
    // parameter is bound.
    auto remappedDeclType = remapType(type);

    // Even if we're skipping a binding site, we still need to remap the decl.
    // TODO: Disallow UnboundAttr for skipping bindings.
    if (::isa<UnboundAttr>(value)) {
      // Set the binding to a declref of the thing itself - that will keep it
      // from becoming #kgen.unbound.  This #param.index.ref will have a level
      // of -1, and we adjust the level of its type by -1 so it balances out
      // correctly when referenced.
      auto adjustedParamType = minusOneAdjuster.replace(remappedDeclType);
      auto value = ParamIndexRefAttr::get(
          /*depth=*/-1, unboundParamTypes.size(), adjustedParamType);
      unboundParamTypes.push_back(remappedDeclType);
      evaluator.appendIndexBinding(value);
    } else {
      // We must remap the value type being provided as well, because it may
      // be referring to outer-context indexed parameters, whose depth will be
      // increased when substituted into this signature, per STCHDDDOS.
      auto valueType = value.getType();
      remappedDeclType = minusOneAdjuster.replace(remappedDeclType);
      if (valueType != remappedDeclType &&
          !isEqualCanon(valueType, remappedDeclType)) {
        if (!emitErrorFn)
          return {};
        emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                      << valueType << " but callee expected type "
                      << remappedDeclType;
        return {};
      }

      // Realign sugar if necessary.
      if (valueType != remappedDeclType)
        value = ParamOperatorAttr::getRebind(value, remappedDeclType);

      evaluator.appendIndexBinding(value);
      boundParams.set(paramNo);
    }
  }

  return result;
}

/// Instantiate a new parameter evaluator with the given parameter values.
ParametricParameterEvaluator::ParametricParameterEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> declBindings)
    : ParameterEvaluator(paramDecls, declBindings) {}
/// Instantiate a new parameter evaluator with the given input parameters.
ParametricParameterEvaluator::ParametricParameterEvaluator(
    ArrayRef<TypedAttr> declBindings)
    : ParameterEvaluator(declBindings) {}

/// Instantiate a new parameter evaluator with the given parameter values.
ParametricParameterEvaluator::ParametricParameterEvaluator(
    DenseMap<StringAttr, TypedAttr> declBindings,
    ArrayRef<TypedAttr> indexBindings, size_t inputDepth)
    : ParameterEvaluator(declBindings, indexBindings, inputDepth) {}
