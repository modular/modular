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

FailureOr<ResolvedStructHandle>
ParameterEvaluationContext::resolveStructOp(TypedAttr /*typeValue*/,
                                            bool /*acceptAsync*/) {
  // Base class cannot resolve anything.
  return failure();
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

void ParameterEvaluationContext::emitEvaluationError(
    const Twine & /*message*/) {
  // Base class does nothing - derived classes can override to emit diagnostics.
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

  if (auto structFieldOffsetByIndex =
          sugarDynCast<StructFieldOffsetByIndexAttr>(typedAttr))
    return evaluateStructFieldOffsetByIndex(structFieldOffsetByIndex);

  if (auto structFieldOffsetByName =
          sugarDynCast<StructFieldOffsetByNameAttr>(typedAttr))
    return evaluateStructFieldOffsetByName(structFieldOffsetByName);

  // Handle VariadicReduceAttr - common across contexts.
  if (auto variadicReduce = sugarDynCast<VariadicReduceAttr>(typedAttr))
    return variadicReduce.evaluateWith(this);

  return failure();
}

FailureOr<TypedAttr>
ParameterEvaluationContext::evaluateGetWitness(GetWitnessAttr getWitness) {
  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(getWitness.getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr))
    return failure();
  ResolvedStructHandle resolved = *resolvedOr;
  Operation *conformanceOp =
      resolveConformanceForStruct(resolved, getWitness.getTraitName());
  if (!conformanceOp) {
    emitEvaluationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' does not have witness table for trait '" +
        getWitness.getTraitName().getValue() + "'");
    return failure();
  }

  auto conformance = cast<ConformanceOp>(conformanceOp);
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  result = getWitness.simplify(conformance, &evaluator);
                });
  if (failed(result)) {
    emitEvaluationError("failed to locate witness entry '" +
                        getWitness.getWitnessName().getValue() +
                        "' for trait '" + getWitness.getTraitName().getValue() +
                        "'");
  }
  return result;
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldTypes(
    StructFieldTypesAttr attr) {
  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_types requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;

  // If concrete instance is available, use its already-substituted field types.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<TypedAttr> resultAttrs;
    for (StructDefFieldAttr field : structType.getFields())
      resultAttrs.push_back(field.getTypeValue());
    return cast<TypedAttr>(VariadicAttr::get(resultAttrs, attr.getType()));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator types and rebind with param values.
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
  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_names requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
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

  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_index_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  auto index = resolved.decl.findFieldIndex(fieldNameAttr.getValue());
  if (!index) {
    emitEvaluationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldNameAttr.getValue() + "'");
    return failure();
  }
  return cast<TypedAttr>(
      Builder(attr.getType().getContext()).getIndexAttr(*index));
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateStructFieldTypeByName(
    StructFieldTypeByNameAttr attr) {
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_type_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();

  // If concrete instance is available, search its fields directly.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    for (StructDefFieldAttr field : structType.getFields())
      if (field.getName().getValue() == fieldName)
        return field.getTypeValue();
    emitEvaluationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field type and rebind.
  TypedAttr fieldType = resolved.decl.getFieldType(fieldName);
  if (!fieldType) {
    emitEvaluationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  result = evaluator.getReboundAttribute(fieldType);
                });
  return result;
}

/// Compute the byte offset of a field within a struct given its field types.
/// Returns failure if the field index is out of bounds or if size/alignment
/// cannot be determined for any field type.
static FailureOr<int64_t>
computeStructFieldOffset(ArrayRef<Type> fieldTypes, int64_t fieldIndex,
                         TargetInfoAttr target,
                         llvm::function_ref<void(const Twine &)> emitError) {
  if (fieldIndex < 0 || fieldIndex >= static_cast<int64_t>(fieldTypes.size())) {
    emitError("field index " + std::to_string(fieldIndex) +
              " is out of bounds for struct with " +
              std::to_string(fieldTypes.size()) + " fields");
    return failure();
  }

  int64_t offset = 0;
  for (int64_t i = 0; i < fieldIndex; ++i) {
    std::optional<int64_t> curFieldAlign =
        DataLayoutInterface::getTypeABIAlign(target, fieldTypes[i]);
    std::optional<int64_t> curFieldSize =
        DataLayoutInterface::getTypeAllocSize(target, fieldTypes[i]);
    if (!curFieldAlign || !curFieldSize) {
      emitError("could not determine size or alignment for field type");
      return failure();
    }
    offset = llvm::alignTo(offset, *curFieldAlign) + *curFieldSize;
  }

  // Align to the target field's alignment.
  std::optional<int64_t> fieldAlign =
      DataLayoutInterface::getTypeABIAlign(target, fieldTypes[fieldIndex]);
  if (!fieldAlign) {
    emitError("could not determine alignment for field type");
    return failure();
  }
  return llvm::alignTo(offset, *fieldAlign);
}

/// Extract field types from a struct, wrapping each in ParamType.
static SmallVector<Type>
getFieldTypesFromStruct(StructInstanceType structType) {
  SmallVector<Type> fieldTypes;
  for (StructDefFieldAttr field : structType.getFields())
    fieldTypes.push_back(ParamType::get(field.getTypeValue()));
  return fieldTypes;
}

/// Extract and rebind field types from a struct declaration using an evaluator.
/// Returns std::nullopt on rebinding failure, or an empty vector for empty
/// structs. If emitError is provided, it will be called with a diagnostic
/// message when rebinding fails.
static std::optional<SmallVector<Type>>
rebindFieldTypes(StructDeclInterface decl, ParameterEvaluator &evaluator,
                 llvm::function_ref<void(const Twine &)> emitError = nullptr) {
  SmallVector<TypedAttr> fieldTypeAttrs;
  decl.getFieldTypes(fieldTypeAttrs);

  SmallVector<Type> fieldTypes;
  for (auto [idx, typeAttr] : llvm::enumerate(fieldTypeAttrs)) {
    FailureOr<TypedAttr> rebound = evaluator.getReboundAttribute(typeAttr);
    if (failed(rebound)) {
      if (emitError)
        emitError("failed to rebind type for field at index " +
                  std::to_string(idx) + " during offset calculation");
      return std::nullopt;
    }
    fieldTypes.push_back(ParamType::get(*rebound));
  }
  return fieldTypes;
}

FailureOr<TypedAttr>
ParameterEvaluationContext::evaluateStructFieldOffsetByIndex(
    StructFieldOffsetByIndexAttr attr) {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet. The evaluation framework will retry later when more
  // information is available. This is the standard pattern for contextually
  // evaluated attrs.
  auto fieldIndexAttr = dyn_cast<IntegerAttr>(attr.getFieldIndex());
  if (!fieldIndexAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(attr.getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_offset_by_index requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  int64_t fieldIndex = fieldIndexAttr.getInt();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = attr.getType().getContext();

  auto emitError = [this](const Twine &msg) { emitEvaluationError(msg); };

  // If concrete instance is available, use its field types directly.
  if (resolved.instance) {
    // When instance is valid, decl must also be valid per ResolvedStructHandle
    // contract. Assert this invariant to catch future refactoring errors.
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);

    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, fieldIndex, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  std::optional<SmallVector<Type>> fieldTypesOpt =
                      rebindFieldTypes(resolved.decl, evaluator, emitError);
                  if (!fieldTypesOpt)
                    return;

                  FailureOr<int64_t> offsetOr = computeStructFieldOffset(
                      *fieldTypesOpt, fieldIndex, target, emitError);
                  if (failed(offsetOr))
                    return;
                  result =
                      cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
                });
  return result;
}

FailureOr<TypedAttr>
ParameterEvaluationContext::evaluateStructFieldOffsetByName(
    StructFieldOffsetByNameAttr attr) {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet. The evaluation framework will retry later when more
  // information is available. This is the standard pattern for contextually
  // evaluated attrs.
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(attr.getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      resolveStructOp(attr.getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    emitEvaluationError("struct_field_offset_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = attr.getType().getContext();

  auto emitError = [this](const Twine &msg) { emitEvaluationError(msg); };

  // Helper to find field index by name and emit error if not found.
  auto findFieldIndexOrError =
      [&](auto fields, StringRef structName) -> std::optional<int64_t> {
    int64_t idx = 0;
    for (auto field : fields) {
      if (field.getName().getValue() == fieldName)
        return idx;
      ++idx;
    }
    emitEvaluationError("struct '" + structName + "' has no field named '" +
                        fieldName + "'");
    return std::nullopt;
  };

  // If concrete instance is available, use its fields directly.
  if (resolved.instance) {
    // When instance is valid, decl must also be valid per ResolvedStructHandle
    // contract. Assert this invariant to catch future refactoring errors.
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    auto fields = structType.getFields();
    StringRef structName =
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue();

    std::optional<int64_t> fieldIndexOpt =
        findFieldIndexOrError(fields, structName);
    if (!fieldIndexOpt)
      return failure();

    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);
    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, *fieldIndexOpt, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Find field index using the decl.
  std::optional<uint64_t> fieldIndexOpt =
      resolved.decl.findFieldIndex(fieldName);
  if (!fieldIndexOpt) {
    emitEvaluationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }
  int64_t fieldIndex = static_cast<int64_t>(*fieldIndexOpt);

  // Use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  std::optional<SmallVector<Type>> fieldTypesOpt =
                      rebindFieldTypes(resolved.decl, evaluator, emitError);
                  if (!fieldTypesOpt)
                    return;

                  FailureOr<int64_t> offsetOr = computeStructFieldOffset(
                      *fieldTypesOpt, fieldIndex, target, emitError);
                  if (failed(offsetOr))
                    return;
                  result =
                      cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
                });
  return result;
}

//===----------------------------------------------------------------------===//
// SymTabEvaluationContext
//===----------------------------------------------------------------------===//

FailureOr<ResolvedStructHandle>
SymTabEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                         bool /*acceptAsync*/) {
  // SymTabEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.
  auto genRef = sugarDynCastIfPresent<TypeGeneratorRefAttr>(
      getTypeRefForTypeValueIfResolved(typeValue));
  if (!genRef)
    return failure();

  auto structDecl =
      symtab.lookupSymbolIn<StructGeneratorOp>(module, genRef.getSymbol());
  if (!structDecl)
    return failure();

  // Return the generator. instance is null since SymTabEvaluationContext
  // doesn't support async concretization.
  return ResolvedStructHandle{
      cast<StructDeclInterface>(structDecl.getOperation()),
      genRef.getParamValues(), nullptr,
      /*instance=*/nullptr};
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
