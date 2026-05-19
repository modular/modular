//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains evaluation/folding implementations for KGEN attributes.
// These methods implement
// ContextuallyEvaluatedAttrInterface::evaluateWithContext.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParamListReduceAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListReduceAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto paramList = sugarDynCast<ParamListAttr>(getParamList());
  auto reducer = sugarDynCast<GeneratorAttr>(getGenerator());

  if (!paramList || !reducer)
    return failure();

  // We have a concrete value for both the generator/variadic, then fold
  unsigned eltCnt = paramList.getValues().size();
  TypedAttr reducedVal = sugarCast<TypedAttr>(getBase());
  for (unsigned i = 0; i < eltCnt; ++i) {
    IntegerAttr vaIdx =
        IntegerAttr::get(IndexType::get(paramList.getContext()), i);
    GeneratorAttr spGen = reducer.getSpecializedGenerator(
        {reducedVal, paramList, vaIdx}, &context);
    if (!spGen)
      return TypedAttr();
    // This should never happen, we should have verified VariadicMapAttr.
    assert(spGen.isFullyBound() && "invalid form of variadic map");
    reducedVal = spGen.getInstantiatedValue();
  }

  return {reducedVal};
}

//===----------------------------------------------------------------------===//
// ParamListTabulateAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListTabulateAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto cntAttr = sugarDynCast<IntegerAttr>(getCount());
  auto genAttr = sugarDynCast<GeneratorAttr>(getGenerator());
  if (!cntAttr || !genAttr)
    return failure();

  int64_t n = cntAttr.getInt();
  if (n < 0)
    return failure();

  SmallVector<TypedAttr> values;
  values.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    IntegerAttr idxAttr = IntegerAttr::get(IndexType::get(getContext()), i);
    GeneratorAttr spGen = genAttr.getSpecializedGenerator({idxAttr}, &context);
    if (!spGen)
      return TypedAttr();
    if (!spGen.isFullyBound())
      return failure();
    values.push_back(sugarCast<TypedAttr>(spGen.getInstantiatedValue()));
  }
  return {ParamListAttr::get(values, getType())};
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
GetWitnessAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr))
    return failure();
  ResolvedStructHandle resolved = *resolvedOr;
  Operation *conformanceOp =
      context.resolveConformanceForStruct(resolved, getTraitName());
  if (!conformanceOp) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' does not have witness table for trait '" +
        getTraitName().getValue() + "'");
    return failure();
  }

  auto conformance = cast<ConformanceOp>(conformanceOp);
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = simplify(conformance, &evaluator);
                        });
  if (failed(result)) {
    context.emitMaterializationError(
        "failed to locate witness entry '" + getWitnessName().getValue() +
        "' for trait '" + getTraitName().getValue() + "'");
  }
  return result;
}

//===----------------------------------------------------------------------===//
// TypeConformsToTraitAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> TypeConformsToTraitAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    // In materialization contexts, failure to resolve means the type is not a
    // struct (e.g. MLIR primitive types like `index`). Non-struct types don't
    // conform to any traits, so return false.
    if (context.isMaterializationContext())
      return {getScalarBoolConstant(getContext(), false)};
    return failure();
  }

  ResolvedStructHandle resolved = *resolvedOr;
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        result = simplify(SymbolTable(resolved.decl.getOperation()), evaluator);
      });
  return result;
}

//===----------------------------------------------------------------------===//
// Struct Field Attr evaluateWithContext implementations
//===----------------------------------------------------------------------===//

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
  // MetaType does not really matter here, they will be striped later by
  // `ParamType::get(rebound)` anyway.
  decl.getFieldTypes(fieldTypeAttrs, TypeType::get(decl.getContext()));

  SmallVector<Type> fieldTypes;
  for (auto [idx, typeAttr] : llvm::enumerate(fieldTypeAttrs)) {
    TypedAttr rebound = evaluator.getReboundAttribute(typeAttr);
    if (!rebound) {
      if (emitError)
        emitError("failed to rebind type for field at index " +
                  std::to_string(idx) + " during offset calculation");
      return std::nullopt;
    }
    fieldTypes.push_back(ParamType::get(rebound));
  }
  return fieldTypes;
}

FailureOr<TypedAttr> StructFieldTypesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_types requires a struct type");
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
    return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator types and rebind with param values.
  SmallVector<TypedAttr> fieldTypes;
  resolved.decl.getFieldTypes(fieldTypes, getType().getElementType());

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        SmallVector<TypedAttr> resultAttrs;
        for (TypedAttr fieldType : fieldTypes)
          resultAttrs.push_back(evaluator.getReboundAttribute(fieldType));
        result = cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_names requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  SmallVector<StringAttr> fieldNames;
  resolved.decl.getFieldNames(fieldNames);

  SmallVector<TypedAttr> resultAttrs;
  MLIRContext *ctx = getContext();
  for (StringAttr name : fieldNames)
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));

  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

//===----------------------------------------------------------------------===//
// Function Reflection Attrs
//===----------------------------------------------------------------------===//

namespace {
/// Resolve a function-valued `TypedAttr` to its defining op via the
/// evaluation context. Returns null if the value is not a direct function
/// reference (`#kgen.symbol.constant<@...>`).
///
/// The returned `FuncInterface` op is `lit.fn` when resolved through the
/// parser or LIT symbol-table contexts, or `kgen.generator` when resolved
/// through the KGEN symbol-table or IR evaluator contexts. Reflection
/// attrs are evaluated during parsing or during elaboration, so the
/// post-elaboration `kgen.func` form is never the resolution target. Both
/// reachable ops also implement `DeclInterface`, so callers needing the
/// param list cast accordingly.
FuncInterface resolveFuncDecl(TypedAttr funcValue,
                              ParameterEvaluationContext &context) {
  // Mojo function values reach reflection as `#kgen.symbol.constant<@func>`.
  // Closure literals are not yet supported.
  auto symbol = dyn_cast<SymbolConstantAttr>(funcValue);
  if (!symbol)
    return nullptr;
  return context.resolveFunctionDecl(symbol.getSymbol());
}
} // namespace

FailureOr<TypedAttr> GetFunctionParameterCountAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_count requires a concrete function value");
    return failure();
  }
  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available so reflection counts remain stable across transforms that
  // rewrite the live `inputParams` (e.g. `RemoveUnusedParams`). Falls back to
  // the live input params for `lit.fn` (pre-LowerLIT reflection) and for
  // generators that don't carry a snapshot.
  size_t count;
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      count = snapshot.size();
    } else {
      count = gen.getInputParams().size();
    }
  } else {
    count = cast<DeclInterface>(func.getOperation()).getInputParams().size();
  }
  return cast<TypedAttr>(IntegerAttr::get(IndexType::get(getContext()), count));
}

FailureOr<TypedAttr> GetFunctionParameterNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_names requires a concrete function value");
    return failure();
  }
  MLIRContext *ctx = getContext();
  SmallVector<TypedAttr> resultAttrs;

  auto appendName = [&](StringAttr name) {
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));
  };

  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available; see the comment in `GetFunctionParameterCountAttr`.
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      resultAttrs.reserve(snapshot.size());
      for (PogMetadataAttr pog : snapshot.getPogs())
        appendName(pog.getName());
      return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
    }
  }

  ArrayRef<ParamDeclAttr> params =
      cast<DeclInterface>(func.getOperation()).getInputParams();
  resultAttrs.reserve(params.size());
  for (ParamDeclAttr param : params)
    appendName(param.getName());
  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

FailureOr<TypedAttr> GetFunctionIsRaisingAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_is_raising requires a concrete function value");
    return failure();
  }
  return cast<TypedAttr>(BoolAttr::get(getContext(), func.isThrows()));
}

FailureOr<TypedAttr> StructFieldIndexByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_index_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  auto index = resolved.decl.findFieldIndex(fieldNameAttr.getValue());
  if (!index) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldNameAttr.getValue() + "'");
    return failure();
  }
  return cast<TypedAttr>(Builder(getType().getContext()).getIndexAttr(*index));
}

FailureOr<TypedAttr> StructFieldTypeByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_type_by_name requires a struct type");
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
    context.emitMaterializationError(
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
  TypedAttr fieldType = resolved.decl.getFieldType(fieldName, getType());
  if (!fieldType) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = evaluator.getReboundAttribute(fieldType);
                        });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByIndexAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet. The evaluation framework will retry later when more
  // information is available.
  auto fieldIndexAttr = dyn_cast<IntegerAttr>(getFieldIndex());
  if (!fieldIndexAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_index requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  int64_t fieldIndex = fieldIndexAttr.getInt();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // If concrete instance is available, use its field types directly.
  if (resolved.instance) {
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
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator, emitError);
        if (!fieldTypesOpt)
          return;

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet.
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // Helper to find field index by name and emit error if not found.
  auto findFieldIndexOrError =
      [&](auto fields, StringRef structName) -> std::optional<int64_t> {
    int64_t idx = 0;
    for (auto field : fields) {
      if (field.getName().getValue() == fieldName)
        return idx;
      ++idx;
    }
    context.emitMaterializationError(
        "struct '" + structName + "' has no field named '" + fieldName + "'");
    return std::nullopt;
  };

  // If concrete instance is available, use its fields directly.
  if (resolved.instance) {
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
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }
  int64_t fieldIndex = static_cast<int64_t>(*fieldIndexOpt);

  // Use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator, emitError);
        if (!fieldTypesOpt)
          return;

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}
