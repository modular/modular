//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParserEvaluationContext.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/SharedState.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Struct Reflection Helpers
//===----------------------------------------------------------------------===//

/// Extract a LIT::StructType from a type-value attribute.
static LIT::StructType getStructTypeForTypeValue(TypedAttr typeValue) {
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return nullptr;
  return sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
}

/// Parser-specific extension of ResolvedLITStructInfo that adds the ASTDecl
/// pointer needed for conformance lookups during parsing.
struct ResolvedParserStructInfo : ResolvedLITStructInfo {
  ASTDecl *decl;

  /// Try to resolve struct info using the parser's SharedState.
  /// Returns std::nullopt if the type cannot be resolved yet.
  static std::optional<ResolvedParserStructInfo>
  tryResolve(SharedState &shared, TypedAttr typeValue) {
    auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
    if (!typeParam)
      return std::nullopt;

    auto resolvedType = sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
    if (!resolvedType)
      return std::nullopt;

    ASTDecl &astDecl =
        shared.declResolver->getDeclForTypeSymbol(resolvedType.getSymbol());
    auto structDeclOp = cast<StructDeclOp>(astDecl.getIfOperation());

    if (failed(shared.declResolver->resolveBody(astDecl, astDecl.getLoc())))
      return std::nullopt;

    return ResolvedParserStructInfo{{structDeclOp, resolvedType}, &astDecl};
  }
};

FailureOr<TypedAttr> ParserEvaluationContext::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
  // Handle simplifiable cases here.
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);
  if (auto getWitness = sugarDynCastIfPresent<GetWitnessAttr>(typedAttr))
    return evaluateGetWitness(
        getWitness.getTypeValue(), getWitness.getTraitName(),
        getWitness.getWitnessName(), getWitness.getType());

  if (auto conformsTo =
          sugarDynCastIfPresent<TypeConformsToTraitAttr>(typedAttr)) {
    if (auto info = ResolvedParserStructInfo::tryResolve(
            shared, conformsTo.getTypeValue()))
      return conformsTo.simplify(SymbolTable(info->structDecl));
    // Try fold tighter trait types.
    ASTType typeToCheck = conformsTo.getTypeValue();
    auto traitToCheck = dyn_cast<TraitType>(typeToCheck.getMetaType());
    return simplifyConformsToAgainstTypeValue(conformsTo, traitToCheck);
  }

  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (auto structTp = getStructTypeForTypeValue(downcast.getInputTypeValue()))
      // FIXME: We should raise an error when the resolved struct type does not
      // conform to the downcast traits. The folding below is unsafe.
      return TypeParamAttr::get(structTp, downcast.getType());
  }

  if (auto variadicReduce =
          sugarDynCastIfPresent<VariadicReduceAttr>(typedAttr))
    return variadicReduce.evaluateWith(this);

  if (auto structFieldTypes =
          sugarDynCastIfPresent<StructFieldTypesAttr>(typedAttr))
    return evaluateStructFieldTypes(structFieldTypes.getTypeValue(),
                                    structFieldTypes.getType());

  if (auto structFieldNames =
          sugarDynCastIfPresent<StructFieldNamesAttr>(typedAttr))
    return evaluateStructFieldNames(structFieldNames.getTypeValue(),
                                    structFieldNames.getType());

  if (auto structFieldIndexByName =
          sugarDynCastIfPresent<StructFieldIndexByNameAttr>(typedAttr))
    return evaluateStructFieldIndexByName(structFieldIndexByName.getTypeValue(),
                                          structFieldIndexByName.getFieldName(),
                                          structFieldIndexByName.getType());

  if (auto structFieldTypeByName =
          sugarDynCastIfPresent<StructFieldTypeByNameAttr>(typedAttr))
    return evaluateStructFieldTypeByName(structFieldTypeByName.getTypeValue(),
                                         structFieldTypeByName.getFieldName(),
                                         structFieldTypeByName.getType());

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

TypedAttr ParserEvaluationContext::getGetWitnessAttr(TypedAttr typeValue,
                                                     StringAttr traitName,
                                                     StringAttr witnessName,
                                                     Type type) {
  // Try to simplify immediately.
  auto simplifiedWitness =
      evaluateGetWitness(typeValue, traitName, witnessName, type);
  if (succeeded(simplifiedWitness))
    return simplifiedWitness.value();

  // Otherwise, use the default builder. No need to re-evaluate the result since
  // the GetWitnessAttr ctor doesn't perform any evaluation itself.
  return GetWitnessAttr::get(typeValue, traitName, witnessName, type);
}

FailureOr<TypedAttr>
ParserEvaluationContext::evaluateGetWitness(TypedAttr typeValue,
                                            StringAttr traitName,
                                            StringAttr witnessName, Type type) {
  auto info = ResolvedParserStructInfo::tryResolve(shared, typeValue);
  if (!info)
    return failure();

  auto conformanceDecls = info->decl->lookupInCurrentScope(traitName);
  // If no conformance exists, still allow it to go through, just don't fold.
  if (conformanceDecls.empty())
    return failure();

  assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
  // Body resolve the conformance op before we extract witness from it.
  ASTDecl &conformDecl = *conformanceDecls.front();
  if (failed(
          shared.declResolver->resolveBody(conformDecl, conformDecl.getLoc())))
    return failure();

  auto conformanceOp = cast<ConformanceOp>(conformDecl.getIfOperation());
  auto evaluator = info->createEvaluator(*this);

  auto getWitness =
      GetWitnessAttr::get(typeValue, traitName, witnessName, type);
  auto simplified = getWitness.simplify(conformanceOp, &evaluator);
  if (failed(simplified) || !simplified.value())
    return cast<TypedAttr>(getWitness);

  return simplified.value();
}

//===----------------------------------------------------------------------===//
// Struct Field Reflection Attrs
//===----------------------------------------------------------------------===//

TypedAttr ParserEvaluationContext::getStructFieldTypesAttr(TypedAttr typeValue,
                                                           VariadicType type) {
  // Try to simplify immediately.
  auto simplified = evaluateStructFieldTypes(typeValue, type);
  if (succeeded(simplified))
    return simplified.value();

  // Otherwise, use the default builder.
  return StructFieldTypesAttr::get(type.getContext(), typeValue, type);
}

FailureOr<TypedAttr>
ParserEvaluationContext::evaluateStructFieldTypes(TypedAttr typeValue,
                                                  VariadicType type) {
  std::optional<ResolvedParserStructInfo> info =
      ResolvedParserStructInfo::tryResolve(shared, typeValue);
  if (!info)
    return failure();

  ParameterEvaluator evaluator = info->createEvaluator(*this);

  // Collect field types, substituting any parameter references.
  SmallVector<TypedAttr> fieldTypes;
  MLIRContext *ctx = type.getContext();
  for (StructFieldOp field : info->structDecl.getFieldDecls()) {
    Type reboundType = evaluator.getReboundType(field.getType());
    fieldTypes.push_back(TypeParamAttr::get(reboundType, TypeType::get(ctx)));
  }

  return cast<TypedAttr>(VariadicAttr::get(fieldTypes, type));
}

TypedAttr ParserEvaluationContext::getStructFieldNamesAttr(TypedAttr typeValue,
                                                           VariadicType type) {
  // Try to simplify immediately.
  auto simplified = evaluateStructFieldNames(typeValue, type);
  if (succeeded(simplified))
    return simplified.value();

  // Otherwise, use the default builder.
  return StructFieldNamesAttr::get(type.getContext(), typeValue, type);
}

FailureOr<TypedAttr>
ParserEvaluationContext::evaluateStructFieldNames(TypedAttr typeValue,
                                                  VariadicType type) {
  std::optional<ResolvedParserStructInfo> info =
      ResolvedParserStructInfo::tryResolve(shared, typeValue);
  if (!info)
    return failure();

  // Collect field names as StringAttrs.
  SmallVector<TypedAttr> fieldNames;
  MLIRContext *ctx = type.getContext();
  for (StructFieldOp field : info->structDecl.getFieldDecls()) {
    fieldNames.push_back(
        StringAttr::get(field.getName(), StringType::get(ctx)));
  }

  return cast<TypedAttr>(VariadicAttr::get(fieldNames, type));
}

TypedAttr ParserEvaluationContext::getStructFieldIndexByNameAttr(
    TypedAttr typeValue, TypedAttr fieldName, IndexType type) {
  // Try to simplify immediately.
  auto simplified = evaluateStructFieldIndexByName(typeValue, fieldName, type);
  if (succeeded(simplified))
    return simplified.value();

  // Otherwise, use the default builder.
  return StructFieldIndexByNameAttr::get(type.getContext(), typeValue,
                                         fieldName, type);
}

FailureOr<TypedAttr> ParserEvaluationContext::evaluateStructFieldIndexByName(
    TypedAttr typeValue, TypedAttr fieldName, IndexType type) {
  auto fieldNameAttr = dyn_cast<StringAttr>(fieldName);
  if (!fieldNameAttr)
    return failure();

  std::optional<ResolvedParserStructInfo> info =
      ResolvedParserStructInfo::tryResolve(shared, typeValue);
  if (!info)
    return failure();

  auto index = info->findFieldIndex(fieldNameAttr.getValue());
  if (!index)
    return failure(); // Field not found; error reported during elaboration.

  return cast<TypedAttr>(Builder(type.getContext()).getIndexAttr(*index));
}

TypedAttr ParserEvaluationContext::getStructFieldTypeByNameAttr(
    TypedAttr typeValue, TypedAttr fieldName, TypeType type) {
  // Try to simplify immediately.
  auto simplified = evaluateStructFieldTypeByName(typeValue, fieldName, type);
  if (succeeded(simplified))
    return simplified.value();

  // Otherwise, use the default builder.
  return StructFieldTypeByNameAttr::get(type.getContext(), typeValue, fieldName,
                                        type);
}

FailureOr<TypedAttr> ParserEvaluationContext::evaluateStructFieldTypeByName(
    TypedAttr typeValue, TypedAttr fieldName, TypeType type) {
  auto fieldNameAttr = dyn_cast<StringAttr>(fieldName);
  if (!fieldNameAttr)
    return failure();

  std::optional<ResolvedParserStructInfo> info =
      ResolvedParserStructInfo::tryResolve(shared, typeValue);
  if (!info)
    return failure();

  auto field = info->findField(fieldNameAttr.getValue());
  if (!field)
    return failure(); // Field not found; error reported during elaboration.

  ParameterEvaluator evaluator = info->createEvaluator(*this);
  Type reboundType = evaluator.getReboundType(field.getType());
  return cast<TypedAttr>(TypeParamAttr::get(reboundType, type));
}
