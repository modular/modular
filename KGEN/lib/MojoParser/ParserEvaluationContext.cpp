//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParserEvaluationContext.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
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

FailureOr<ResolvedStructHandle>
ParserEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                         bool /*acceptAsync*/) {
  // Parser doesn't support async concretization, so acceptAsync is ignored -
  // we always return the generator.
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  // Typically, this is a LIT struct type.
  if (auto resolvedType =
          sugarDynCast<LIT::StructType>(typeParam.getTypeValue())) {
    ASTDecl &astDecl =
        shared.declResolver->getDeclForTypeSymbol(resolvedType.getSymbol());
    auto structDeclOp = cast<StructDeclOp>(astDecl.getIfOperation());

    if (failed(shared.declResolver->resolveBody(astDecl, astDecl.getLoc())))
      return failure();

    // Return the decl. instance is null since this is not an IREvaluator
    // context.
    return ResolvedStructHandle{
        cast<StructDeclInterface>(structDeclOp.getOperation()),
        resolvedType.getParamValues(), &astDecl,
        /*instance=*/nullptr};
  }

  // Otherwise, this is a raw StructGeneratorOp for a closure.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(typeValue);
  auto genRef = dyn_cast_if_present<TypeGeneratorRefAttr>(typeRef);
  if (!genRef)
    return failure();

  // Look up the ASTDecl for the struct generator using its symbol.
  ASTDecl *structGenDecl =
      shared.declResolver->getDeclForTypeSymbolIfExists(genRef.getSymbol());
  if (!structGenDecl)
    return failure();

  auto structGen =
      dyn_cast_or_null<StructGeneratorOp>(structGenDecl->getIfOperation());
  if (!structGen)
    return failure();

  return ResolvedStructHandle{
      cast<StructDeclInterface>(structGen.getOperation()),
      genRef.getParamValues(), structGenDecl, /*instance=*/nullptr};
}

Operation *ParserEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, StringAttr traitName) {
  auto *astDecl = static_cast<ASTDecl *>(resolved.handle);

  // Typically, this is a StructDeclOp created by the parser.
  if (isa<StructDeclOp>(astDecl->getIfOperation())) {
    auto conformanceDecls = astDecl->lookupInCurrentScope(traitName);
    if (conformanceDecls.empty())
      return nullptr;

    assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
    ASTDecl &conformDecl = *conformanceDecls.front();
    if (failed(shared.declResolver->resolveBody(conformDecl,
                                                conformDecl.getLoc())))
      return nullptr;

    return conformDecl.getIfOperation();
  }

  // This is a raw StructGeneratorOp for a closure.
  auto structGen = cast<StructGeneratorOp>(astDecl->getIfOperation());
  // Find the ConformanceOp within the struct generator directly as they're
  // always created resolved.
  for (auto conformance : structGen.getBody().getOps<ConformanceOp>())
    if (conformance.getSymName() == traitName)
      return conformance;

  return nullptr;
}

void ParserEvaluationContext::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParameterEvaluator evaluator(paramDecls, paramValues);
  evaluator.setEvaluationContext(this);
  callback(evaluator);
}

FailureOr<TypedAttr> ParserEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle TypeConformsToTraitAttr.
  if (auto conformsTo =
          sugarDynCastIfPresent<TypeConformsToTraitAttr>(typedAttr)) {
    // Try LIT-specific trait type folding first, then fall back to the attr
    // folder for struct resolution.
    FailureOr<TypedAttr> result = simplifyConformsToAgainstTypeValue(
        conformsTo, [&](SymbolRefAttr symbol) -> TraitDeclOp {
          ASTDecl &decl = shared.declResolver->getDeclForTypeSymbol(symbol);
          return cast<TraitDeclOp>(decl.getIfOperation());
        });
    if (succeeded(result))
      return result;
    return conformsTo.evaluateWithContext(*this);
  }

  // Handle DowncastAttr.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (TypedAttr folded = LIT::foldDowncastToStructType(downcast))
      return folded;
    // If we are downcasting a more-refined trait to a less-refined trait, use
    // the more refined trait.
    if (TraitType toTrait = sugarDynCast<TraitType>(downcast.getType())) {
      auto fromType = ASTType(downcast.getInputTypeValue());
      bool fromImpliesTo = fromType.checkConformance(toTrait, shared, {}) ==
                           ConformanceResult::Yes;
      if (fromImpliesTo)
        return UpcastAttr::get(downcast.getType(),
                               downcast.getInputTypeValue());
    }
  }

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}
