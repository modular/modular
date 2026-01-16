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

FailureOr<ResolvedStructHandle>
ParserEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                         bool /*acceptAsync*/) {
  // Parser doesn't support async concretization, so acceptAsync is ignored -
  // we always return the generator.
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  auto resolvedType = sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
  if (!resolvedType)
    return failure();

  ASTDecl &astDecl =
      shared.declResolver->getDeclForTypeSymbol(resolvedType.getSymbol());
  auto structDeclOp = cast<StructDeclOp>(astDecl.getIfOperation());

  if (failed(shared.declResolver->resolveBody(astDecl, astDecl.getLoc())))
    return failure();

  // Return the decl. instance is null since this is not an IREvaluator context.
  return ResolvedStructHandle{
      cast<StructDeclInterface>(structDeclOp.getOperation()),
      resolvedType.getParamValues(), &astDecl,
      /*instance=*/nullptr};
}

Operation *ParserEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, StringAttr traitName) {
  auto *astDecl = static_cast<ASTDecl *>(resolved.handle);

  auto conformanceDecls = astDecl->lookupInCurrentScope(traitName);
  if (conformanceDecls.empty())
    return nullptr;

  assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
  ASTDecl &conformDecl = *conformanceDecls.front();
  if (failed(
          shared.declResolver->resolveBody(conformDecl, conformDecl.getLoc())))
    return nullptr;

  return conformDecl.getIfOperation();
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
    FailureOr<ResolvedStructHandle> resolvedOr =
        resolveStructOp(conformsTo.getTypeValue(), /*acceptAsync=*/false);
    if (succeeded(resolvedOr))
      return conformsTo.simplify(SymbolTable(resolvedOr->decl.getOperation()));

    // Try fold tighter trait types.
    ASTType typeToCheck = conformsTo.getTypeValue();
    auto traitToCheck = dyn_cast<TraitType>(typeToCheck.getMetaType());
    return simplifyConformsToAgainstTypeValue(conformsTo, traitToCheck);
  }

  // Handle DowncastAttr.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (auto structTp = getStructTypeForTypeValue(downcast.getInputTypeValue()))
      // FIXME: We should raise an error when the resolved struct type does not
      // conform to the downcast traits. The folding below is unsafe.
      return TypeParamAttr::get(structTp, downcast.getType());
  }

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}
