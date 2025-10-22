//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParserEvaluationContext.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/SharedState.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

static ASTDecl *getDeclForTypeValue(SharedState &shared, TypedAttr typeValue) {
  // We can only simplify if the type reference is resolved already.
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return nullptr;

  auto structType = dyn_cast<LIT::StructType>(typeParam.getTypeValue());
  if (!structType)
    return nullptr;

  return &shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
}

FailureOr<TypedAttr> ParserEvaluationContext::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
  // Handle simplifiable cases here.
  if (auto getWitness = dyn_cast<GetWitnessAttr>(attr))
    return evaluateGetWitness(
        getWitness.getTypeValue(), getWitness.getTraitName(),
        getWitness.getWitnessName(), getWitness.getType());

  if (auto conformsTo = dyn_cast<TypeConformsToTraitAttr>(attr)) {
    if (auto *decl = getDeclForTypeValue(shared, conformsTo.getTypeValue())) {
      auto structDeclOp = cast<StructDeclOp>(decl->getIfOperation());
      return conformsTo.simplify(SymbolTable(structDeclOp));
    }
  }

  if (auto downcast = dyn_cast<DowncastAttr>(attr)) {
    if (getDeclForTypeValue(shared, downcast.getInputTypeValue())) {
      // FIXME: We should raise an error when the resolved struct type does not
      // conforms to the downcast traits. The folding below leads to an indirect
      // error message. However, there is currently no good way to emit an error
      // in evaluation context where the downcast error can be detected, and the
      // current error message is better than an elaboration error (if we do not
      // fold it).
      return downcast.getInputTypeValue();
    }
  }

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
  // We can only simplify if the type reference is resolved already.
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  auto structType = dyn_cast<LIT::StructType>(typeParam.getTypeValue());
  if (!structType)
    return failure();

  // Find the struct decl for the instance.
  ASTDecl &decl =
      shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
  auto structDeclOp = cast<StructDeclOp>(decl.getIfOperation());
  if (failed(shared.declResolver->resolveBody(decl, decl.getLoc()))) {
    return failure();
  }
  auto conformanceDecls = decl.lookupInCurrentScope(traitName);
  // If no conformance exists, still allow it to go through, just don't fold.
  if (conformanceDecls.empty())
    return failure();

  assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
  // Body resolve the conformance op before we extract witness from it.
  ASTDecl &conformDecl = *conformanceDecls.front();
  if (failed(shared.declResolver->resolveBody(conformDecl,
                                              conformDecl.getLoc()))) {
    return failure();
  }
  auto conformanceOp = cast<ConformanceOp>(conformDecl.getIfOperation());
  ParserParameterEvaluator nestedEvaluator(
      shared, structDeclOp.getInputParams(), structType.getParamValues());

  auto getWitness =
      GetWitnessAttr::get(typeParam, traitName, witnessName, type);
  FailureOr<TypedAttr> simplified =
      getWitness.simplify(conformanceOp, &nestedEvaluator);
  if (failed(simplified) || !simplified.value())
    return cast<TypedAttr>(getWitness);

  return simplified.value();
}
