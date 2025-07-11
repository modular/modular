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

FailureOr<TypedAttr> ParserEvaluationContext::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
  if (auto bindParams = dyn_cast<BindParamsAttr>(attr))
    return evaluateBindParams(bindParams.getGenerator(),
                              bindParams.getParamValues());

  // Handle simplifiable cases here.
  if (auto getWitness = dyn_cast<GetWitnessAttr>(attr))
    return evaluateGetWitness(
        getWitness.getTypeValue(), getWitness.getTraitName(),
        getWitness.getWitnessName(), getWitness.getType());

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}

//===----------------------------------------------------------------------===//
// BindParamsAttr
//===----------------------------------------------------------------------===//

TypedAttr
ParserEvaluationContext::getBindParamsAttr(TypedAttr generator,
                                           ArrayRef<TypedAttr> paramValues) {
  // Try to simplify immediately.
  auto specializedGenerator = evaluateBindParams(generator, paramValues);
  if (succeeded(specializedGenerator))
    return specializedGenerator.value();

  // Otherwise, use the default builder and then attempt evaluation again in
  // case any folders were triggered at build time.
  auto bindParamsAttr = BindParamsAttr::get(generator, paramValues, this);
  if (auto evaluatable =
          dyn_cast<ContextuallyEvaluatedAttrInterface>(bindParamsAttr)) {
    return evaluateExpression(evaluatable).value_or(bindParamsAttr);
  }
  return bindParamsAttr;
}

FailureOr<TypedAttr>
ParserEvaluationContext::evaluateBindParams(TypedAttr generator,
                                            ArrayRef<TypedAttr> paramValues) {
  // Can simplify if the generator is a GeneratorAttr.
  auto genAttr = dyn_cast<GeneratorAttr>(generator);
  if (!genAttr)
    return failure();

  // If the params aren't fully bound, no point in simplifying yet.
  if (paramValues.size() != genAttr.getInputParamTypes().size())
    return failure();

  if (llvm::any_of(paramValues, [](TypedAttr paramValue) {
        return isa<UnboundAttr>(paramValue);
      }))
    return failure();

  GeneratorAttr specializedGenerator =
      genAttr.getSpecializedGenerator(paramValues, /*emitErrorFn=*/{}, this);
  return specializedGenerator.getInstantiatedBody();
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
  auto conformanceDecls = decl.lookupInCurrentScope(traitName);
  // If no conformance exists, still allow it to go through, just don't fold.
  if (conformanceDecls.empty())
    return failure();

  assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
  auto conformanceOp =
      dyn_cast<ConformanceOp>(conformanceDecls.front()->getIfOperation());

  ParserParameterEvaluator nestedEvaluator(shared);
  for (auto [param, value] :
       llvm::zip(structDeclOp.getInputParams(), structType.getParamValues()))
    nestedEvaluator.setParameterValue(param, value);

  auto getWitness =
      GetWitnessAttr::get(typeParam, traitName, witnessName, type);
  FailureOr<TypedAttr> simplified =
      getWitness.simplify(conformanceOp, &nestedEvaluator);
  if (failed(simplified) || !simplified.value())
    return cast<TypedAttr>(getWitness);

  return simplified.value();
}
