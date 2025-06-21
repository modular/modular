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

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}

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
