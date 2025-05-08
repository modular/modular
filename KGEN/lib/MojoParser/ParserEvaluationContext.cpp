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
  // Handle simplifiable cases here.

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}
