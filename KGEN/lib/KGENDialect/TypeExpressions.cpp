//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"

using namespace M;
using namespace KGEN;

Optional<TypedAttr>
ParameterEvaluator::evaluateSymbolicExpression(ParamOperatorAttr op) {
  // No supported symbolic expressions yet.
  return {op};
}
