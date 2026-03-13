//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/FoldUtils.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"

namespace M::KGEN {

FailureOr<TypedAttr> foldAttrWithTarget(ParameterEvaluationContext &context,
                                        ArrayRef<Attribute> operands,
                                        TargetAwareFoldFn fold) {
  auto target = context.getTargetInfo();
  if (!target)
    return failure();
  if (auto result = fold(FoldValues(operands), target)) {
    assert(result.getAttr() && "attribute fold should produce an attribute");
    return result.getAttr();
  }
  return failure();
}

ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       InterpreterState &state,
                                       TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), state.getTarget())) {
    if (auto attr = result.getAttr()) {
      state.mapResults(attr);
      return success();
    }
  }
  return ErrorTree(loc, "failed to interpret " + opName);
}

ErrorTreeOrSuccess interpretOpWithFold(Location loc, StringRef opName,
                                       ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state,
                                       TargetAwareFoldFn fold) {
  if (auto result = fold(FoldValues(operands), state.getTarget())) {
    if (auto attr = result.getAttr()) {
      state.mapResults(attr);
      return success();
    }
  }
  return ErrorTree(loc, "failed to interpret " + opName);
}

} // namespace M::KGEN
