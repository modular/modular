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

template <typename T>
static bool compareConstants(CmpPredicate pred, T lhs, T rhs) {
  switch (pred) {
  case CmpPredicate::EQ:
    return lhs == rhs;
  case CmpPredicate::NE:
    return lhs != rhs;
  case CmpPredicate::LT:
    return lhs < rhs;
  case CmpPredicate::GT:
    return lhs > rhs;
  case CmpPredicate::LE:
    return lhs <= rhs;
  case CmpPredicate::GE:
    return lhs >= rhs;
  }
  llvm_unreachable("invalid CmpPredicate");
}

FoldValue foldSIMDCmp(CmpPredicate cc, FoldValues operands,
                      TargetInfoAttr target) {
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();
  assert(operands.size() == 2 && "expected binary compare operands");

  std::optional<int64_t> size =
      cast<SIMDType>(operands[0].getType()).getResolvedSize();

  std::optional<KGENDType> inDType =
      cast<SIMDType>(operands[0].getType()).getResolvedDType();
  if (!inDType)
    inDType = cast<SIMDType>(operands[1].getType()).getResolvedDType();

  // Fold cmp(x, x) for int-like types (NaN prevents this for floats).
  if (operands[0] == operands[1] && size && inDType && inDType->isIntLike()) {
    bool isTrue = llvm::is_contained(
        {CmpPredicate::EQ, CmpPredicate::LE, CmpPredicate::GE}, cc);
    SmallVector<DTypeValue> vals(*size, {isTrue, KGENDType::kBool});
    MLIRContext *ctx = operands[0].getType().getContext();
    return FoldValue(
        SIMDAttr::get(vals, SIMDType::get(ctx, *size, KGENDType::kBool)));
  }

  // Constant fold when both operands are SIMDAttr constants.
  if (auto fold = foldSIMDOpResult<kOtherResult>(
          operands.getAttrs(), KGENDType::kBool, indexBitWidth,
          [&](APSInt l, APSInt r) { return compareConstants(cc, l, r); },
          [&](APFloat l, APFloat r) { return compareConstants(cc, l, r); },
          [&](bool l, bool r) { return compareConstants(cc, l, r); }))
    return FoldValue(fold);

  auto lhsAttr = operands.getAttr<SIMDAttr>(0);
  auto rhsAttr = operands.getAttr<SIMDAttr>(1);

  // Fold `eq(true, x) -> x` and `ne(false, x) -> x`.
  if (inDType && *inDType == KGENDType::kBool &&
      llvm::is_contained({CmpPredicate::EQ, CmpPredicate::NE}, cc)) {
    SIMDAttr constAttr = lhsAttr ? lhsAttr : rhsAttr;
    FoldValue otherValue = lhsAttr ? operands[1] : operands[0];
    if (constAttr && otherValue && llvm::all_equal(constAttr.getValues()) &&
        (cc == CmpPredicate::EQ) == constAttr.getValues().front().getBoolVal())
      return otherValue;
  }

  // Fold unsigned comparisons with zero:
  //   gt(0, x) -> false, le(0, x) -> true
  //   ge(x, 0) -> true,  lt(x, 0) -> false
  if (inDType && size && inDType->isUInt()) {
    auto tryFoldWithZero = [&](SIMDAttr zeroCandidate, SIMDAttr otherCandidate,
                               CmpPredicate foldTrue,
                               CmpPredicate foldFalse) -> TypedAttr {
      if (!llvm::is_contained({foldTrue, foldFalse}, cc))
        return {};
      if (zeroCandidate && otherCandidate)
        return {};
      if (!zeroCandidate)
        return {};
      if (llvm::all_equal(zeroCandidate.getValues()) &&
          zeroCandidate.getValues()[0].getData().isZero()) {
        SmallVector<DTypeValue> values(
            *size, DTypeValue(cc == foldTrue, KGENDType::kBool));
        MLIRContext *ctx = operands[0].getType().getContext();
        return SIMDAttr::get(values,
                             SIMDType::get(ctx, *size, KGENDType::kBool));
      }
      return {};
    };
    if (auto res = tryFoldWithZero(lhsAttr, rhsAttr, CmpPredicate::LE,
                                   CmpPredicate::GT))
      return res;
    if (auto res = tryFoldWithZero(rhsAttr, lhsAttr, CmpPredicate::GE,
                                   CmpPredicate::LT))
      return res;
  }

  return {};
}

} // namespace M::KGEN
