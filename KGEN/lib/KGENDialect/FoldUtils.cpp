//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/FoldUtils.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/KGENUtils.h"
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

/// Whether this dtype's value depends on the target's index bit width.
static bool isTargetDependentDType(KGENDType dtype) {
  return dtype.isIndex() || dtype.isUIndex() || dtype.isAddress();
}

static bool hasTargetDependentDType(SIMDAttr attr) {
  std::optional<KGENDType> dtype = attr.getType().getResolvedDType();
  return dtype && isTargetDependentDType(*dtype);
}

/// What a value tree holds that stops its representation from being canonical
/// for the value it denotes.
struct NonCanonicalLeaves {
  /// A value nobody knows. Nothing decides a comparison against it.
  bool unknown = false;
  /// Index-like data, so the target's index bit width decides the comparison.
  bool targetDependent = false;
};

/// Find non-canonical leaves anywhere in an attribute tree.
static NonCanonicalLeaves findNonCanonicalLeaves(Attribute attr) {
  NonCanonicalLeaves found;
  attr.walk([&](Attribute sub) {
    if (isa<UnknownAttr>(sub)) {
      // An unknown decides the whole verdict, so nothing further matters.
      found.unknown = true;
      return mlir::WalkResult::interrupt();
    }
    if (auto simd = dyn_cast<SIMDAttr>(sub);
        simd && hasTargetDependentDType(simd)) {
      found.targetDependent = true;
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

bool containsUnknownValue(Attribute attr) {
  return findNonCanonicalLeaves(attr).unknown;
}

/// Re-express `attr`'s index-like leaves as the values they denote at
/// `indexBitWidth`, so that two trees denoting the same value at that width
/// become the same uniqued attribute.
static Attribute normalizeIndexData(Attribute attr, int64_t indexBitWidth) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](SIMDAttr simd) -> std::pair<Attribute, mlir::WalkResult> {
        if (!hasTargetDependentDType(simd))
          return {simd, mlir::WalkResult::advance()};

        KGENDType dtype = *simd.getType().getResolvedDType();
        // Index is signed; the other index-like dtypes are unsigned.
        bool isUnsigned = !dtype.isIndex();
        assert(!isUnsigned || dtype.isUIndex() || dtype.isAddress());
        SmallVector<DTypeValue> values;
        for (const DTypeValue &value : simd.getValues()) {
          // Store at exactly `indexBitWidth`, so both sides of a comparison end
          // up stored the same way.
          const APInt &data = value.getData();
          values.emplace_back(isUnsigned ? data.zextOrTrunc(indexBitWidth)
                                         : data.sextOrTrunc(indexBitWidth),
                              dtype);
        }
        // Nothing below a SIMD constant needs normalizing.
        return {SIMDAttr::get(values, simd.getType()),
                mlir::WalkResult::skip()};
      });
  return replacer.replace(attr);
}

std::optional<bool> areSimpleConstantsEqual(TypedAttr lhs, TypedAttr rhs,
                                            TargetInfoAttr target) {
  NonCanonicalLeaves lhsLeaves = findNonCanonicalLeaves(lhs);
  NonCanonicalLeaves rhsLeaves = findNonCanonicalLeaves(rhs);

  // An unknown stands in for a value nobody knows, so no comparison of the
  // representations says anything about the values, not even the same
  // representation twice.
  if (lhsLeaves.unknown || rhsLeaves.unknown)
    return std::nullopt;

  // Uniquing already answers this, whatever the target.
  if (lhs == rhs)
    return true;

  // With no index-like data anywhere in either tree, the representation is
  // canonical for the value and inequality is the answer.
  if (!lhsLeaves.targetDependent && !rhsLeaves.targetDependent)
    return false;

  if (target) {
    int64_t indexBitWidth = target.resolveIndexBitWidth();
    return normalizeIndexData(lhs, indexBitWidth) ==
           normalizeIndexData(rhs, indexBitWidth);
  }

  // No target: decide only where both candidate index widths agree, mirroring
  // how `foldSIMDOpIndex` folds target-dependent arithmetic without a target.
  //
  // Equal values at 64 bits truncate to equal values at 32, so agreement at 64
  // settles every width. Only a mismatch there needs the 32-bit answer, to tell
  // "a different value" apart from "depends on the target".
  if (normalizeIndexData(lhs, 64) == normalizeIndexData(rhs, 64))
    return true;
  if (normalizeIndexData(lhs, 32) == normalizeIndexData(rhs, 32))
    return std::nullopt;
  return false;
}

} // namespace M::KGEN
