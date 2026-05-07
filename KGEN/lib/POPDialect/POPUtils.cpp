//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions primarily for parsing, printing and
// verifying POP related operations and types.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPUtils.h"
#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;
using namespace POP;

/// Get the value of a scalar index-like parameter value.
/// This is a temporary helper utility during the Int->SIMD unification project.
/// After it's done, we should remove the IntegerAttr case.
ErrorOr<int64_t> POP::getScalarIndexValue(TypedAttr value) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return intAttr.getInt();
  if (auto simdAttr = dyn_cast<KGEN::SIMDAttr>(value)) {
    ArrayRef<DTypeValue> values = simdAttr.getValues();
    if (values.size() != 1)
      return Error("expected a scalar SIMD value");
    if (!values.front().getDType().isIndex())
      return Error("expected an index-typed SIMD value");
    return values.front().getIndexVal();
  }
  return Error("expected an integer or scalar SIMD");
}

// verifyConversionCast / foldCastToBuiltin / foldCastFromBuiltin migrated to
// KGENUtils.h/cpp.

/// Fold a cast between two SIMD types.
OpFoldResult POP::foldCast(TypedAttr operand, SIMDType resultType,
                           SIMDType inputType, SIMDType outputType,
                           std::optional<int64_t> indexBitWidth) {
  auto simdOperand = sugarDynCastIfPresent<SIMDAttr>(operand);
  std::optional<KGENDType> dtype = resultType.getResolvedDType();

  if (!simdOperand || !dtype) {
    if (inputType == outputType)
      return operand;
    return {};
  }

  std::optional<KGENDType> inType = simdOperand.getType().getResolvedDType();

  // Exit early if the input and output dtypes are the same.
  if (*dtype == *inType)
    return simdOperand;

  if (dtype->isFloat()) {
    // Cannot fold cast to unsupported float dtype.
    const llvm::fltSemantics *sem = dtype->getFloatSemantics();
    if (!sem)
      return {};
    return foldSIMDOpResult<kOtherResult>(
        simdOperand, *dtype, indexBitWidth,
        [&](const APSInt &in) -> APFloat {
          APFloat fp(*sem);
          fp.convertFromAPInt(in, in.isSigned(), APFloat::rmNearestTiesToEven);
          return fp;
        },
        [&](APFloat in) {
          bool ignored;
          in.convert(*sem, APFloat::rmNearestTiesToEven, &ignored);
          return in;
        },
        [&](bool in) { return APFloat(*sem, in); });
  }

  if (dtype->isInt()) {
    // Note that float to integer casts are undefined if the float value is
    // too large to fit in the integer dtype.
    unsigned intWidth = dtype->getIntegerWidthInBits();
    return foldSIMDOpResult<kOtherResult>(
        simdOperand, *dtype, indexBitWidth,
        [&](const APSInt &in) -> APSInt { return in.extOrTrunc(intWidth); },
        [&](const APFloat &in) -> std::optional<APSInt> {
          APSInt iv(intWidth, /*isUnsigned=*/dtype->isUInt());
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv;
        },
        [&](bool in) { return APSInt(APInt(intWidth, in), dtype->isUInt()); });
  }

  if (dtype->isIndex() || dtype->isUIndex() || dtype->isAddress()) {
    // The folding infra ensures that platform-dependent types are only folded
    // if the folding result is the same on 32 and 64 bit platforms. This is the
    // right thing to do for most operations, but casting can be safely allowed
    // between platform-dependent types.
    if (inType->isIndex() || inType->isUIndex() || inType->isAddress()) {
      return Detail::foldSIMDOpImpl(
          std::make_index_sequence<1>(), simdOperand,
          [inType](const APSInt &in) -> int64_t {
            return inType->isSInt() ? in.getSExtValue() : in.getZExtValue();
          },
          *dtype,
          [](DTypeValue val) {
            return APSInt(APInt(64, val.getIndexVal()),
                          /*isUnsigned=*/!val.getDType().isIndex());
          });
    }

    // Cast to index like it's a 64-bit integer. Address is handled like index.
    return foldSIMDOpResult<kOtherResult>(
        simdOperand, *dtype,
        [inType](const APSInt &in) -> int64_t {
          if (in.getSignificantBits() > 64) {
            auto truncated = in.trunc(64);
            return inType->isSInt() ? truncated.getSExtValue()
                                    : truncated.getZExtValue();
          }
          return inType->isSInt() ? in.getSExtValue() : in.getZExtValue();
        },
        [](const APFloat &in) -> std::optional<int64_t> {
          APSInt iv(64, /*isUnsigned=*/false);
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv.getSExtValue();
        },
        [](bool in) { return static_cast<int64_t>(in); });
  }

  if (dtype->isInvalid()) {
    // Invalid is not inhabitable.
    return {};
  }

  assert(dtype->isBool());
  return foldSIMDOpResult<kOtherResult>(
      simdOperand, *dtype,
      [](const APSInt &in) -> bool { return !in.isZero(); },
      [](const APFloat &in) -> bool { return !in.isZero(); });
}

/// Fold a SIMD Or-reduction operation.
OpFoldResult POP::foldSIMDReduceOr(Value vectorVal, Attribute vectorAttr,
                                   SIMDType vectorType) {
  // If the vector only has one element, it's already reduced.
  if (vectorType.getResolvedSize() == 1)
    return vectorAttr ? OpFoldResult(vectorAttr) : vectorVal;

  if (auto dtype = vectorType.getResolvedDType(); !dtype || !dtype->isIntLike())
    return {};

  return foldBitwiseSIMDReduceOp(
      vectorAttr, [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs | rhs; });
}

/// Fold a SIMD And-reduction operation.
OpFoldResult POP::foldSIMDReduceAnd(Value vectorVal, Attribute vectorAttr,
                                    SIMDType vectorType) {
  // If the vector only has one element, it's already reduced.
  if (vectorType.getResolvedSize() == 1)
    return vectorAttr ? OpFoldResult(vectorAttr) : vectorVal;

  if (auto dtype = vectorType.getResolvedDType(); !dtype || !dtype->isIntLike())
    return {};

  return foldBitwiseSIMDReduceOp(
      vectorAttr, [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs & rhs; });
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

FoldValue POP::foldSIMDCmp(CmpPredicate cc, FoldValues operands,
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

FoldValue POP::foldSIMDShl(FoldValues operands, TargetInfoAttr target) {
  assert(operands.size() == 2 && "expected binary shift operands");
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  if (auto fold =
          foldSIMDOp(operands.getAttrs(), indexBitWidth,
                     [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                       if (rhs.uge(lhs.getBitWidth()))
                         return std::nullopt;
                       return APSInt(lhs.shl(rhs), !lhs.isSigned());
                     }))
    return FoldValue(fold);

  return {};
}

FoldValue POP::foldSIMDShr(FoldValues operands, TargetInfoAttr target) {
  assert(operands.size() == 2 && "expected binary shift operands");
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  if (auto fold = foldSIMDOp(
          operands.getAttrs(), indexBitWidth,
          [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
            if (rhs.uge(lhs.getBitWidth()))
              return std::nullopt;
            return APSInt(lhs.isSigned() ? lhs.ashr(rhs) : lhs.lshr(rhs),
                          !lhs.isSigned());
          }))
    return FoldValue(fold);

  return {};
}

FoldValue POP::foldSIMDAbs(FoldValues operands, TargetInfoAttr target) {
  assert(operands.size() == 1 && "expected unary abs operand");

  auto operandAttr = operands.getAttr<SIMDAttr>(0);
  if (!operandAttr)
    return {};
  std::optional<KGENDType> dtype = operandAttr.getType().getResolvedDType();
  if (!dtype || dtype->isInvalid())
    return {};

  // Bools or unsigned types are already abs'd.
  if (dtype->isBool() || dtype->isUInt())
    return operands[0];

  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  if (auto fold = foldSIMDOp(
          operands.getAttrs(), indexBitWidth,
          [](APFloat val) -> APFloat {
            val.clearSign();
            return val;
          },
          [](APSInt val) -> std::optional<APSInt> {
            return APSInt(val.abs(), /*isUnsigned=*/false);
          }))
    return FoldValue(fold);

  return {};
}

OpFoldResult POP::foldSIMDRound(Attribute operand, TargetInfoAttr targetInfo) {
  auto operandSIMD = sugarDynCastIfPresent<SIMDAttr>(operand);
  if (!operandSIMD)
    return {};
  std::optional<KGENDType> dtype = operandSIMD.getType().getResolvedDType();
  if (!dtype || dtype->isInvalid())
    return {};

  // Anything that isn't a floating-point value is already rounded
  if (!dtype->isFloat())
    return operand;

  return foldSIMDOp(operandSIMD, [](APFloat operand) -> APFloat {
    operand.roundToIntegral(APFloat::rmNearestTiesToEven);
    return operand;
  });
}

FoldValue POP::foldSIMDDiv(FoldValues operands, TargetInfoAttr target) {
  assert(operands.size() == 2 && "expected binary div operands");
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  if (auto fold = foldSIMDOp(
          operands.getAttrs(), indexBitWidth,
          [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
            if (rhs.isZero())
              return std::nullopt;
            return lhs / rhs;
          },
          [](APFloat lhs, APFloat rhs) { return lhs / rhs; },
          [](bool lhs, bool rhs) -> std::optional<bool> {
            if (!rhs)
              return std::nullopt;
            return lhs;
          }))
    return FoldValue(fold);

  return {};
}

FoldValue POP::foldSIMDFloorDiv(FoldValues operands, TargetInfoAttr target) {
  assert(operands.size() == 2 && "expected binary floordiv operands");
  std::optional<int64_t> indexBitWidth;
  if (target)
    indexBitWidth = target.resolveIndexBitWidth();

  if (auto fold = foldSIMDOp(
          operands.getAttrs(), indexBitWidth,
          [](APFloat lhs, APFloat rhs) -> APFloat {
            auto div = lhs / rhs;
            div.roundToIntegral(APFloat::rmTowardNegative);
            return div;
          },
          [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
            // Integer division by zero is UB - don't fold.
            if (rhs.isZero())
              return std::nullopt;
            APSInt div = lhs / rhs;
            if (lhs.isUnsigned())
              return div;
            // int div = lhs / rhs;
            // return div * rhs == lhs ? div : div - ((lhs < 0) ^ (rhs < 0));
            int xorOp = (lhs < 0) ^ (rhs < 0);
            return APSInt(div * rhs == lhs ? div : div - xorOp,
                          /*isUnsigned=*/false);
          }))
    return FoldValue(fold);

  return {};
}

template <typename State>
static ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src,
                                          Attribute len, Location loc,
                                          State &state) {
  auto lenAttr = dyn_cast<IntegerAttr>(len);
  if (!lenAttr)
    return ErrorTree(loc, "interpreting memcpy 3nd operand len is not "
                          "interpreted correctly");

  if (!lenAttr.getInt())
    return success();

  auto dstPtr = dyn_cast<M::PointerAttr>(dst);
  auto srcPtr = dyn_cast<M::PointerAttr>(src);

  if (!dstPtr)
    return ErrorTree(loc, "interpreting memcpy 1st operand dst addr is "
                          "not interpreted correctly");
  if (!srcPtr)
    return ErrorTree(loc, "interpreting memcpy 2nd operand src addr is "
                          "not interpreted correctly");

  ErrorOr<void *> dstAddrOr =
      state.getWritableMemory(dstPtr.getAddr(), size_t(lenAttr.getInt()));
  ErrorOr<const void *> srcAddrOr =
      state.getReadableMemory(srcPtr.getAddr(), size_t(lenAttr.getInt()));

  if (dstAddrOr.isError())
    return ErrorTree(
        loc, "interpreting memcpy can't get dst memory from the interpreter");

  if (srcAddrOr.isError())
    return ErrorTree(
        loc, "interpreting memcpy can't get src memory from the interpreter");

  std::memcpy(*dstAddrOr, *srcAddrOr, lenAttr.getInt());

  return success();
}

ErrorTreeOrSuccess POP::interpretMemcpy(Attribute dst, Attribute src,
                                        Attribute len, Location loc,
                                        InterpreterState &state) {
  return ::interpretMemcpy<InterpreterState>(dst, src, len, loc, state);
}

ErrorTreeOrSuccess POP::interpretMemcpy(Attribute dst, Attribute src,
                                        Attribute len, Location loc,
                                        ParametricInterpreterState &state) {
  return ::interpretMemcpy<ParametricInterpreterState>(dst, src, len, loc,
                                                       state);
}
