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

#include "Mojo/ToolCommon/KGENPasses.h"

#include "CABILowering.h"
#include "LLVMLoweringUtils.h"
#include "LowerPOPToLLVMExternalCalls.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetLowering.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include <mlir/IR/SymbolTable.h>
#include <utility>

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERGLOBALPOPTOLLVM
#define GEN_PASS_DEF_LOWERPOPTOLLVM
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// POP dialect fastmath flags match the LLVM ones.
static LLVM::FastmathFlagsAttr convertFastmathFlags(FastmathFlags fmf,
                                                    OpBuilder &builder) {
  return builder.getAttr<LLVM::FastmathFlagsAttr>(
      static_cast<LLVM::FastmathFlags>(fmf));
}

/// Returns the divisor of `op`, at `bitWidth` bits, when every lane is the same
/// integer constant. The divisor is still a `kgen.param.constant` if it has not
/// been lowered yet, and an `llvm.mlir.constant` if it has.
static std::optional<APInt>
getSplatConstantDivisor(FloorDivOp op, Value llvmRhs, unsigned bitWidth) {
  auto getSplat = [&](auto &&values) -> std::optional<APInt> {
    std::optional<APInt> result;
    for (APInt value : values) {
      value = value.sextOrTrunc(bitWidth);
      if (result && *result != value)
        return std::nullopt;
      result = value;
    }
    return result;
  };

  if (auto cst = op.getRhs().getDefiningOp<ParamConstantOp>())
    if (auto simd = dyn_cast<SIMDAttr>(cst.getValue()))
      return getSplat(llvm::map_range(
          simd.getValues(),
          [](const DTypeValue &value) -> APInt { return value.getData(); }));

  auto cst = llvmRhs.getDefiningOp<LLVM::ConstantOp>();
  if (!cst)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
    return intAttr.getValue().sextOrTrunc(bitWidth);
  if (auto array = dyn_cast<IntArrayElementsAttr>(cst.getValue()))
    return getSplat(array.getValues());
  if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
    return getSplat(dense.getValues<APInt>());
  return std::nullopt;
}

/// Constants for Ben Joffe's signed floor division by a constant, from
/// https://www.benjoffe.com/fast-floordiv and the reference implementation at
/// https://github.com/benjoffe/fastdiv/blob/main/algorithms/benjoffe.hpp.
/// For a W-bit dividend x, every mode computes `(t * m) >> shift` on the 2W-bit
/// product, which is exact over the whole signed range:
///
///   SignSelected  t = x,  m = multiplier + (x >> (W - 1))
///   RoundUp       t = x + (x < -1)                    (d > 0)
///                 t = x - (x > 1)                     (d < 0)
///   RoundDown     t = x + (uint(x) < 2^(W-1) - 1)     (d > 0)
///                 t = x - (uint(x) > 2^(W-1))         (d < 0)
struct FloorDivMagic {
  enum class Mode { SignSelected, RoundUp, RoundDown };
  Mode mode;
  /// In [W, 2W).
  unsigned shift;
  /// Signed W-bit multiplier carrying the divisor's sign.
  APInt multiplier;
};

/// Returns the constants for dividing by `divisor`, or std::nullopt for zero
/// and for +-2^k, which the generic sequence already handles with shifts.
static std::optional<FloorDivMagic> computeFloorDivMagic(const APInt &divisor) {
  if (divisor.isZero())
    return std::nullopt;
  unsigned width = divisor.getBitWidth();
  // Twice the width keeps |INT_MIN|, 2^shift and bound * err exact.
  unsigned wideWidth = 2 * width;
  bool isNeg = divisor.isNegative();
  APInt absDivisor = divisor.sext(wideWidth).abs();
  if (absDivisor.isPowerOf2())
    return std::nullopt;

  // A rounding direction works at shift S when 2^S > bound * err and the
  // multiplier fits, where err is the multiplier's rounding error at 2^S and
  // bound is the largest dividend magnitude, snapped to a multiple of |d|, at
  // which that error could flip the quotient.
  APInt intMinMagnitude = APInt::getOneBitSet(wideWidth, width - 1);
  APInt top = intMinMagnitude + (isNeg ? 1 : 0);
  APInt boundUp = top.udiv(absDivisor) * absDivisor - 1;
  APInt downMultiple = intMinMagnitude.udiv(absDivisor) * absDivisor + 1;
  APInt boundDown = (isNeg && downMultiple == intMinMagnitude)
                        ? downMultiple - 1
                        : downMultiple;

  auto getMultiplier = [&](unsigned shift, bool roundUp) {
    return APInt::getOneBitSet(wideWidth, shift).udiv(absDivisor) +
           (roundUp ? 1 : 0);
  };
  auto findShift = [&](bool roundUp) -> std::optional<unsigned> {
    for (unsigned shift = width; shift < wideWidth; ++shift) {
      APInt pow = APInt::getOneBitSet(wideWidth, shift);
      APInt rem = pow.urem(absDivisor);
      APInt err = roundUp ? absDivisor - rem : rem;
      if (pow.ugt((roundUp ? boundUp : boundDown) * err) &&
          getMultiplier(shift, roundUp).ult(intMinMagnitude))
        return shift;
    }
    return std::nullopt;
  };
  std::optional<unsigned> upShift = findShift(/*roundUp=*/true);
  std::optional<unsigned> downShift = findShift(/*roundUp=*/false);

  // The reference favors a round-down shift of exactly W, whose result is the
  // high half of the product, as it wins on every machine it measured. It also
  // favors a round-up shift of exactly W off x86; this lowering does not know
  // the target, so it takes the sign-selected form there, which the reference
  // found ties or wins on x86.
  FloorDivMagic magic;
  bool roundUp;
  if (downShift &&
      (!upShift || (width > 8 && *upShift != width && *downShift == width))) {
    magic.mode = FloorDivMagic::Mode::RoundDown;
    magic.shift = *downShift;
    roundUp = false;
  } else if (upShift && !downShift) {
    magic.mode = FloorDivMagic::Mode::RoundUp;
    magic.shift = *upShift;
    roundUp = true;
  } else if (upShift && downShift) {
    magic.mode = FloorDivMagic::Mode::SignSelected;
    magic.shift = std::max(*upShift, *downShift);
    // Positive divisors round up and negative ones round down, so that adding
    // the dividend's sign mask selects the other direction for x < 0.
    roundUp = !isNeg;
  } else {
    return std::nullopt;
  }

  APInt multiplier = getMultiplier(magic.shift, roundUp);
  if (magic.mode == FloorDivMagic::Mode::SignSelected &&
      multiplier.uge(intMinMagnitude))
    return std::nullopt;
  magic.multiplier = multiplier.trunc(width);
  if (isNeg)
    magic.multiplier.negate();
  return magic;
}

/// Creates an integer constant of `type`, splatted if `type` is a vector.
static Value createIntConstant(OpBuilder &rewriter, Location loc, Type type,
                               const APInt &value) {
  if (auto vecType = dyn_cast<VectorType>(type)) {
    SmallVector<APInt> values(vecType.getNumElements(), value);
    return LLVM::ConstantOp::create(
        rewriter, loc,
        cast<TypedAttr>(IntArrayElementsAttr::get(vecType, values)));
  }
  return LLVM::ConstantOp::create(rewriter, loc,
                                  rewriter.getIntegerAttr(type, value));
}

/// Emits `floor(lhs / d)` for the constant divisor described by `magic`. For
/// iW, with M the multiplier and S the shift, the emitted LLVM is:
///
///   ; SignSelected
///   %sign = ashr iW %x, W-1
///   %m    = add iW M, %sign
///   %t    = %x
///   ; RoundUp (d > 0: slt -1, add; d < 0: sgt 1, sub)
///   ; RoundDown (d > 0: ult INT_MIN-1, add; d < 0: ugt INT_MIN, sub)
///   %c    = icmp <pred> iW %x, <threshold>
///   %bias = zext i1 %c to iW
///   %t    = add/sub iW %x, %bias
///   %m    = M
///   ; All modes
///   %tw   = sext iW %t to i2W
///   %mw   = sext iW %m to i2W
///   %p    = mul i2W %tw, %mw
///   %hi   = ashr i2W %p, W
///   %q    = trunc i2W %hi to iW
///   %q    = ashr iW %q, S-W          ; only when S > W
static Value emitFloorDivByConstant(OpBuilder &rewriter, Location loc,
                                    Type type, Value lhs,
                                    const FloorDivMagic &magic) {
  unsigned width = magic.multiplier.getBitWidth();
  bool isNeg = magic.multiplier.isNegative();
  auto getConstant = [&](const APInt &value) {
    return createIntConstant(rewriter, loc, type, value);
  };

  Value factor = lhs;
  Value multiplier = getConstant(magic.multiplier);
  if (magic.mode == FloorDivMagic::Mode::SignSelected) {
    Value signMask = LLVM::AShrOp::create(rewriter, loc, type, lhs,
                                          getConstant(APInt(width, width - 1)));
    multiplier = LLVM::AddOp::create(rewriter, loc, type, multiplier, signMask);
  } else {
    APInt intMin = APInt::getSignedMinValue(width);
    LLVM::ICmpPredicate predicate;
    APInt threshold;
    if (magic.mode == FloorDivMagic::Mode::RoundUp) {
      predicate = isNeg ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::slt;
      threshold = isNeg ? APInt(width, 1) : APInt::getAllOnes(width);
    } else {
      predicate = isNeg ? LLVM::ICmpPredicate::ugt : LLVM::ICmpPredicate::ult;
      threshold = isNeg ? intMin : intMin - 1;
    }
    Value bias = LLVM::ICmpOp::create(rewriter, loc, predicate, lhs,
                                      getConstant(threshold));
    bias = LLVM::ZExtOp::create(rewriter, loc, type, bias);
    if (isNeg)
      factor = LLVM::SubOp::create(rewriter, loc, type, lhs, bias);
    else
      factor = LLVM::AddOp::create(rewriter, loc, type, lhs, bias);
  }

  // Take the high half before the remaining shift so the backend sees the
  // signed multiply-high idiom.
  Type wideType = IntegerType::get(rewriter.getContext(), 2 * width);
  if (auto vecType = dyn_cast<VectorType>(type))
    wideType = vecType.clone(wideType);
  Value product = LLVM::MulOp::create(
      rewriter, loc, wideType,
      LLVM::SExtOp::create(rewriter, loc, wideType, factor),
      LLVM::SExtOp::create(rewriter, loc, wideType, multiplier));
  Value high = LLVM::AShrOp::create(
      rewriter, loc, wideType, product,
      createIntConstant(rewriter, loc, wideType, APInt(2 * width, width)));
  Value result = LLVM::TruncOp::create(rewriter, loc, type, high);
  if (magic.shift > width)
    result =
        LLVM::AShrOp::create(rewriter, loc, type, result,
                             getConstant(APInt(width, magic.shift - width)));
  return result;
}

/// Emits `floor(lhs / d)` for a runtime, uniform divisor `d` with
/// `|d| <= 2^31`, using the same multiply-high shape the constant path uses,
/// with the magic constants computed at run time:
///
///   A   = |d|                       ; 0 < A <= 2^31
///   s   = 31 + bitwidth(A - 1)      ; 2^s >= A * 2^31
///   c   = ceil(2^s / A)             ; the 33-bit multiplier of div32.h,
///                                   ; up to one more from the estimate
///   u   = |x|
///   sel = sign(x) != sign(d)        ; the quotient is negated
///   b   = sel & (u != 0)            ; floor or ceiling of |x| / A
///   t   = u - b
///   q   = A is a power of two ? t >> log2(A) : mulhi_u32(t, c) >> (s - 32)
///   q  += b
///   q   = sel ? -q : q
///
/// `c` is derived from a `fdiv` and one integer fixup rather than a `udiv`,
/// because `fdiv` is speculatable and `udiv` is not: LLVM hoists the whole
/// constant computation out of a loop even when the division sits under a
/// branch. `|x|` is taken before the division so the dividend never needs a
/// 33rd bit, and the product stays in 32-bit lanes so `mulhi` maps to a vector
/// multiply-high and the sequence vectorizes. See
/// https://github.com/lemire/fastmod for the truncating form and
/// https://www.benjoffe.com/fast-floordiv for the rounding direction.
static Value emitFloorDivByRuntimeDivisor(OpBuilder &rewriter, Location loc,
                                          Type type, Value lhs, Value rhs,
                                          unsigned width) {
  MLIRContext *ctx = rewriter.getContext();
  Type elementType = type;
  Type workType = IntegerType::get(ctx, 32);
  Type wideType = IntegerType::get(ctx, 64);
  Type floatType = Float64Type::get(ctx);
  if (auto vecType = dyn_cast<VectorType>(type)) {
    workType = vecType.clone(cast<IntegerType>(workType));
    wideType = vecType.clone(cast<IntegerType>(wideType));
    floatType = vecType.clone(cast<FloatType>(floatType));
  }

  auto getIntConstant = [&](Type ty, const APInt &value) {
    return createIntConstant(rewriter, loc, ty, value);
  };
  auto zero = getIntConstant(wideType, APInt(64, 0));
  auto constWide = [&](int64_t v) {
    return getIntConstant(wideType, APInt(64, v));
  };

  // A = |d|, as a 64-bit magnitude.
  Value signMask = LLVM::AShrOp::create(
      rewriter, loc, elementType, rhs,
      getIntConstant(elementType, APInt(width, width - 1)));
  Value magnitude = LLVM::SubOp::create(
      rewriter, loc, elementType,
      LLVM::XOrOp::create(rewriter, loc, elementType, rhs, signMask), signMask);
  Value absDivisor = LLVM::ZExtOp::create(rewriter, loc, wideType, magnitude);
  Value absDivisorMinusOne =
      LLVM::SubOp::create(rewriter, loc, wideType, absDivisor, constWide(1));

  Value isPow2 =
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                           LLVM::AndOp::create(rewriter, loc, wideType,
                                               absDivisor, absDivisorMinusOne),
                           zero);

  // s = 95 - ctlz(A - 1), which is 31 + bitwidth(A - 1); ctlz(0) is 64, so
  // A == 1 gives s = 31.
  Value leadingZeros = LLVM::CountLeadingZerosOp::create(
      rewriter, loc, wideType, absDivisorMinusOne, /*is_zero_poison=*/false);
  Value shift =
      LLVM::SubOp::create(rewriter, loc, wideType, constWide(95), leadingZeros);
  Value multiplyShift = LLVM::SubOp::create(
      rewriter, loc, wideType,
      LLVM::UMaxOp::create(rewriter, loc, wideType, shift, constWide(32)),
      constWide(32));
  Value twoToShift =
      LLVM::ShlOp::create(rewriter, loc, wideType, constWide(1), shift);

  // c = trunc(2^s / absDivisor) + 1, corrected by one integer fixup. The double
  // quotient is within 2^-21 of the exact value, so the estimate lands in
  // [ceil(2^s / absDivisor), ceil(2^s / absDivisor) + 1]: the fixup only rules
  // out the lower end, and both ends stay inside the multiplier interval, whose
  // slack is 2^22 times the estimate's error.
  Value quotient = LLVM::FDivOp::create(
      rewriter, loc, floatType,
      LLVM::UIToFPOp::create(rewriter, loc, floatType, twoToShift),
      LLVM::UIToFPOp::create(rewriter, loc, floatType, absDivisor));
  Value estimate = LLVM::AddOp::create(
      rewriter, loc, wideType,
      LLVM::FPToUIOp::create(rewriter, loc, wideType, quotient), constWide(1));
  Value tooSmall = LLVM::ICmpOp::create(
      rewriter, loc, LLVM::ICmpPredicate::ult,
      LLVM::MulOp::create(rewriter, loc, wideType, estimate, absDivisor),
      twoToShift);
  Value multiplier = LLVM::AddOp::create(
      rewriter, loc, wideType, estimate,
      LLVM::ZExtOp::create(rewriter, loc, wideType, tooSmall));

  Value trailingZeros = LLVM::CountTrailingZerosOp::create(
      rewriter, loc, wideType, absDivisor, /*is_zero_poison=*/false);
  Value multiplier32 =
      LLVM::TruncOp::create(rewriter, loc, workType, multiplier);
  Value multiplyShift32 =
      LLVM::TruncOp::create(rewriter, loc, workType, multiplyShift);
  Value trailingZeros32 =
      LLVM::TruncOp::create(rewriter, loc, workType, trailingZeros);

  // The per-element work stays in 32-bit lanes so the multiply-high remains a
  // vector operation. Narrower elements are widened to those lanes.
  auto widen = [&](Value value) -> Value {
    return width < 32 ? LLVM::SExtOp::create(rewriter, loc, workType, value)
                      : value;
  };
  Value x = widen(lhs);
  Value d = widen(rhs);
  auto constWork = [&](int64_t v) {
    return getIntConstant(workType, APInt(32, v));
  };

  // sel = sign(x) != sign(d); u = |x|; b = sel & (u != 0); t = u - b.
  Value signDiff = LLVM::AShrOp::create(
      rewriter, loc, workType,
      LLVM::XOrOp::create(rewriter, loc, workType, x, d), constWork(31));
  Value negatedX =
      LLVM::AShrOp::create(rewriter, loc, workType, x, constWork(31));
  Value u = LLVM::SubOp::create(
      rewriter, loc, workType,
      LLVM::XOrOp::create(rewriter, loc, workType, x, negatedX), negatedX);
  Value nonZero = LLVM::ZExtOp::create(
      rewriter, loc, workType,
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne, u,
                           constWork(0)));
  Value roundUp =
      LLVM::AndOp::create(rewriter, loc, workType, signDiff, nonZero);
  Value dividend = LLVM::SubOp::create(rewriter, loc, workType, u, roundUp);

  // mulhi_u32(dividend, c) >> (s - 32)
  Value product = LLVM::MulOp::create(
      rewriter, loc, wideType,
      LLVM::ZExtOp::create(rewriter, loc, wideType, dividend),
      LLVM::ZExtOp::create(rewriter, loc, wideType, multiplier32));
  Value productHigh = LLVM::TruncOp::create(
      rewriter, loc, workType,
      LLVM::LShrOp::create(rewriter, loc, wideType, product, constWide(32)));
  Value multiply = LLVM::LShrOp::create(rewriter, loc, workType, productHigh,
                                        multiplyShift32);

  // Powers of two (including |d| == 1) take an exact shift.
  Value pow2Shift =
      LLVM::LShrOp::create(rewriter, loc, workType, dividend, trailingZeros32);
  Value magnitudeQuotient = LLVM::SelectOp::create(rewriter, loc, workType,
                                                   isPow2, pow2Shift, multiply);
  Value rounded =
      LLVM::AddOp::create(rewriter, loc, workType, magnitudeQuotient, roundUp);
  Value result = LLVM::SubOp::create(
      rewriter, loc, workType,
      LLVM::XOrOp::create(rewriter, loc, workType, rounded, signDiff),
      signDiff);
  return width < 32 ? LLVM::TruncOp::create(rewriter, loc, elementType, result)
                    : result;
}

/// Returns true when `op`'s divisor is a runtime value that is uniform across
/// the operation: a scalar, or a splat of a scalar. A lane-varying divisor
/// would need a multiplier per lane, which costs more than the division.
static bool
isUniformRuntimeDivisor(FloorDivOp op, Value loweredDivisor,
                        const llvm::DenseSet<Value> &uniformVectors) {
  if (op.getType().isScalar())
    return true;
  // The operands are lowered before the op, so the splat is recognized through
  // the value its lowering produced, possibly behind a materialization cast.
  while (auto cast =
             loweredDivisor.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    loweredDivisor = cast.getOperand(0);
  // The splat may not have been lowered yet, in which case the operand still
  // names the original value.
  return uniformVectors.contains(loweredDivisor) ||
         loweredDivisor.getDefiningOp<SIMDSplatOp>() != nullptr;
}

/// Generate the LLVM IR to materialize an alloca with the given LLVM type and
/// count. The alloca is created at the top of the given block, and lifetime
/// markers are inserted at the end of the given operation's block.
static Value materializeLLVMAlloca(OpBuilder &b, TargetInfoAttr target,
                                   Type elementType, int64_t count,
                                   Operation *op, int64_t typeAllocSize,
                                   int64_t align) {
  unsigned addressSpace = 0;
  auto alloca = dyn_cast<StackAllocationOp>(op);
  if (alloca)
    addressSpace = alloca.getType().getAddrSpaceOrZero();

  bool needAddrSpaceCast = false;
  if (addressSpace == 0) {
    addressSpace = target.getDataLayout().getAllocaAddrSpace();
    needAddrSpaceCast = addressSpace != 0;
  }

  Value countVal =
      LLVM::ConstantOp::create(b, op->getLoc(), b.getI64IntegerAttr(count));
  Value ptr = LLVM::AllocaOp::create(
      b, op->getLoc(), LLVM::LLVMPointerType::get(b.getContext(), addressSpace),
      elementType, countVal, align);

  if (alloca && alloca.getMarkedLifetimes()) {
    // If this alloca has marked lifetimes, it always begins as dead.
    LLVM::LifetimeEndOp::create(b, op->getLoc(), ptr);
  } else {
    // Insert lifetime markers starting from the op to the end of its block.
    b.setInsertionPoint(op);
    auto start = LLVM::LifetimeStartOp::create(b, op->getLoc(), ptr);
    b.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    LLVM::LifetimeEndOp::create(b, op->getLoc(), ptr);
    b.setInsertionPointAfter(start);
  }

  if (needAddrSpaceCast) {
    ptr = LLVM::AddrSpaceCastOp::create(
        b, op->getLoc(),
        LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/0), ptr);
  }

  return ptr;
}

static Value stripAddrspaceCast(Value value) {
  while (auto cast = value.getDefiningOp<LLVM::AddrSpaceCastOp>())
    value = cast.getOperand();
  return value;
}

template <typename OpT>
static LogicalResult lowerLifetimeMarker(Operation *op, ValueRange values,
                                         TargetInfoAttr target,
                                         POPLoweringBuilder &b) {
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    Value value = values[i];
    SmallVector<Value> newValues;

    // LLVM verifies that pointers used by lifetime marks are defined by alloca
    // instruction directly.
    value = stripAddrspaceCast(value);
    if (!isa_and_present<LLVM::AllocaOp>(value.getDefiningOp()))
      return op->emitError("lifetime marker is only allowed for alloca");

    OpT::create(b, op->getLoc(), value);
  }
  b.deleteLater(op);
  return success();
}

static LLVM::AtomicOrdering getAtomicOrdering(AtomicOrdering ordering) {
  switch (ordering) {
  case AtomicOrdering::NOT_ATOMIC:
    return LLVM::AtomicOrdering::not_atomic;
  case AtomicOrdering::UNORDERED:
    return LLVM::AtomicOrdering::unordered;
  case AtomicOrdering::MONOTONIC:
    return LLVM::AtomicOrdering::monotonic;
  case AtomicOrdering::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case AtomicOrdering::RELEASE:
    return LLVM::AtomicOrdering::release;
  case AtomicOrdering::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  case AtomicOrdering::SEQUENTIALLY_CONSISTENT:
    return LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown atomic ordering");
}

/// Expand one level of structs so kgen.pack elements are passed as individual
/// values instead of as a kgen.struct.
///
/// Takes both the converted LLVM values and the original KGEN types to
/// correctly map logical field indices to LLVM field indices (which may
/// differ due to alignment padding fields).
static SmallVector<Value> expandOperands(OpBuilder &rewriter, Location loc,
                                         ValueRange args, TypeRange origTypes,
                                         const POPToLLVMTypeConverter &tc) {
  SmallVector<Value> operands;
  operands.reserve(args.size());
  for (auto [value, origType] : llvm::zip(args, origTypes)) {
    // Squash pointless conversion casts that will get in the way of folds.
    value = squashPointlessCasts(value);

    // Check if the original type is a KGEN struct type.
    if (auto kgenStructTy = dyn_cast<StructType>(origType)) {
      // Get the element types - should be resolved at this point.
      if (auto elemTypes = kgenStructTy.getElementTypes()) {
        // Iterate over KGEN logical field indices and extract using remapped
        // LLVM indices to account for any padding fields.
        for (size_t i = 0, e = elemTypes->size(); i != e; ++i) {
          int64_t llvmIdx = tc.getRemappedFieldIndex(kgenStructTy, i);
          auto elt =
              rewriter.createOrFold<LLVM::ExtractValueOp>(loc, value, llvmIdx);
          operands.push_back(elt);
        }
      } else {
        // Parametric struct - just push the value as-is.
        operands.push_back(value);
      }
    } else {
      operands.push_back(value);
    }
  }
  return operands;
}

static FailureOr<Value> materializeLLVMUnionAlloca(TargetInfoAttr target,
                                                   OpBuilder &b, Operation *op,
                                                   Type popUnionType,
                                                   Type llvmUnionType) {
  std::optional<int64_t> typeAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, popUnionType);
  std::optional<int64_t> typeABIAlign =
      DataLayoutInterface::getTypeABIAlign(target, popUnionType);
  if (!typeAllocSize || !typeABIAlign)
    return op->emitError("failed to get union type size and alignment");

  return materializeLLVMAlloca(b, target, llvmUnionType, 1, op, *typeAllocSize,
                               *typeABIAlign);
}

} // namespace

//===----------------------------------------------------------------------===//
// LowerPOPToLLVMPass
//===----------------------------------------------------------------------===//

namespace {
struct LowerPOPToLLVMPass
    : public KGEN::impl::LowerPOPToLLVMBase<LowerPOPToLLVMPass> {
  using LowerPOPToLLVMBase::LowerPOPToLLVMBase;
  LowerPOPToLLVMPass(const LowerPOPToLLVMPass &other)
      : LowerPOPToLLVMBase(other) {}

  void runOnOperation() override;

private:
  /// Verify that the operation is a function and has no nested CFGs.
  FailureOr<mlir::FunctionOpInterface> validateOperation();

  LogicalResult lowerBlock(Block &block);
  LogicalResult lowerOperation(Operation *op);
  Value getLoweredValue(Value value);
  ValueRange getLoweredOperands(Operation *op);

  template <typename SourceOp, typename TargetOp>
  LogicalResult lowerOneToOne(SourceOp op) {
    SmallVector<Value> operands = getLoweredOperands(op);
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    builder->setInsertionPoint(op);
    OperationState state(op.getLoc(), TargetOp::getOperationName());
    state.addTypes(resultTypes);
    state.addOperands(operands);
    state.addAttributes(op->getAttrs());
    Operation *newOp = builder->create(state);
    builder->replaceOp(op, newOp);
    return success();
  }

  template <typename Op, typename FloatOp, typename SIntOp,
            typename UIntOp = SIntOp>
  LogicalResult lowerFloatOrInt(Op op) {
    auto adaptor = typename Op::Adaptor(getLoweredOperands(op), op);
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = typeConverter->convertType(op.getType());
    builder->setInsertionPoint(op);
    if (dtype.isBool() || dtype.isInt() || dtype.isIndex() ||
        dtype.isUIndex()) {
      if (std::is_same_v<SIntOp, UIntOp> || dtype.isSInt() || dtype.isIndex()) {
        if constexpr (std::is_same_v<SIntOp, LLVM::SDivOp>)
          builder->replaceOpWithNewOp<SIntOp>(
              op, type, adaptor.getLhs(), adaptor.getRhs(), op.getIsExact());
        else
          builder->replaceOpWithNewOp<SIntOp>(op, type, adaptor.getLhs(),
                                              adaptor.getRhs());
      } else {
        if constexpr (std::is_same_v<UIntOp, LLVM::UDivOp>)
          builder->replaceOpWithNewOp<UIntOp>(
              op, type, adaptor.getLhs(), adaptor.getRhs(), op.getIsExact());
        else
          builder->replaceOpWithNewOp<UIntOp>(op, type, adaptor.getLhs(),
                                              adaptor.getRhs());
      }
      return success();
    }
    builder->replaceOpWithNewOp<FloatOp>(op, type, adaptor.getLhs(),
                                         adaptor.getRhs(),
                                         fastmathFlagsOrDefault(op));
    return success();
  }

  LogicalResult lowerPOPNeg(NegOp op);
  LogicalResult lowerPOPFloor(FloorOp op);
  LogicalResult lowerPOPFloorDiv(FloorDivOp op);
  LogicalResult lowerPOPCeil(CeilOp op);
  LogicalResult lowerPOPTrunc(TruncOp op);
  LogicalResult lowerPOPRound(RoundOp op);
  LogicalResult lowerPOPAbs(AbsOp op);
  LogicalResult lowerPOPShr(ShrOp op);
  LogicalResult lowerPOPFMA(FMAOp op);
  LogicalResult lowerPOPCmp(CmpOp op);
  LogicalResult lowerPOPCast(CastOp op);
  LogicalResult lowerPOPSIMDSelect(SIMDSelectOp op);
  LogicalResult lowerPOPSIMDSplat(SIMDSplatOp op);
  LogicalResult lowerPOPSIMDInsertElement(SIMDInsertElementOp op);
  LogicalResult lowerPOPSIMDShuffle(SIMDShuffleOp op);
  LogicalResult lowerPOPSIMDExtractElement(SIMDExtractElementOp op);
  LogicalResult lowerPOPOffset(OffsetOp op);
  LogicalResult lowerPOPSelect(SelectOp op);
  LogicalResult lowerPOPStackAllocation(StackAllocationOp op);
  LogicalResult lowerPOPStackAllocLifetimeStart(StackAllocLifetimeStartOp op);
  LogicalResult lowerPOPStackAllocLifetimeEnd(StackAllocLifetimeEndOp op);
  LogicalResult lowerPOPArrayCreate(ArrayCreateOp op);
  LogicalResult lowerPOPArrayRepeat(ArrayRepeatOp op);
  LogicalResult lowerPOPArrayGet(ArrayGetOp op);
  LogicalResult lowerPOPArrayReplace(ArrayReplaceOp op);
  LogicalResult lowerPOPArrayGEP(ArrayGEPOp op);
  LogicalResult lowerPOPLoad(LoadOp op);
  LogicalResult lowerPOPStore(StoreOp op);
  LogicalResult lowerPOPMemcpy(MemcpyOp op);
  LogicalResult lowerPOPCastToBuiltin(CastToBuiltinOp op);
  LogicalResult lowerPOPCastFromBuiltin(CastFromBuiltinOp op);
  LogicalResult lowerPOPInlineAsm(InlineAsmOp op);
  LogicalResult lowerPOPAtomicCmpXchg(AtomicCmpXchgOp op);
  LogicalResult lowerPOPAtomicRMW(AtomicRMWOp op);
  LogicalResult lowerPOPFence(FenceOp op);
  LogicalResult lowerPOPStringAddress(StringAddressOp op);
  LogicalResult lowerPOPStringSize(StringSizeOp op);
  LogicalResult lowerPOPDTypeToUI8(DTypeToUI8 op);
  LogicalResult lowerPOPDTypeFromUI8(DTypeFromUI8 op);
  LogicalResult lowerPOPCallLLVMIntrinsic(CallLLVMIntrinsicOp op);
  LogicalResult lowerPOPPointerBitcast(PointerBitcastOp op);
  LogicalResult lowerPOPUnionBitcast(UnionBitcastOp op);
  LogicalResult lowerPOPUnionWrap(UnionWrapOp op);
  LogicalResult lowerPOPUnionUnwrap(UnionUnwrapOp op);
  LogicalResult lowerPOPSIMDReduceOr(SIMDReduceOrOp op);
  LogicalResult lowerPOPSIMDReduceAnd(SIMDReduceAndOp op);
  LogicalResult lowerPOPAnd(AndOp op);
  LogicalResult lowerPOPOr(OrOp op);
  LogicalResult lowerPOPXOr(XOrOp op);
  LogicalResult lowerPOPSIMDAnd(SIMDAndOp op);
  LogicalResult lowerPOPSIMDOr(SIMDOrOp op);
  LogicalResult lowerPOPSIMDXOr(SIMDXOrOp op);
  LogicalResult lowerPOPAdd(AddOp op);
  LogicalResult lowerPOPSub(SubOp op);
  LogicalResult lowerPOPMul(MulOp op);
  LogicalResult lowerPOPDiv(DivOp op);
  LogicalResult lowerPOPRem(RemOp op);
  LogicalResult lowerPOPMax(MaxOp op);
  LogicalResult lowerPOPMin(MinOp op);
  LogicalResult lowerPOPBitcast(BitcastOp op);
  LogicalResult lowerPOPShl(ShlOp op);
  LogicalResult lowerPOPPointerToIndex(PointerToIndexOp op);

  TargetInfoAttr targetInfo;
  const TargetLowering *targetLowering = nullptr;
  std::unique_ptr<POPToLLVMTypeConverter> typeConverter;
  POPLoweringBuilder *builder = nullptr;
  DenseMap<Value, Value> loweredValues;
  SmallVector<Value> loweredOperands;
  /// Values produced by lowering a `pop.simd.splat`, i.e. vector splats of one
  /// scalar. A runtime divisor drawn from this set needs one multiplier.
  llvm::DenseSet<Value> uniformVectors;
};
} // namespace

FailureOr<mlir::FunctionOpInterface> LowerPOPToLLVMPass::validateOperation() {
  auto func = dyn_cast<mlir::FunctionOpInterface>(getOperation());
  if (!func)
    return getOperation()->emitError(
        "lower-pop-to-llvm must be nested on a FunctionOpInterface");

  // Stack allocations cannot be lowered in the presence of CFGs.
  Operation *cfgOp = nullptr;
  func->walk([&cfgOp](Operation *op) {
    if (llvm::none_of(op->getRegions(), [](Region &region) {
          return region.getBlocks().size() > 1;
        }))
      return WalkResult::advance();
    cfgOp = op;
    return WalkResult::interrupt();
  });
  if (!cfgOp)
    return func;

  InFlightDiagnostic diag = cfgOp->emitError(
      "lower-pop-to-llvm cannot run on operations with CFG regions");
  diag.attachNote() << "try running it before lower-control-flow";
  return diag;
}

Value LowerPOPToLLVMPass::getLoweredValue(Value value) {
  auto it = loweredValues.find(value);
  if (it != loweredValues.end())
    return it->second;

  Type loweredType = typeConverter->convertType(value.getType());
  assert(loweredType && "operand type must be convertible to LLVM");
  if (loweredType == value.getType()) {
    loweredValues.try_emplace(value, value);
    return value;
  }

  OpBuilder::InsertionGuard guard(*builder);
  if (Operation *def = value.getDefiningOp())
    builder->setInsertionPointAfter(def);
  else
    builder->setInsertionPointToStart(cast<BlockArgument>(value).getOwner());
  Value lowered = mlir::UnrealizedConversionCastOp::create(
                      *builder, value.getLoc(), loweredType, value)
                      .getResult(0);
  loweredValues.try_emplace(value, lowered);
  return lowered;
}

ValueRange LowerPOPToLLVMPass::getLoweredOperands(Operation *op) {
  loweredOperands.clear();
  loweredOperands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands())
    loweredOperands.push_back(getLoweredValue(operand));
  return loweredOperands;
}

LogicalResult LowerPOPToLLVMPass::lowerBlock(Block &block) {
  SmallVector<Operation *> pending;
  SmallVector<Operation *> operations;
  for (Operation &op : llvm::reverse(block))
    pending.push_back(&op);

  // Build the preorder worklist iteratively because generated code can contain
  // region trees deep enough to exhaust the compiler's stack under ASAN.
  while (!pending.empty()) {
    Operation *op = pending.pop_back_val();
    operations.push_back(op);
    for (Region &region : llvm::reverse(op->getRegions()))
      for (Block &nestedBlock : llvm::reverse(region))
        for (Operation &nestedOp : llvm::reverse(nestedBlock))
          pending.push_back(&nestedOp);
  }

  for (Operation *op : operations) {
    if (!op->getBlock())
      continue;
    if (failed(lowerOperation(op)))
      return failure();
  }
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerOperation(Operation *op) {
  if (!isa_and_present<POPDialect>(op->getDialect()))
    return success();

  if (targetLowering &&
      isa<POP::MaxOp, POP::MinOp, POP::AtomicRMWOp, POP::AtomicCmpXchgOp,
          POP::CastOp, POP::FenceOp, CallLLVMIntrinsicOp>(op) &&
      targetLowering->isLoweredInGlobalPOPPass(op))
    return success();

  if (isa<GlobalAllocOp, GlobalConstantOp, ExternalCallOp,
          ExternPointerSymbolOp, AlignedAllocOp, AlignedFreeOp>(op))
    return success();

#define LOWER_POP_CASE(OpType, Method)                                         \
  if (auto typedOp = dyn_cast<OpType>(op)) {                                   \
    if (succeeded(Method(typedOp)))                                            \
      return success();                                                        \
    op->emitError("failed to legalize operation '")                            \
        << op->getName() << "' that was explicitly marked illegal";            \
    return failure();                                                          \
  }
  LOWER_POP_CASE(NegOp, lowerPOPNeg);
  LOWER_POP_CASE(FloorOp, lowerPOPFloor);
  LOWER_POP_CASE(FloorDivOp, lowerPOPFloorDiv);
  LOWER_POP_CASE(CeilOp, lowerPOPCeil);
  LOWER_POP_CASE(TruncOp, lowerPOPTrunc);
  LOWER_POP_CASE(RoundOp, lowerPOPRound);
  LOWER_POP_CASE(AbsOp, lowerPOPAbs);
  LOWER_POP_CASE(ShrOp, lowerPOPShr);
  LOWER_POP_CASE(FMAOp, lowerPOPFMA);
  LOWER_POP_CASE(CmpOp, lowerPOPCmp);
  LOWER_POP_CASE(CastOp, lowerPOPCast);
  LOWER_POP_CASE(SIMDSelectOp, lowerPOPSIMDSelect);
  LOWER_POP_CASE(SIMDSplatOp, lowerPOPSIMDSplat);
  LOWER_POP_CASE(SIMDInsertElementOp, lowerPOPSIMDInsertElement);
  LOWER_POP_CASE(SIMDShuffleOp, lowerPOPSIMDShuffle);
  LOWER_POP_CASE(SIMDExtractElementOp, lowerPOPSIMDExtractElement);
  LOWER_POP_CASE(OffsetOp, lowerPOPOffset);
  LOWER_POP_CASE(SelectOp, lowerPOPSelect);
  LOWER_POP_CASE(StackAllocationOp, lowerPOPStackAllocation);
  LOWER_POP_CASE(StackAllocLifetimeStartOp, lowerPOPStackAllocLifetimeStart);
  LOWER_POP_CASE(StackAllocLifetimeEndOp, lowerPOPStackAllocLifetimeEnd);
  LOWER_POP_CASE(ArrayCreateOp, lowerPOPArrayCreate);
  LOWER_POP_CASE(ArrayRepeatOp, lowerPOPArrayRepeat);
  LOWER_POP_CASE(ArrayGetOp, lowerPOPArrayGet);
  LOWER_POP_CASE(ArrayReplaceOp, lowerPOPArrayReplace);
  LOWER_POP_CASE(ArrayGEPOp, lowerPOPArrayGEP);
  LOWER_POP_CASE(LoadOp, lowerPOPLoad);
  LOWER_POP_CASE(StoreOp, lowerPOPStore);
  LOWER_POP_CASE(MemcpyOp, lowerPOPMemcpy);
  LOWER_POP_CASE(CastToBuiltinOp, lowerPOPCastToBuiltin);
  LOWER_POP_CASE(CastFromBuiltinOp, lowerPOPCastFromBuiltin);
  LOWER_POP_CASE(InlineAsmOp, lowerPOPInlineAsm);
  LOWER_POP_CASE(AtomicCmpXchgOp, lowerPOPAtomicCmpXchg);
  LOWER_POP_CASE(AtomicRMWOp, lowerPOPAtomicRMW);
  LOWER_POP_CASE(FenceOp, lowerPOPFence);
  LOWER_POP_CASE(StringAddressOp, lowerPOPStringAddress);
  LOWER_POP_CASE(StringSizeOp, lowerPOPStringSize);
  LOWER_POP_CASE(DTypeToUI8, lowerPOPDTypeToUI8);
  LOWER_POP_CASE(DTypeFromUI8, lowerPOPDTypeFromUI8);
  LOWER_POP_CASE(CallLLVMIntrinsicOp, lowerPOPCallLLVMIntrinsic);
  LOWER_POP_CASE(PointerBitcastOp, lowerPOPPointerBitcast);
  LOWER_POP_CASE(UnionBitcastOp, lowerPOPUnionBitcast);
  LOWER_POP_CASE(UnionWrapOp, lowerPOPUnionWrap);
  LOWER_POP_CASE(UnionUnwrapOp, lowerPOPUnionUnwrap);
  LOWER_POP_CASE(SIMDReduceOrOp, lowerPOPSIMDReduceOr);
  LOWER_POP_CASE(SIMDReduceAndOp, lowerPOPSIMDReduceAnd);
  LOWER_POP_CASE(AndOp, lowerPOPAnd);
  LOWER_POP_CASE(OrOp, lowerPOPOr);
  LOWER_POP_CASE(XOrOp, lowerPOPXOr);
  LOWER_POP_CASE(SIMDAndOp, lowerPOPSIMDAnd);
  LOWER_POP_CASE(SIMDOrOp, lowerPOPSIMDOr);
  LOWER_POP_CASE(SIMDXOrOp, lowerPOPSIMDXOr);
  LOWER_POP_CASE(AddOp, lowerPOPAdd);
  LOWER_POP_CASE(SubOp, lowerPOPSub);
  LOWER_POP_CASE(MulOp, lowerPOPMul);
  LOWER_POP_CASE(DivOp, lowerPOPDiv);
  LOWER_POP_CASE(RemOp, lowerPOPRem);
  LOWER_POP_CASE(MaxOp, lowerPOPMax);
  LOWER_POP_CASE(MinOp, lowerPOPMin);
  LOWER_POP_CASE(BitcastOp, lowerPOPBitcast);
  LOWER_POP_CASE(ShlOp, lowerPOPShl);
  LOWER_POP_CASE(PointerToIndexOp, lowerPOPPointerToIndex);
#undef LOWER_POP_CASE

  op->emitError("failed to legalize operation '")
      << op->getName() << "' that was explicitly marked illegal";
  return failure();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPNeg(NegOp op) {
  NegOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  KGENDType dtype = *op.getType().getResolvedDType();
  if (!dtype.isInt() && !dtype.isIndex() && !dtype.isUIndex()) {
    builder->replaceOpWithNewOp<LLVM::FNegOp>(op, adaptor.getOperand(),
                                              fastmathFlagsOrDefault(op));
    return success();
  }
  Type type = adaptor.getOperand().getType();
  Value zero;
  if (auto vec = dyn_cast<VectorType>(type)) {
    auto intType = dyn_cast<IntegerType>(vec.getElementType());
    if (!intType)
      return op.emitError("could not handle integer type");
    zero = LLVM::ConstantOp::create(
        *builder, op.getLoc(),
        DenseIntElementsAttr::get(vec, APInt::getZero(intType.getWidth())));
  } else {
    zero = LLVM::ConstantOp::create(*builder, op.getLoc(), type, 0);
  }
  builder->replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getOperand());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPFloor(FloorOp op) {
  FloorOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (!op.getType().getResolvedDType()->isFloat()) {
    builder->replaceOp(op, adaptor.getOperand());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, typeConverter->convertType(op.getType()),
      builder->getStringAttr("llvm.floor"),
      SmallVector<Value>{adaptor.getOperand()});
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPCeil(CeilOp op) {
  CeilOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (!op.getType().getResolvedDType()->isFloat()) {
    builder->replaceOp(op, adaptor.getOperand());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, typeConverter->convertType(op.getType()),
      builder->getStringAttr("llvm.ceil"),
      SmallVector<Value>{adaptor.getOperand()});
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPTrunc(TruncOp op) {
  TruncOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (!op.getType().getResolvedDType()->isFloat()) {
    builder->replaceOp(op, adaptor.getOperand());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, typeConverter->convertType(op.getType()),
      builder->getStringAttr("llvm.trunc"),
      SmallVector<Value>{adaptor.getOperand()});
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPShr(ShrOp op) {
  ShrOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  KGENDType dtype = *op.getType().getResolvedDType();
  if (dtype.isSInt() || dtype.isIndex())
    builder->replaceOpWithNewOp<LLVM::AShrOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
  else
    builder->replaceOpWithNewOp<LLVM::LShrOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPFMA(FMAOp op) {
  FMAOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  KGENDType dtype = *op.getType().getResolvedDType();
  if (dtype.isInt() || dtype.isIndex() || dtype.isUIndex()) {
    Value lhs = LLVM::MulOp::create(*builder, op.getLoc(), adaptor.getA(),
                                    adaptor.getB());
    builder->replaceOpWithNewOp<LLVM::AddOp>(op, lhs, adaptor.getC());
  } else {
    builder->replaceOpWithNewOp<LLVM::FMAOp>(
        op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
        convertFastmathFlags(op.getFastmathFlags(), *builder));
  }
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPCmp(CmpOp op) {
  CmpOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  auto getICmpPredicate = [](CmpPredicate pred, bool isSigned) {
    switch (pred) {
    case CmpPredicate::EQ:
      return LLVM::ICmpPredicate::eq;
    case CmpPredicate::NE:
      return LLVM::ICmpPredicate::ne;
    case CmpPredicate::LT:
      return isSigned ? LLVM::ICmpPredicate::slt : LLVM::ICmpPredicate::ult;
    case CmpPredicate::GT:
      return isSigned ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::ugt;
    case CmpPredicate::LE:
      return isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule;
    case CmpPredicate::GE:
      return isSigned ? LLVM::ICmpPredicate::sge : LLVM::ICmpPredicate::uge;
    }
    llvm_unreachable("unknown predicate");
  };
  auto getFCmpPredicate = [](CmpPredicate pred) {
    switch (pred) {
    case CmpPredicate::EQ:
      return LLVM::FCmpPredicate::oeq;
    case CmpPredicate::NE:
      return LLVM::FCmpPredicate::une;
    case CmpPredicate::LT:
      return LLVM::FCmpPredicate::olt;
    case CmpPredicate::GT:
      return LLVM::FCmpPredicate::ogt;
    case CmpPredicate::LE:
      return LLVM::FCmpPredicate::ole;
    case CmpPredicate::GE:
      return LLVM::FCmpPredicate::oge;
    }
    llvm_unreachable("unknown predicate");
  };

  KGENDType dtype = *op.getLhs().getType().getResolvedDType();
  if (dtype.isBool() || dtype.isInt() || dtype.isIndex() || dtype.isUIndex() ||
      dtype.isAddress()) {
    builder->replaceOpWithNewOp<LLVM::ICmpOp>(
        op, getICmpPredicate(op.getPred(), dtype.isSInt()), adaptor.getLhs(),
        adaptor.getRhs());
    return success();
  }
  assert(dtype.isFloat());
  Type i1Type = builder->getI1Type();
  if (auto simd = dyn_cast<SIMDType>(op.getLhs().getType())) {
    uint64_t size = *simd.getResolvedSize();
    if (size != 1)
      i1Type = VectorType::get(size, i1Type);
  }
  builder->replaceOpWithNewOp<LLVM::FCmpOp>(
      op, i1Type, getFCmpPredicate(op.getPred()), adaptor.getLhs(),
      adaptor.getRhs(), fastmathFlagsOrDefault(op));
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDSelect(SIMDSelectOp op) {
  SIMDSelectOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::SelectOp>(
      op, adaptor.getCondition(), adaptor.getTrueValue(),
      adaptor.getFalseValue(), fastmathFlagsOrDefault(op));
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDSplat(SIMDSplatOp op) {
  SIMDSplatOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (op.getType().isScalar()) {
    builder->replaceOp(op, adaptor.getScalar());
    return success();
  }
  int64_t size = *op.getType().getResolvedSize();
  Value undef = LLVM::UndefOp::create(*builder, op.getLoc(),
                                      typeConverter->convertType(op.getType()));
  Value zero = LLVM::ConstantOp::create(*builder, op.getLoc(),
                                        builder->getI32IntegerAttr(0));
  Value vector = LLVM::InsertElementOp::create(*builder, op.getLoc(), undef,
                                               adaptor.getScalar(), zero);
  auto splat = builder->replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
      op, vector, undef, SmallVector<int32_t>(size, 0));
  uniformVectors.insert(splat.getResult());
  return success();
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPSIMDInsertElement(SIMDInsertElementOp op) {
  SIMDInsertElementOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (op.getVector().getType().isScalar()) {
    builder->replaceOp(op, adaptor.getValue());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::InsertElementOp>(
      op, typeConverter->convertType(op.getType()), adaptor.getOperands());
  return success();
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPSIMDExtractElement(SIMDExtractElementOp op) {
  SIMDExtractElementOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  if (op.getVector().getType().isScalar()) {
    builder->replaceOp(op, adaptor.getVector());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::ExtractElementOp>(
      op, typeConverter->convertType(op.getType()), adaptor.getVector(),
      adaptor.getPosition());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDShuffle(SIMDShuffleOp op) {
  SIMDShuffleOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  auto mask = cast<POP::ArrayAttr>(adaptor.getMask());
  SmallVector<int32_t> maskValues;
  for (TypedAttr maskElement : mask.getValues())
    maskValues.push_back(cast<IntegerAttr>(maskElement).getInt());
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  if (*op.getLhs().getType().getResolvedSize() != 1) {
    builder->replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
        op, lhs, rhs, builder->getDenseI32ArrayAttr(maskValues));
    return success();
  }
  KGENDType dtype = *op.getType().getResolvedDType();
  auto llvmVecType = VectorType::get(
      mask.getValues().size(),
      *getMLIRTypeForDType(op.getContext(), dtype,
                           typeConverter->getIndexTypeBitwidth()));
  Value result = LLVM::UndefOp::create(*builder, op.getLoc(), llvmVecType);
  for (auto [idx, maskElement] : llvm::enumerate(maskValues)) {
    Value pos = LLVM::ConstantOp::create(*builder, op.getLoc(),
                                         builder->getI32IntegerAttr(idx));
    result = LLVM::InsertElementOp::create(*builder, op.getLoc(), result,
                                           maskElement == 0 ? lhs : rhs, pos);
  }
  builder->replaceOp(op, result);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPOffset(OffsetOp op) {
  OffsetOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type elementType =
      typeConverter->convertType(op.getPtr().getType().getElementType());
  unsigned addrSpace = op.getPtr().getType().getAddrSpaceOrZero();
  Type intPtrType =
      IntegerType::get(&typeConverter->getContext(),
                       typeConverter->getPointerBitwidth(addrSpace));
  size_t pointerWidth = intPtrType.getIntOrFloatBitWidth();
  size_t indexWidth = adaptor.getIndex().getType().getIntOrFloatBitWidth();
  Value offset;
  if (pointerWidth == indexWidth)
    offset = adaptor.getIndex();
  else if (pointerWidth < indexWidth)
    offset = builder->createOrFold<LLVM::TruncOp>(op.getLoc(), intPtrType,
                                                  adaptor.getIndex());
  else
    offset = builder->createOrFold<LLVM::SExtOp>(op.getLoc(), intPtrType,
                                                 adaptor.getIndex());
  builder->replaceOpWithNewOp<LLVM::GEPOp>(
      op, adaptor.getPtr().getType(), elementType, adaptor.getPtr(),
      ValueRange{offset}, LLVM::GEPNoWrapFlags::inbounds);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSelect(SelectOp op) {
  SelectOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::SelectOp>(
      op, adaptor.getCondition(), adaptor.getTrueValue(),
      adaptor.getFalseValue(), fastmathFlagsOrDefault(op));
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPFloorDiv(FloorDivOp op) {
  FloorDivOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Location loc = op.getLoc();
  Type type = typeConverter->convertType(op.getType());
  KGENDType dtype = *op.getType().getResolvedDType();
  if (dtype.isFloat()) {
    Value fdiv = LLVM::FDivOp::create(*builder, loc, type, adaptor.getLhs(),
                                      adaptor.getRhs());
    builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, builder->getStringAttr("llvm.floor"),
        SmallVector<Value>{fdiv});
    return success();
  }
  if (dtype.isUInt()) {
    builder->replaceOpWithNewOp<LLVM::UDivOp>(
        op, type, SmallVector<Value>{adaptor.getLhs(), adaptor.getRhs()});
    return success();
  }

  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  bool canWiden = intType && (2 * intType.getWidth() <= 64 || !targetLowering ||
                              targetLowering->supportsI128());
  if (intType && intType.getWidth() >= 8 && intType.getWidth() <= 64 &&
      llvm::isPowerOf2_32(intType.getWidth()) && canWiden) {
    std::optional<APInt> divisor =
        getSplatConstantDivisor(op, adaptor.getRhs(), intType.getWidth());
    if (divisor) {
      if (std::optional<FloorDivMagic> magic = computeFloorDivMagic(*divisor)) {
        builder->replaceOp(op,
                           emitFloorDivByConstant(*builder, loc, type,
                                                  adaptor.getLhs(), *magic));
        return success();
      }
    } else if (intType.getWidth() <= 32 &&
               (!targetLowering || targetLowering->isHostTarget()) &&
               isUniformRuntimeDivisor(op, adaptor.getRhs(), uniformVectors)) {
      // A runtime divisor that is uniform across the operation takes the
      // multiply-high path too. Like the sequence it replaces, it leaves
      // d == 0 and INT_MIN / -1 undefined, but they produce a wrapped or
      // poison result instead of a trap. The product stays 64-bit, so the only
      // width limit is the 33-bit dividend the round-up bias needs; 64-bit
      // dividends would need a 128-bit multiplier and a 192-bit product, so
      // they keep the generic sequence. It is host-only: the multiplier comes
      // from an `f64` reciprocal, which Metal cannot compile, and on the GPU
      // the extra live values push Moka's KDA and top-k kernels into spilling
      // at their per-thread register budget.
      builder->replaceOp(op, emitFloorDivByRuntimeDivisor(
                                 *builder, loc, type, adaptor.getLhs(),
                                 adaptor.getRhs(), intType.getWidth()));
      return success();
    }
  }

  Value sdiv = LLVM::SDivOp::create(*builder, loc, type, adaptor.getLhs(),
                                    adaptor.getRhs());
  Value mul = LLVM::MulOp::create(*builder, loc, type, sdiv, adaptor.getRhs());
  Value icmp = LLVM::ICmpOp::create(*builder, loc, LLVM::ICmpPredicate::eq, mul,
                                    adaptor.getLhs());
  Value xorOp = LLVM::XOrOp::create(*builder, loc, type, adaptor.getLhs(),
                                    adaptor.getRhs());

  Value signBitShift;
  Value zero;
  if (auto vecType = dyn_cast<VectorType>(type)) {
    uint64_t size = *op.getType().getResolvedSize();
    unsigned width = vecType.getElementType().getIntOrFloatBitWidth();
    auto zeroAttrs = cast<TypedAttr>(IntArrayElementsAttr::get(
        vecType, SmallVector<APInt>{size, APInt::getZero(width)}));
    auto shiftAttrs = cast<TypedAttr>(IntArrayElementsAttr::get(
        vecType, SmallVector<APInt>{size, APInt(width, width - 1)}));
    zero = LLVM::ConstantOp::create(*builder, loc, zeroAttrs);
    signBitShift = LLVM::ConstantOp::create(*builder, loc, shiftAttrs);
  } else {
    assert(type.isIntOrFloat() && "Unexpected type in floordiv");
    zero = LLVM::ConstantOp::create(*builder, loc,
                                    builder->getIntegerAttr(type, 0));
    signBitShift = LLVM::ConstantOp::create(
        *builder, loc,
        builder->getIntegerAttr(type, type.getIntOrFloatBitWidth() - 1));
  }
  Value ashr = LLVM::AShrOp::create(*builder, loc, type, xorOp, signBitShift);
  Value sel = LLVM::SelectOp::create(*builder, loc, type, icmp, zero, ashr);
  builder->replaceOpWithNewOp<LLVM::AddOp>(op, type,
                                           SmallVector<Value>{sdiv, sel});
  return success();
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPStackAllocation(StackAllocationOp op) {
  StackAllocationOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  PointerType ptrType = cast<PointerType>(op.getType());
  Type elementType = typeConverter->convertType(ptrType.getElementType());
  if (!elementType)
    return op.emitError("could not lower pointer element type");
  std::optional<int64_t> typeAllocSize = DataLayoutInterface::getTypeAllocSize(
      targetInfo, ptrType.getElementType());
  if (!typeAllocSize)
    return op.emitError("could not get size of variadic element");
  std::optional<int64_t> typeAlign = DataLayoutInterface::getTypeABIAlign(
      targetInfo, ptrType.getElementType());
  if (!typeAlign)
    return op.emitError("could not get alignment of element type");
  int64_t explicitAlign =
      op.getAlignment() ? cast<IntegerAttr>(*op.getAlignment()).getInt() : 0;
  int64_t alignment = std::max(*typeAlign, explicitAlign);

  StoreOp theStore;
  SmallVector<LoadOp> loads;
  bool allSimple = true;
  for (Operation *user : op->getUsers()) {
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
      continue;
    if (auto storeOp = dyn_cast<StoreOp>(user);
        storeOp && storeOp.getOperand(1) == op.getResult() && !theStore &&
        !storeOp.mightBeVolatile()) {
      theStore = storeOp;
      continue;
    }
    if (auto loadOp = dyn_cast<LoadOp>(user);
        loadOp && !loadOp.mightBeVolatile()) {
      loads.push_back(loadOp);
      continue;
    }
    allSimple = false;
    break;
  }
  if (allSimple && theStore) {
    mlir::DominanceInfo domInfo;
    bool dominates = llvm::all_of(
        loads, [&](LoadOp load) { return domInfo.dominates(theStore, load); });
    if (dominates) {
      for (LoadOp load : loads) {
        load.replaceAllUsesWith(theStore.getOperand(0));
        builder->deleteLater(load);
      }
    }
  }

  Value alloca = materializeLLVMAlloca(
      *builder, targetInfo, elementType,
      cast<IntegerAttr>(op.getCount()).getInt(), op, *typeAllocSize, alignment);
  builder->replaceOp(op, alloca);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPStackAllocLifetimeStart(
    StackAllocLifetimeStartOp op) {
  StackAllocLifetimeStartOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  return lowerLifetimeMarker<LLVM::LifetimeStartOp>(op, adaptor.getValues(),
                                                    targetInfo, *builder);
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPStackAllocLifetimeEnd(StackAllocLifetimeEndOp op) {
  StackAllocLifetimeEndOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  return lowerLifetimeMarker<LLVM::LifetimeEndOp>(op, adaptor.getValues(),
                                                  targetInfo, *builder);
}

LogicalResult LowerPOPToLLVMPass::lowerPOPArrayCreate(ArrayCreateOp op) {
  ArrayCreateOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type type = typeConverter->convertType(op.getType());
  if (!type)
    return op.emitError("failed to convert array type");
  Value array = LLVM::UndefOp::create(*builder, op.getLoc(), type);
  for (auto [idx, value] : llvm::enumerate(adaptor.getOperands()))
    array =
        LLVM::InsertValueOp::create(*builder, op.getLoc(), array, value, idx);
  builder->replaceOp(op, array);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPArrayRepeat(ArrayRepeatOp op) {
  ArrayRepeatOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type type = typeConverter->convertType(op.getType());
  if (!type)
    return op.emitError("failed to convert array type");
  Value array = LLVM::UndefOp::create(*builder, op.getLoc(), type);
  for (unsigned i = 0, size = *op.getType().getResolvedSize(); i < size;) {
    for (Value value : adaptor.getOperands()) {
      if (i == size)
        break;
      array =
          LLVM::InsertValueOp::create(*builder, op.getLoc(), array, value, i++);
    }
  }
  builder->replaceOp(op, array);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPArrayGet(ArrayGetOp op) {
  ArrayGetOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::ExtractValueOp>(
      op, adaptor.getArray(), cast<IntegerAttr>(op.getIndex()).getInt());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPArrayReplace(ArrayReplaceOp op) {
  ArrayReplaceOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::InsertValueOp>(
      op, adaptor.getArray(), adaptor.getValue(),
      cast<IntegerAttr>(op.getIndex()).getInt());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPArrayGEP(ArrayGEPOp op) {
  ArrayGEPOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type ptrType = typeConverter->convertType(op.getType());
  Type elementType =
      typeConverter->convertType(op.getArray().getType().getElementType());
  if (!ptrType)
    return op.emitError("failed to convert result type");
  builder->replaceOpWithNewOp<LLVM::GEPOp>(
      op, ptrType, elementType, adaptor.getArray(),
      ArrayRef<LLVM::GEPArg>{0, adaptor.getIndex()},
      LLVM::GEPNoWrapFlags::inbounds);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPLoad(LoadOp op) {
  LoadOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  auto ptrType = cast<PointerType>(op.getPtr().getType());
  Type elementType = typeConverter->convertType(ptrType.getElementType());
  unsigned alignment =
      getAlignment(typeConverter.get(), ptrType, adaptor.getAlignmentAttr());
  builder->replaceOpWithNewOp<LLVM::LoadOp>(
      op, elementType, adaptor.getPtr(), alignment,
      cast<BoolAttr>(adaptor.getIsVolatile()).getValue(),
      cast<BoolAttr>(adaptor.getIsNonTemporal()).getValue(),
      cast<BoolAttr>(adaptor.getIsInvariant()).getValue(),
      /*isInvariantGroup=*/false, /*isNoundef=*/false,
      getAtomicOrdering(adaptor.getOrdering()),
      adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                             : StringRef());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPStore(StoreOp op) {
  StoreOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  auto ptrType = cast<PointerType>(op.getPtr().getType());
  unsigned alignment =
      getAlignment(typeConverter.get(), ptrType, adaptor.getAlignmentAttr());
  builder->replaceOpWithNewOp<LLVM::StoreOp>(
      op, adaptor.getArg(), adaptor.getPtr(), alignment,
      cast<BoolAttr>(adaptor.getIsVolatile()).getValue(),
      cast<BoolAttr>(adaptor.getIsNonTemporal()).getValue(),
      /*isInvariantGroup=*/false, getAtomicOrdering(adaptor.getOrdering()),
      adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                             : StringRef());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPMemcpy(MemcpyOp op) {
  MemcpyOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      cast<BoolAttr>(adaptor.getIsVolatile()).getValue());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPCastToBuiltin(CastToBuiltinOp op) {
  CastToBuiltinOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->replaceOp(op, adaptor.getInput());
  return success();
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPCastFromBuiltin(CastFromBuiltinOp op) {
  CastFromBuiltinOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->replaceOp(op, adaptor.getInput());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPCast(CastOp op) {
  SmallVector<Value> operands = getLoweredOperands(op);
  builder->setInsertionPoint(op);
  if (targetLowering && succeeded(targetLowering->lowerFunctionLocalPOPToLLVM(
                            op, operands, *typeConverter, *builder)))
    return success();
  CastOpAdaptor adaptor(operands, op);
  KGENDType inDType = *op.getInput().getType().getResolvedDType();
  KGENDType outDType = *op.getOutput().getType().getResolvedDType();
  if ((inDType.isFloat() && inDType.getWidthInBits() <= 8) ||
      (outDType.isFloat() && outDType.getWidthInBits() <= 8))
    return op.emitError() << "illegal conversion";

  auto getByteCount = [&](KGENDType dtype) -> int64_t {
    if (dtype.isIndex() || dtype.isUIndex())
      return typeConverter->getIndexTypeBitwidth() / CHAR_BIT;
    return dtype.getSizeInBytes();
  };
  int64_t inByteCount = getByteCount(inDType);
  int64_t outByteCount = getByteCount(outDType);
  StringRef opName;
  if (inDType.isBool() || inDType.isInt() || inDType.isIndex() ||
      inDType.isUIndex()) {
    if (outDType.isBool() || outDType.isInt() || outDType.isIndex() ||
        outDType.isUIndex()) {
      if (outByteCount > inByteCount || inDType.isBool())
        opName = inDType.isSInt() ? LLVM::SExtOp::getOperationName()
                                  : LLVM::ZExtOp::getOperationName();
      else if (outByteCount < inByteCount || outDType.isBool())
        opName = LLVM::TruncOp::getOperationName();
    } else {
      opName = inDType.isSInt() ? LLVM::SIToFPOp::getOperationName()
                                : LLVM::UIToFPOp::getOperationName();
    }
  } else if (outDType.isBool() || outDType.isInt() || outDType.isIndex() ||
             outDType.isUIndex()) {
    opName = outDType.isSInt() ? LLVM::FPToSIOp::getOperationName()
                               : LLVM::FPToUIOp::getOperationName();
  } else if (outByteCount > inByteCount) {
    opName = LLVM::FPExtOp::getOperationName();
  } else if (outByteCount < inByteCount) {
    if (isFPTyLoweredAsInt(getEquivalentFloatType(op.getContext(), outDType))) {
      return emitError(op.getLoc(),
                       Twine("casts between " + inDType.getAsString() +
                             " and " + outDType.getAsString() +
                             " unsupported"));
    }
    opName = LLVM::FPTruncOp::getOperationName();
  } else if (outDType != inDType) {
    return op.emitError("casts between 'bf16' and 'f16' unsupported");
  }
  if (opName.empty()) {
    builder->replaceOp(op, adaptor.getInput());
    return success();
  }
  OperationState state(op.getLoc(), opName);
  state.addOperands(adaptor.getInput());
  state.addTypes(typeConverter->convertType(op.getOutput().getType()));
  builder->replaceOp(op, builder->create(state));
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPRound(RoundOp op) {
  SmallVector<Value> operands = getLoweredOperands(op);
  builder->setInsertionPoint(op);
  if (targetLowering && succeeded(targetLowering->lowerFunctionLocalPOPToLLVM(
                            op, operands, *typeConverter, *builder)))
    return success();
  RoundOpAdaptor adaptor(operands, op);
  if (!op.getType().getResolvedDType()->isFloat()) {
    builder->replaceOp(op, adaptor.getOperand());
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, typeConverter->convertType(op.getType()),
      builder->getStringAttr("llvm.roundeven"),
      SmallVector<Value>{adaptor.getOperand()});
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPAbs(AbsOp op) {
  SmallVector<Value> operands = getLoweredOperands(op);
  builder->setInsertionPoint(op);
  if (targetLowering && succeeded(targetLowering->lowerFunctionLocalPOPToLLVM(
                            op, operands, *typeConverter, *builder)))
    return success();
  AbsOpAdaptor adaptor(operands, op);
  KGENDType dtype = *op.getType().getResolvedDType();
  if (dtype.isUInt() || dtype.isBool()) {
    builder->replaceOp(op, adaptor.getOperand());
    return success();
  }
  Type type = typeConverter->convertType(op.getType());
  if (!dtype.isFloat()) {
    Value isIntMinPoison = LLVM::ConstantOp::create(
        *builder, op.getLoc(), builder->getIntegerType(1), false);
    builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, builder->getStringAttr("llvm.abs"),
        SmallVector<Value>{adaptor.getOperand(), isIntMinPoison});
    return success();
  }
  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, type, builder->getStringAttr("llvm.fabs"),
      SmallVector<Value>{adaptor.getOperand()});
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPInlineAsm(InlineAsmOp op) {
  InlineAsmOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  SmallVector<Type, 1> types;
  if (op.getNumResults()) {
    types.push_back(typeConverter->packFunctionResults(op->getResultTypes()));
    if (!types.back())
      return failure();
  }
  auto asmOp = LLVM::InlineAsmOp::create(
      *builder, op.getLoc(), types.empty() ? Type() : types.front(),
      expandOperands(*builder, op.getLoc(), adaptor.getOperands(),
                     op.getOperands().getTypes(), *typeConverter),
      cast<StringAttr>(adaptor.getAssembly()),
      cast<StringAttr>(adaptor.getConstraints()),
      cast<BoolAttr>(adaptor.getHasSideEffects()).getValue(),
      cast<BoolAttr>(adaptor.getIsStackAligned()).getValue(),
      LLVM::TailCallKind::None, /*convergent=*/false,
      LLVM::AsmDialectAttr::get(op->getContext(), LLVM::AsmDialect::AD_ATT),
      adaptor.getOperandAttrsAttr());
  if (op.getNumResults() <= 1) {
    builder->replaceOp(op, asmOp);
    return success();
  }
  SmallVector<Value> results;
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    results.push_back(LLVM::ExtractValueOp::create(*builder, op.getLoc(),
                                                   asmOp.getResult(0), i));
  builder->replaceOp(op, results);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPAtomicCmpXchg(AtomicCmpXchgOp op) {
  AtomicCmpXchgOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  unsigned alignment = 0;
  if (auto attr = adaptor.getAlignment()) {
    alignment = cast<IntegerAttr>(*attr).getInt();
  } else if (Operation *def = adaptor.getPtr().getDefiningOp();
             def && def->hasAttr("alignment")) {
    alignment = cast<IntegerAttr>(def->getAttr("alignment")).getInt();
  }
  builder->replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
      op, adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal(),
      getAtomicOrdering(op.getSuccessOrdering()),
      getAtomicOrdering(op.getFailureOrdering()),
      adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                             : StringRef(),
      alignment, adaptor.getWeak().has_value());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPAtomicRMW(AtomicRMWOp op) {
  AtomicRMWOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  KGENDType dtype = *cast<SIMDType>(op.getType()).getResolvedDType();
  auto getBinOp = [](KGENDType dtype, AtomicBinOp binOp) {
    switch (binOp) {
    case AtomicBinOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    case AtomicBinOp::ADD:
      return dtype.isFloat() ? LLVM::AtomicBinOp::fadd : LLVM::AtomicBinOp::add;
    case AtomicBinOp::SUB:
      return dtype.isFloat() ? LLVM::AtomicBinOp::fsub : LLVM::AtomicBinOp::sub;
    case AtomicBinOp::AND:
      return LLVM::AtomicBinOp::_and;
    case AtomicBinOp::NAND:
      return LLVM::AtomicBinOp::nand;
    case AtomicBinOp::OR:
      return LLVM::AtomicBinOp::_or;
    case AtomicBinOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case AtomicBinOp::MAX:
      if (dtype.isSInt())
        return LLVM::AtomicBinOp::max;
      if (dtype.isUInt())
        return LLVM::AtomicBinOp::umax;
      if (dtype.isFloat())
        return LLVM::AtomicBinOp::fmax;
      break;
    case AtomicBinOp::MIN:
      if (dtype.isSInt())
        return LLVM::AtomicBinOp::min;
      if (dtype.isUInt())
        return LLVM::AtomicBinOp::umin;
      if (dtype.isFloat())
        return LLVM::AtomicBinOp::fmin;
      break;
    }
    llvm_unreachable("unknown atomic operation");
  };
  builder->replaceOpWithNewOp<LLVM::AtomicRMWOp>(
      op, getBinOp(dtype, adaptor.getBinOp()), adaptor.getPtr(),
      adaptor.getVal(), getAtomicOrdering(op.getOrdering()),
      adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                             : StringRef());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPFence(FenceOp op) {
  FenceOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::FenceOp>(
      op, getAtomicOrdering(adaptor.getOrdering()), adaptor.getSyncscopeAttr());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPStringAddress(StringAddressOp op) {
  StringAddressOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Value address = LLVM::ExtractValueOp::create(
      *builder, op.getLoc(), adaptor.getOperands().front(), 0);
  builder->replaceOp(op, address);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPStringSize(StringSizeOp op) {
  StringSizeOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Value size = LLVM::ExtractValueOp::create(*builder, op.getLoc(),
                                            adaptor.getOperands().front(), 1);
  builder->replaceOp(op, size);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPDTypeToUI8(DTypeToUI8 op) {
  DTypeToUI8Adaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  builder->replaceOpWithNewOp<LLVM::BitcastOp>(
      op, typeConverter->convertType(op.getType()), adaptor.getDType());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPDTypeFromUI8(DTypeFromUI8 op) {
  DTypeFromUI8Adaptor adaptor(getLoweredOperands(op), op);
  builder->replaceOp(op, adaptor.getValue());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPPointerBitcast(PointerBitcastOp op) {
  PointerBitcastOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type resultType = typeConverter->convertType(op.getType());
  if (!resultType)
    return failure();
  Value src = adaptor.getOperands()[0];
  if (src.getType() != resultType)
    builder->replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultType, src);
  else
    builder->replaceOp(op, src);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPUnionBitcast(UnionBitcastOp op) {
  UnionBitcastOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->replaceOp(op, adaptor.getValue());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPUnionWrap(UnionWrapOp op) {
  UnionWrapOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  auto variantType = dyn_cast_or_null<LLVM::LLVMStructType>(
      typeConverter->convertType(op.getType()));
  if (!variantType)
    return failure();
  FailureOr<Value> ptr = materializeLLVMUnionAlloca(
      typeConverter->getTarget(), *builder, op, op.getType(), variantType);
  if (failed(ptr))
    return failure();
  LLVM::StoreOp::create(*builder, op.getLoc(), adaptor.getValue(), *ptr);
  builder->replaceOpWithNewOp<LLVM::LoadOp>(op, variantType, *ptr);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPUnionUnwrap(UnionUnwrapOp op) {
  UnionUnwrapOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  Type valueType = typeConverter->convertType(op.getType());
  if (!valueType)
    return failure();
  auto contentType = cast<LLVM::LLVMStructType>(adaptor.getValue().getType());
  if (contentType.getBody().empty()) {
    builder->replaceOpWithNewOp<LLVM::UndefOp>(op, contentType);
    return success();
  }
  assert((contentType.getBody().size() == 1 ||
          contentType.getBody().size() == 2) &&
         "must have 1 or 2 fields for union struct");
  FailureOr<Value> ptr =
      materializeLLVMUnionAlloca(typeConverter->getTarget(), *builder, op,
                                 op.getValue().getType(), contentType);
  if (failed(ptr))
    return failure();
  LLVM::StoreOp::create(*builder, op.getLoc(), adaptor.getValue(), *ptr);
  builder->replaceOpWithNewOp<LLVM::LoadOp>(op, valueType, *ptr);
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDReduceOr(SIMDReduceOrOp op) {
  SIMDReduceOrOpAdaptor adaptor(getLoweredOperands(op), op);
  if (op.getVector().getType().isScalar()) {
    builder->replaceOp(op, adaptor.getVector());
    return success();
  }
  builder->setInsertionPoint(op);
  Type resultType = typeConverter->convertType(op.getType());
  builder->replaceOpWithNewOp<LLVM::vector_reduce_or>(op, resultType,
                                                      adaptor.getVector());
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDReduceAnd(SIMDReduceAndOp op) {
  SIMDReduceAndOpAdaptor adaptor(getLoweredOperands(op), op);
  if (op.getVector().getType().isScalar()) {
    builder->replaceOp(op, adaptor.getVector());
    return success();
  }
  builder->setInsertionPoint(op);
  Type resultType = typeConverter->convertType(op.getType());
  builder->replaceOpWithNewOp<LLVM::vector_reduce_and>(op, resultType,
                                                       adaptor.getVector());
  return success();
}

LogicalResult
LowerPOPToLLVMPass::lowerPOPCallLLVMIntrinsic(CallLLVMIntrinsicOp op) {
  CallLLVMIntrinsicOpAdaptor adaptor(getLoweredOperands(op), op);
  builder->setInsertionPoint(op);
  SmallVector<Type> types;
  if (failed(typeConverter->convertTypes(op.getResultTypes(), types)))
    return failure();
  StringRef intrinsicName = cast<StringAttr>(op.getIntrin());
  auto operands = expandOperands(*builder, op.getLoc(), adaptor.getOperands(),
                                 op.getOperands().getTypes(), *typeConverter);

  auto getAlignment = [&](Value alignment,
                          StringRef name) -> FailureOr<IntegerAttr> {
    auto constant = alignment.getDefiningOp<LLVM::ConstantOp>();
    if (!constant)
      return op.emitError() << name
                            << " alignment must be a constant, got runtime "
                               "value";
    auto attr = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attr)
      return op.emitError() << name << " alignment must be an integer";
    return attr;
  };

  if (intrinsicName == "llvm.masked.load") {
    if (operands.size() != 4)
      return op.emitError("llvm.masked.load expects 4 operands "
                          "(ptr, alignment, mask, passthrough), got ")
             << operands.size();
    FailureOr<IntegerAttr> alignment =
        getAlignment(operands[1], "llvm.masked.load");
    if (failed(alignment))
      return failure();
    builder->replaceOpWithNewOp<LLVM::MaskedLoadOp>(
        op, types[0], operands[0], operands[2], operands[3],
        alignment->getValue().getZExtValue());
    return success();
  }

  if (intrinsicName == "llvm.masked.store") {
    if (operands.size() != 4)
      return op.emitError("llvm.masked.store expects 4 operands "
                          "(value, ptr, alignment, mask), got ")
             << operands.size();
    FailureOr<IntegerAttr> alignment =
        getAlignment(operands[2], "llvm.masked.store");
    if (failed(alignment))
      return failure();
    builder->replaceOpWithNewOp<LLVM::MaskedStoreOp>(
        op, operands[0], operands[1], operands[3],
        alignment->getValue().getZExtValue());
    return success();
  }

  auto createMaskedIntrinsic = [&](ValueRange callOperands,
                                   unsigned pointerIndex,
                                   IntegerAttr alignment) {
    auto emptyAttrs = builder->getDictionaryAttr({});
    auto pointerAttrs =
        builder->getDictionaryAttr({builder->getNamedAttr("align", alignment)});
    SmallVector<Attribute> attrs(callOperands.size(), emptyAttrs);
    attrs[pointerIndex] = pointerAttrs;
    return LLVM::CallIntrinsicOp::create(
        *builder, op.getLoc(), types, cast<StringAttr>(op.getIntrin()),
        callOperands, convertFastmathFlags(op.getFastmathFlags(), *builder),
        /*op_bundle_operands=*/ArrayRef<ValueRange>{},
        /*op_bundle_tags=*/mlir::ArrayAttr{},
        /*arg_attrs=*/builder->getArrayAttr(attrs),
        /*res_attrs=*/mlir::ArrayAttr{});
  };

  if (intrinsicName == "llvm.masked.gather") {
    if (operands.size() != 4)
      return op.emitError("llvm.masked.gather expects 4 operands "
                          "(ptr_vec, alignment, mask, passthrough), got ")
             << operands.size();
    FailureOr<IntegerAttr> alignment =
        getAlignment(operands[1], "llvm.masked.gather");
    if (failed(alignment))
      return failure();
    Value callOperands[] = {operands[0], operands[2], operands[3]};
    builder->replaceOp(op, createMaskedIntrinsic(callOperands, 0, *alignment));
    return success();
  }

  if (intrinsicName == "llvm.masked.scatter") {
    if (operands.size() != 4)
      return op.emitError("llvm.masked.scatter expects 4 operands "
                          "(value, ptr_vec, alignment, mask), got ")
             << operands.size();
    FailureOr<IntegerAttr> alignment =
        getAlignment(operands[2], "llvm.masked.scatter");
    if (failed(alignment))
      return failure();
    Value callOperands[] = {operands[0], operands[1], operands[3]};
    builder->replaceOp(op, createMaskedIntrinsic(callOperands, 1, *alignment));
    return success();
  }

  builder->replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
      op, types, cast<StringAttr>(op.getIntrin()), operands,
      convertFastmathFlags(op.getFastmathFlags(), *builder));
  return success();
}

LogicalResult LowerPOPToLLVMPass::lowerPOPAnd(AndOp op) {
  return lowerOneToOne<AndOp, LLVM::AndOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPOr(OrOp op) {
  return lowerOneToOne<OrOp, LLVM::OrOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPXOr(XOrOp op) {
  return lowerOneToOne<XOrOp, LLVM::XOrOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDAnd(SIMDAndOp op) {
  return lowerOneToOne<SIMDAndOp, LLVM::AndOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDOr(SIMDOrOp op) {
  return lowerOneToOne<SIMDOrOp, LLVM::OrOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPSIMDXOr(SIMDXOrOp op) {
  return lowerOneToOne<SIMDXOrOp, LLVM::XOrOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPAdd(AddOp op) {
  return lowerFloatOrInt<AddOp, LLVM::FAddOp, LLVM::AddOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPSub(SubOp op) {
  return lowerFloatOrInt<SubOp, LLVM::FSubOp, LLVM::SubOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPMul(MulOp op) {
  return lowerFloatOrInt<MulOp, LLVM::FMulOp, LLVM::MulOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPDiv(DivOp op) {
  return lowerFloatOrInt<DivOp, LLVM::FDivOp, LLVM::SDivOp, LLVM::UDivOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPRem(RemOp op) {
  return lowerFloatOrInt<RemOp, LLVM::FRemOp, LLVM::SRemOp, LLVM::URemOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPMax(MaxOp op) {
  return lowerFloatOrInt<MaxOp, LLVM::MaxNumOp, LLVM::SMaxOp, LLVM::UMaxOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPMin(MinOp op) {
  return lowerFloatOrInt<MinOp, LLVM::MinNumOp, LLVM::SMinOp, LLVM::UMinOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPBitcast(BitcastOp op) {
  return lowerOneToOne<BitcastOp, LLVM::BitcastOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPShl(ShlOp op) {
  return lowerOneToOne<ShlOp, LLVM::ShlOp>(op);
}
LogicalResult LowerPOPToLLVMPass::lowerPOPPointerToIndex(PointerToIndexOp op) {
  return lowerOneToOne<PointerToIndexOp, LLVM::PtrToIntOp>(op);
}

void LowerPOPToLLVMPass::runOnOperation() {
  loweredValues.clear();
  loweredOperands.clear();
  uniformVectors.clear();
  builder = nullptr;
  typeConverter.reset();
  targetLowering = nullptr;
  targetInfo = {};

  FailureOr<mlir::FunctionOpInterface> func = validateOperation();
  if (failed(func))
    return signalPassFailure();

  // If the function body is empty, return.
  if (func->getFunctionBody().empty())
    return;

  // Set LLVM lowering options.
  targetInfo = lookupTargetInfo(*func);
  if (!targetInfo) {
    mlir::emitError(func->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(targetInfo.getTriple());
  targetLowering = loweringOr.isError() ? nullptr : *loweringOr;

  typeConverter = std::make_unique<POPToLLVMTypeConverter>(targetInfo);
  POPLoweringBuilder loweringBuilder(&getContext(), loweredValues);
  builder = &loweringBuilder;

  if (failed(lowerBlock(func->getFunctionBody().front())))
    return signalPassFailure();

  // The index dialect is lowered with the upstream conversion patterns, which
  // require the dialect conversion driver; run it rollback-free over the
  // residue.
  mlir::ConversionTarget indexTarget(getContext());
  indexTarget.addIllegalDialect<mlir::index::IndexDialect>();
  indexTarget.addLegalDialect<LLVM::LLVMDialect>();
  indexTarget.addLegalOp<mlir::UnrealizedConversionCastOp>();
  mlir::RewritePatternSet indexPatterns(&getContext());
  mlir::index::populateIndexToLLVMConversionPatterns(*typeConverter,
                                                     indexPatterns);
  mlir::ConversionConfig config;
  config.allowPatternRollback = false;
  config.foldingMode = mlir::DialectConversionFoldingMode::Never;
  if (failed(mlir::applyPartialConversion(*func, indexTarget,
                                          std::move(indexPatterns), config)))
    return signalPassFailure();

  // Collapse cast pairs left across the two conversions.
  SmallVector<mlir::UnrealizedConversionCastOp> casts;
  func->walk([&](mlir::UnrealizedConversionCastOp castOp) {
    casts.push_back(castOp);
  });
  mlir::reconcileUnrealizedCasts(casts);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPAlignedAlloc
//===----------------------------------------------------------------------===//

static constexpr llvm::StringLiteral kAllocFamilyName =
    "kgen_aligned_allocator";

/// This pattern will generate the aligned alloc function with the appropriate
/// attributes to teach LLVM about the allocator. This would enable LLVM, for
/// example, to promote heap-to-stack among other optimizations. This enables
/// the aligned alloc function to receive similar treatment to `malloc`.
struct ConvertPOPAlignedAlloc : public ConvertSymbolOpToLLVM<AlignedAllocOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  static constexpr llvm::StringLiteral kAllocFnName =
      "KGEN_CompilerRT_AlignedAlloc";

  LogicalResult matchAndRewrite(AlignedAllocOp op,
                                AlignedAllocOpAdaptor adaptor,
                                DirectRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(kAllocFnName);
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      const mlir::LLVMTypeConverter &tc = *getTypeConverter();
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();

      // The function signature is `ptr(index, index)`.
      auto allocFnSig =
          LLVM::LLVMFunctionType::get(LLVM::LLVMPointerType::get(getContext()),
                                      {tc.getIndexType(), tc.getIndexType()});

      SmallVector<Attribute> passthrough;
      func = LLVM::LLVMFuncOp::create(b, mlir::UnknownLoc::get(getContext()),
                                      kAllocFnName, allocFnSig);

      // `noalias` result.
      func.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                         b.getUnitAttr());
      // `allocalign` on the first argument.
      func.setArgAttr(0, LLVM::LLVMDialect::getAllocAlignAttrName(),
                      b.getUnitAttr());

      // `allockind("alloc,aligned,uninitialized")` enum encoding.
      // FIXME: The encoding of integer attributes is a string?!
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allockind"),
           b.getStringAttr(Twine(static_cast<int64_t>(
               llvm::AllocFnKind::Alloc | llvm::AllocFnKind::Aligned |
               llvm::AllocFnKind::Uninitialized)))}));

      // `allocsize(1)` with `-1` in lower 32 bits.
      // FIXME: The encoding of integer attributes is a string?!
      // FIXME: `packAllocSizeArgs` is not an exposed function.
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allocsize"),
           b.getStringAttr(Twine(uint32_t(-1) | (uint64_t(1) << 32)))}));
      // `"alloc-family"="kgen_alloc"`.
      passthrough.push_back(
          b.getArrayAttr({b.getStringAttr("alloc-family"),
                          b.getStringAttr(kAllocFamilyName)}));

      func.setPassthroughAttr(attachTargetPassthroughAttrs(
          b, getTypeConverter()->getTarget(), b.getArrayAttr(passthrough)));
      symtab.insert(func);
    }

    LLVM::CallOp call =
        createLLVMCall(b, op.getLoc(), func, adaptor.getOperands());
    b.replaceOpWithNewOp<LLVM::BitcastOp>(op, convertType(op.getType()),
                                          call.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAlignedFree
//===----------------------------------------------------------------------===//

/// This pattern will generate the aligned free function with the appropriate
/// attributes to teach LLVM about the allocator. This would enable LLVM, for
/// example, to promote heap-to-stack among other optimizations. This enables
/// the aligned free function to receive similar treatment to `free`.
struct ConvertPOPAlignedFree : public ConvertSymbolOpToLLVM<AlignedFreeOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  static constexpr llvm::StringLiteral kFreeFnName =
      "KGEN_CompilerRT_AlignedFree";

  LogicalResult matchAndRewrite(AlignedFreeOp op, AlignedFreeOpAdaptor adaptor,
                                DirectRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(kFreeFnName);
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();

      // The function signature is `void(ptr)`.
      auto freeFnSig =
          LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()),
                                      LLVM::LLVMPointerType::get(getContext()));

      SmallVector<Attribute> passthrough;
      func = LLVM::LLVMFuncOp::create(b, mlir::UnknownLoc::get(getContext()),
                                      kFreeFnName, freeFnSig);

      // `allocptr` on first argument.
      func.setArgAttr(0, LLVM::LLVMDialect::getAllocatedPointerAttrName(),
                      b.getUnitAttr());

      // `allockind("alloc,aligned,uninitialized")` enum encoding.
      // FIXME: The encoding of integer attributes is a string?!
      passthrough.push_back(b.getArrayAttr(
          {b.getStringAttr("allockind"),
           b.getStringAttr(
               Twine(static_cast<uint64_t>(llvm::AllocFnKind::Free)))}));

      // `"alloc-family"="kgen_alloc"`.
      passthrough.push_back(
          b.getArrayAttr({b.getStringAttr("alloc-family"),
                          b.getStringAttr(kAllocFamilyName)}));

      func.setPassthroughAttr(attachTargetPassthroughAttrs(
          b, getTypeConverter()->getTarget(), b.getArrayAttr(passthrough)));
      symtab.insert(func);
    }

    Value ptr = LLVM::BitcastOp::create(
        b, op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
        adaptor.getPtr());
    LLVM::CallOp call = createLLVMCall(b, op.getLoc(), func, ptr);
    b.replaceOp(op, call);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalAlloc
//===----------------------------------------------------------------------===//

struct ConvertPOPGlobalAlloc : public ConvertSymbolOpToLLVM<GlobalAllocOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  // Tracks the next available local index per parent function to give
  // deterministic names.
  // mark `mutable` here because it is mutated in a `const` method
  // matchAndRewrite. This pattern must remain an ModuleOp pattern
  // instead of FunctionOp pattern so that this pattern is not being running in
  // parallel while this mutable is not thread-safe.
  mutable llvm::DenseMap<mlir::Operation *, unsigned> gpuSharedAllocIdx;

  LogicalResult matchAndRewrite(GlobalAllocOp op, GlobalAllocOpAdaptor adaptor,
                                DirectRewriter &b) const override {
    // Set the alignment if specified. Otherwise use the natural alignment.
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto elementType = typeConverter->convertType(kgenPtrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), kgenPtrType, op.getAlignmentAttr());

    // Set the address space if specified.
    unsigned addrSpace = op.getType().getAddrSpaceOrZero();

    // (HACK) Add a postfix to the name here so that we can identify
    // this type of variables in the llvm module.
    // This is a workaround to LLVM MLIR lowering doesn't allow
    // GlobalValues to have passthrough metadata.

    std::string name = cast<StringAttr>(adaptor.getName()).str();
    if (op.getMemoryType() == POP::GlobalAllocAddressSpace::GPU_SHARED) {
      // Scope the name to the enclosing function so every GPU_SHARED global
      // gets a deterministic unique name per kernel.
      if (auto parentFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>()) {
        unsigned localIdx = gpuSharedAllocIdx[parentFuncOp.getOperation()]++;
        name = parentFuncOp.getSymName().str() + "." + name + "_" +
               std::to_string(localIdx);
      }
      name += "._gpu_shared_mem";
    }

    // Create the global.
    b.clearInsertionPoint();
    auto arrayType = LLVM::LLVMArrayType::get(
        elementType, cast<IntegerAttr>(op.getCount()).getInt());
    auto global = LLVM::GlobalOp::create(
        b, op.getLoc(), arrayType,
        /*isConstant=*/false, LLVM::Linkage::Internal, name,
        /*value=*/Attribute(), alignment, addrSpace);

    // If an initializer is present, emit an initializer region.
    // The verifier guarantees count == 1 when an initializer is set.
    if (auto init = op.getInitializer()) {
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder ib(op.getLoc(), op.getContext());
      ib.setInsertionPointToStart(global.getBody());
      ErrorOr<Value> value =
          convertParameterToLLVM(ib, *getTypeConverter(), /*imc=*/nullptr,
                                 /*scope=*/nullptr, *init);
      if (value.isError()) {
        ib.emitError(value.getError());
        return failure();
      }
      // Wrap the scalar value in the array type.
      Value array = LLVM::UndefOp::create(ib, arrayType);
      array = LLVM::InsertValueOp::create(ib, array, value.get(),
                                          ArrayRef<int64_t>{0});
      LLVM::ReturnOp::create(ib, array);
    }

    symtab.insert(global);

    // Replace the alloc op with an `addressof`.
    b.setInsertionPoint(op);
    auto opaquePtrType = LLVM::LLVMPointerType::get(getContext(), addrSpace);
    auto ptr = LLVM::AddressOfOp::create(b, op.getLoc(), global);
    b.replaceOpWithNewOp<LLVM::BitcastOp>(
        op,
        LLVM::LLVMPointerType::get(opaquePtrType.getContext(),
                                   opaquePtrType.getAddressSpace()),
        ptr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPNoAliasPointerCast
//===----------------------------------------------------------------------===//

static LLVM::LLVMFuncOp getOrCreateNoAliasCastIntrinsic(SymbolTable &symtab,
                                                        mlir::RewriterBase &b) {
  constexpr llvm::StringLiteral fnName = "__kgen_noalias_cast";
  auto name = b.getStringAttr(fnName);
  if (auto func = symtab.lookup<LLVM::LLVMFuncOp>(name))
    return func;

  // Create the function. It has the form `noalias ptr (ptr noalias returned)`.
  // Since the returned pointer can have arbitrary effects on it, we can't
  // annotate the argument with any.
  OpBuilder::InsertionGuard guard(b);
  b.clearInsertionPoint();
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());
  auto func = LLVM::LLVMFuncOp::create(
      b, UnknownLoc::get(b.getContext()), name,
      LLVM::LLVMFunctionType::get(ptrType, ptrType), LLVM::Linkage::Internal);
  symtab.insert(func);

  // Set the `noalias` attributes.
  func.setArgAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(), b.getUnitAttr());
  func.setResultAttr(0, LLVM::LLVMDialect::getNoAliasAttrName(),
                     b.getUnitAttr());

  constexpr llvm::StringLiteral funcAttrs[] = {
      "alwaysinline", "mustprogress", "nofree",    "norecurse",
      "nosync",       "nounwind",     "willreturn"};
  SmallVector<Attribute> attrs;
  for (StringRef attr : funcAttrs)
    attrs.push_back(b.getStringAttr(attr));
  // memory(none)
  attrs.push_back(
      b.getArrayAttr({b.getStringAttr("memory"), b.getStringAttr("0")}));
  func.setPassthroughAttr(b.getArrayAttr(attrs));

  // Populate the body.
  Block *body = b.createBlock(&func.getBody());
  LLVM::ReturnOp::create(b, func.getLoc(),
                         body->addArgument(ptrType, func.getLoc()));

  return func;
}

struct ConvertPOPNoAliasPointerCast
    : ConvertSymbolOpToLLVM<NoAliasPointerCastOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(NoAliasPointerCastOp op,
                                NoAliasPointerCastOpAdaptor adaptor,
                                DirectRewriter &b) const override {
    LLVM::LLVMFuncOp func = getOrCreateNoAliasCastIntrinsic(symtab, b);
    LLVM::CallOp call = createLLVMCall(b, op.getLoc(), func, adaptor.getIn());
    b.replaceOp(op, call);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalConstant
//===----------------------------------------------------------------------===//

/// Lower a global constant. Unique the constant value.
class ConvertPOPGlobalConstant
    : public ConvertPOPToLLVMPattern<GlobalConstantOp> {
public:
  ConvertPOPGlobalConstant(
      SymbolTable &symtab,
      DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> &constants,
      mlir::LLVMTypeConverter &typeConverter, InterpreterMemoryConverter &imc)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        constants(constants), imc(imc) {}

  LogicalResult matchAndRewrite(GlobalConstantOp op,
                                GlobalConstantOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto opaquePtrType = LLVM::LLVMPointerType::get(getContext());
    Type elementType = convertType(kgenPtrType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "failed to convert constant result type");

    // Unique the constant.
    auto [it, inserted] = constants.try_emplace(
        std::make_pair(op.getValue(), op.getAlignmentAttr()), nullptr);
    if (inserted) {
      // If the constant doesn't exist, create it and insert it in the module.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();

      LLVM::GlobalOp global = LLVM::GlobalOp::create(
          rewriter, op.getLoc(), elementType, true, LLVM::Linkage::Internal,
          "global_constant", Attribute(),
          getAlignment(getTypeConverter(), kgenPtrType,
                       adaptor.getAlignmentAttr()));
      // Emit the constant using an initializer region.
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
      b.setInsertionPointToStart(global.getBody());
      InterpreterMemoryConverter::MaterializationScope scope =
          imc.createScope();
      ErrorOr<Value> value = convertParameterToLLVM(
          b, *getTypeConverter(), /*imc=*/&imc, &scope, op.getValue());
      if (value.isError()) {
        b.emitError(value.getError());
        return failure();
      }
      LLVM::ReturnOp::create(b, value.get());

      // Insert the global into the module.
      symtab.insert(it->second = global);
    }

    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, opaquePtrType, FlatSymbolRefAttr::get(it->second.getSymNameAttr()));
    return success();
  }

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// Uniqued constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> &constants;
  InterpreterMemoryConverter &imc;
};

//===----------------------------------------------------------------------===//
// ConvertExternPointerSymbol
//===----------------------------------------------------------------------===//

/// Lower external pointer symbol, this replaces the pointer with an external
/// global value.
struct ConvertExternPointerSymbol
    : public ConvertSymbolOpToLLVM<ExternPointerSymbolOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(ExternPointerSymbolOp op,
                                ExternPointerSymbolOpAdaptor adaptor,
                                DirectRewriter &b) const override {
    int64_t addressSpace = op.getResSymbol().getType().getAddrSpaceOrZero();
    Type resType = convertType(op.getResSymbol().getType().getElementType());
    unsigned align = getAlignment(
        getTypeConverter(), op.getResSymbol().getType(), op.getAlignmentAttr());

    b.clearInsertionPoint();
    auto global = LLVM::GlobalOp::create(
        b, op.getLoc(), resType, /*constant=*/false, LLVM::Linkage::External,
        cast<StringAttr>(op.getName()), /*value=*/nullptr, align, addressSpace,
        /*dso_local=*/true);
    symtab.insert(global);

    b.setInsertionPoint(op);
    b.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, LLVM::LLVMPointerType::get(getContext(), addressSpace),
        FlatSymbolRefAttr::get(global.getSymNameAttr()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerGlobalPOPToLLVMPass
//===----------------------------------------------------------------------===//

struct LowerGlobalPOPToLLVMPass
    : public KGEN::impl::LowerGlobalPOPToLLVMBase<LowerGlobalPOPToLLVMPass> {
  using LowerGlobalPOPToLLVMBase::LowerGlobalPOPToLLVMBase;

  void runOnOperation() override;
};

} // namespace

void LowerGlobalPOPToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(targetInfo.getTriple());
  const TargetLowering *lowering = loweringOr.isError() ? nullptr : *loweringOr;

  POPToLLVMTypeConverter typeConverter(targetInfo);
  InterpreterMemoryConverter imc(symtab, typeConverter);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Convert external calls.
  patterns.insert<ConvertPOPGlobalAlloc, ConvertExternPointerSymbol,
                  ConvertPOPAlignedAlloc, ConvertPOPAlignedFree,
                  ConvertPOPNoAliasPointerCast>(typeConverter, symtab);
  if (lowering)
    lowering->populateLowerGlobalPOPToLLVMPatterns(patterns, typeConverter,
                                                   symtab, targetInfo);
  populateLowerPOPExternalCallPatterns(patterns, typeConverter, symtab);

  // Convert global constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> constants;
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter,
                                            imc);

  // Ops the target lowers in this pass are exactly the ones the per-func pass
  // skipped; leave the others to it.
  auto skipOp = [&](Operation *op) {
    return isa<POP::MaxOp, POP::MinOp, POP::AtomicRMWOp, POP::AtomicCmpXchgOp,
               POP::CastOp, POP::FenceOp, CallLLVMIntrinsicOp>(op) &&
           !(lowering && lowering->isLoweredInGlobalPOPPass(op));
  };
  auto mustConvert = [&](Operation *op) {
    return isa<GlobalAllocOp, GlobalConstantOp, ExternalCallOp,
               ExternPointerSymbolOp, POP::MaxOp, POP::MinOp, POP::AtomicRMWOp,
               POP::AtomicCmpXchgOp, POP::CastOp, POP::FenceOp,
               CallLLVMIntrinsicOp>(op);
  };
  if (failed(M::applyDirectConversion(
          theModule, &typeConverter, std::move(patterns), skipOp, mustConvert)))
    return signalPassFailure();
}
