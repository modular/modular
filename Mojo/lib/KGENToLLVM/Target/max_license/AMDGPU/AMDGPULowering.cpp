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
//
// MLIR-lowering policy for AMD GPU (amdgcn) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"
#include "Target/max_license/AMDGPU/AMDGPUTraits.h"

#include "KGENToLLVM/LLVMLoweringUtils.h"
#include "KGENToLLVM/Target/max_license/GPUCommon.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/TargetParser/Triple.h"

#include <string>
#include <utility>

using namespace mlir;

namespace M::KGEN {
namespace {

/// Returns whether `target` is an AMDGPU target whose arch is one of
/// `allowedGPUs`.
bool isAMDGPU(TargetInfoAttr target,
              llvm::ArrayRef<llvm::StringRef> allowedGPUs) {
  if (!target || !target.getTriple().isAMDGPU())
    return false;
  return llvm::is_contained(allowedGPUs, target.getArch());
}

//===----------------------------------------------------------------------===//
// AMDGPUCastToLLVM
//===----------------------------------------------------------------------===//

/// Lowers `pop.cast`s that have a faster AMDGPU-specific instruction sequence
/// (f32->bf16 and f32<->f8 conversions). Registered with a higher benefit than
/// the generic `ConvertPOPCast`, so it runs first; when the cast isn't one it
/// handles, it fails and the generic pattern takes over.
struct AMDGPUCastToLLVM : public ConvertPOPToLLVMPattern<POP::CastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(POP::CastOp op, POP::CastOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    TargetInfoAttr target = getTypeConverter()->getTarget();
    if (!target || !target.getTriple().isAMDGPU())
      return failure();

    return convertToTargetSpecificCast(rewriter, op, adaptor.getInput());
  }

private:
  Type getConvertedScalarType(SIMDType simd) const {
    return convertType(
        SIMDType::get(simd.getContext(), /*size=*/1, *simd.getResolvedDType()));
  }

  Type convertKGENDType(MLIRContext *ctx, KGENDType dtype) const {
    return getConvertedScalarType(SIMDType::get(ctx, /*size=*/1, dtype));
  }

  /// Fast conversion of f32 to bf16 on AMDGPU that is not supported by LLVM and
  /// has different handling of NaNs.
  /// The generated sequence has been moved from stdlib (see reference
  /// implementation in PR##54249) to compiler in order:
  ///   - remove boilerplate code from stdlib
  ///   - reduce compile time as stdlib's code won't be parsed and will simply
  ///   be represented by `pop.cast`
  LogicalResult
  convertF32ToBF16OnAMDGPU(DirectRewriter &rewriter, POP::CastOp op,
                           Value value, APFloat::Semantics fromFloatSemantics,
                           APFloat::Semantics toFloatSemantics) const {
    assert(getTypeConverter()->getTarget().getTriple().isAMDGPU() &&
           "fast lowering of f32 to bf16 is only supported on AMDGPU");
    assert(POP::FastmathFlagsInterface(op).isFast() &&
           "fast lowering of f32 to bf16 requires `fast` on the cast");

    Location loc = op.getLoc();
    auto simd = cast<SIMDType>(op.getInput().getType());
    const uint64_t size = *simd.getResolvedSize();

    // This implementation is a faster version for fp32 to bf16 type conversion
    // It is from CK:
    // https://github.com/cgmillette/composable_kernel/commit/b8addae29
    // It uses less VGPR and less number of instructions compared to the
    // previous implementation
    Value roundedBias = createConstant<uint32_t>(
        rewriter, loc, std::numeric_limits<int16_t>::max());
    Type vecI64 = convertKGENDType(rewriter.getContext(), KGENDType::ui64);
    Type bf16Type = getConvertedScalarType(cast<SIMDType>(op.getType()));
    Type f32Type = getConvertedScalarType(simd);

    // Helper function to convert a single F32 value to BF16
    auto convertSingleValue = [&](Value value) {
      Value unorderedMask =
          createInlineAsm(rewriter, loc, "v_cmp_u_f32 $0, $1, $1", "=s,v",
                          vecI64, {value})
              .getResult(0);

      Type vecI32 = convertType(
          SIMDType::get(rewriter.getContext(), /*size=*/1, KGENDType::ui32));
      Value lsb = createInlineAsm(rewriter, loc, "v_bfe_u32 $0, $1, 16, 1",
                                  "=v,v", vecI32, {value})
                      .getResult(0);

      Value roundedVal =
          createInlineAsm(rewriter, loc, "v_add3_u32 $0, $1, $2, $3",
                          "=v,v,v,s", vecI32, {value, lsb, roundedBias})
              .getResult(0);

      Value nan = createConstant(
          rewriter, loc,
          APFloat::getNaN(APFloat::EnumToSemantics(fromFloatSemantics)));

      Value floatBits =
          createInlineAsm(rewriter, loc, "v_cndmask_b32 $0, $1, $2, $3",
                          "=v,v,v,s", vecI32, {roundedVal, nan, unorderedMask})
              .getResult(0);

      Value mantissaDiff = createConstant<uint32_t>(
          rewriter, loc,
          APFloat::semanticsPrecision(
              APFloat::EnumToSemantics(fromFloatSemantics)) -
              APFloat::semanticsPrecision(
                  APFloat::EnumToSemantics(toFloatSemantics)));
      Value shifted =
          LLVM::LShrOp::create(rewriter, loc, floatBits, mantissaDiff);

      shifted = LLVM::TruncOp::create(rewriter, loc,
                                      rewriter.getIntegerType(16), shifted);

      return LLVM::BitcastOp::create(rewriter, loc, bf16Type, shifted);
    };

    Value res;
    if (size > 1) {
      res = LLVM::UndefOp::create(rewriter, op.getLoc(),
                                  VectorType::get(size, bf16Type));
      for (uint32_t i = 0; i < size; ++i) {
        Value element = extractElement(rewriter, loc, f32Type, value, i);
        Value converted = convertSingleValue(element);
        res = LLVM::InsertElementOp::create(
            rewriter, loc, res, converted,
            createConstant<uint32_t>(rewriter, loc, i));
      }
    } else {
      res = convertSingleValue(value);
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  Value getClampedFloatValue(DirectRewriter &rewriter, Location loc,
                             Value value, APFloat::Semantics floatSemantics,
                             size_t size, Type outType) const {
    APFloat m = APFloat::getLargest(APFloat::EnumToSemantics(floatSemantics));
    Value max, min;
    if (size == 1) {
      max = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF32FloatAttr(m.convertToFloat()));
      min = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF32FloatAttr(-m.convertToFloat()));

    } else {
      VectorType vecType = VectorType::get(size, rewriter.getF32Type());
      SmallVector<APFloat> maxValues{size, APFloat(m.convertToFloat())};
      SmallVector<APFloat> minValues{size, APFloat(-m.convertToFloat())};
      auto maxAttrs =
          cast<TypedAttr>(FloatArrayElementsAttr::get(vecType, maxValues));
      auto minAttrs =
          cast<TypedAttr>(FloatArrayElementsAttr::get(vecType, minValues));
      max = LLVM::ConstantOp::create(rewriter, loc, maxAttrs);
      min = LLVM::ConstantOp::create(rewriter, loc, minAttrs);
    }

    Value clamped = LLVM::MaxNumOp::create(rewriter, loc, value, min,
                                           LLVM::FastmathFlags::none);
    clamped = LLVM::MinNumOp::create(rewriter, loc, clamped, max,
                                     LLVM::FastmathFlags::none);
    return clamped;
  }

  Value getIsNaN(DirectRewriter &rewriter, Location loc, Value value,
                 APFloat::Semantics floatSemantics, size_t size) const {
    // Using FCmp with predicate of FCMP_UNO to check nan
    // FCMP_UNO 	True if unordered: isnan(X) | isnan(Y)
    // https://llvm.org/doxygen/classllvm_1_1CmpInst.html#a2be3583dac92a031fa1458d4d992c78b
    Type type = convertType(SIMDType::get(rewriter.getContext(), size,
                                          KGENDType(DType(DType::ui1))));
    Value result = LLVM::FCmpOp::create(
        rewriter, loc, type, mlir::LLVM::FCmpPredicate::uno, value, value);
    return result;
  }

  /// Helper function to convert scalar of F32 to F8 (F8E4M3FNUZ or F8E5M2FNUZ)
  /// on AMDGPU without replacing the operation
  Value
  convertF32ToF8OnAMDGPUHelper(DirectRewriter &rewriter, POP::CastOp op,
                               Value value, Type outType,
                               APFloat::Semantics fromFloatSemantics,
                               APFloat::Semantics toFloatSemantics) const {
    assert(getTypeConverter()->getTarget().getTriple().isAMDGPU() &&
           "fast lowering of f32 to f8 is only supported on AMDGPU");

    Location loc = op.getLoc();
    auto simd = cast<SIMDType>(op.getInput().getType());
    const uint64_t size = *simd.getResolvedSize();

    Type f8Type = outType;
    Type i32Type = rewriter.getIntegerType(32);
    Type ui8Type = rewriter.getIntegerType(8);
    Type ui16Type = rewriter.getIntegerType(16);
    Type f32Type = rewriter.getF32Type();

    Value clamped = getClampedFloatValue(rewriter, loc, value, toFloatSemantics,
                                         size, f8Type);
    Value isNaN = getIsNaN(rewriter, loc, value, fromFloatSemantics, size);
    Value sel = LLVM::SelectOp::create(rewriter, loc, isNaN, value, clamped,
                                       LLVM::FastmathFlags::none);

    StringRef intrinsicStr = [&]() -> StringRef {
      if (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FNUZ ||
          toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN)
        return "llvm.amdgcn.cvt.pk.fp8.f32";

      if (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2FNUZ ||
          toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2)
        return "llvm.amdgcn.cvt.pk.bf8.f32";

      llvm_unreachable("conversion format not supported on AMDGPU");
    }();

    assert(llvm::isPowerOf2_64(size) && "SIMD size must be a power of 2");

    Value result;

    if (size == 1) {
      SmallVector<Value> operands{
          sel, sel,
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(0)),
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                   rewriter.getBoolAttr(false))};

      result = LLVM::CallIntrinsicOp::create(
                   rewriter, loc, i32Type, rewriter.getStringAttr(intrinsicStr),
                   operands)
                   .getResult(0);
      result = LLVM::TruncOp::create(rewriter, loc, ui8Type, result);
    } else if (size == 2) {
      SmallVector<Value> operands{
          extractElement(rewriter, loc, f32Type, sel, 0),
          extractElement(rewriter, loc, f32Type, sel, 1),
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(0)),
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                   rewriter.getBoolAttr(false))};

      result = LLVM::CallIntrinsicOp::create(
                   rewriter, loc, i32Type, rewriter.getStringAttr(intrinsicStr),
                   operands)
                   .getResult(0);

      result = LLVM::TruncOp::create(rewriter, loc, ui16Type, result);
      result = LLVM::BitcastOp::create(rewriter, loc, f8Type, result);
    } else {
      result = LLVM::UndefOp::create(
          rewriter, op.getLoc(),
          VectorType::get(size / 4, rewriter.getIntegerType(32)));

      for (uint64_t i = 0; i < size; i += 4) {
        SmallVector<Value> operands{
            extractElement(rewriter, loc, f32Type, sel, i),
            extractElement(rewriter, loc, f32Type, sel, i + 1),
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(0)),
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                     rewriter.getBoolAttr(false))};

        Value loWord = LLVM::CallIntrinsicOp::create(
                           rewriter, loc, i32Type,
                           rewriter.getStringAttr(intrinsicStr), operands)
                           .getResult(0);

        operands[0] = extractElement(rewriter, loc, f32Type, sel, i + 2);
        operands[1] = extractElement(rewriter, loc, f32Type, sel, i + 3);
        operands[2] = loWord;
        operands[3] = LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));

        Value hiLoWord = LLVM::CallIntrinsicOp::create(
                             rewriter, loc, i32Type,
                             rewriter.getStringAttr(intrinsicStr), operands)
                             .getResult(0);

        result = LLVM::InsertElementOp::create(
            rewriter, loc, result, hiLoWord,
            createConstant<uint32_t>(rewriter, loc, i / 4));
      }

      result = LLVM::BitcastOp::create(rewriter, loc, f8Type, result);
    }
    return result;
  }

  LogicalResult
  convertF32ToF8OnAMDGPU(DirectRewriter &rewriter, POP::CastOp op, Value value,
                         APFloat::Semantics fromFloatSemantics,
                         APFloat::Semantics toFloatSemantics) const {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();

    Value converted =
        convertF32ToF8OnAMDGPUHelper(rewriter, op, value, types.front(),
                                     fromFloatSemantics, toFloatSemantics);

    rewriter.replaceOp(op, converted);
    return success();
  }

  Value
  convertF8ToF32OnAMDGPUHelper(DirectRewriter &rewriter, POP::CastOp op,
                               Value value, Type outType,
                               APFloat::Semantics fromFloatSemantics,
                               APFloat::Semantics toFloatSemantics) const {
    assert(getTypeConverter()->getTarget().getTriple().isAMDGPU() &&
           "fast lowering of f32 to f8 is only supported on AMDGPU");

    Location loc = op.getLoc();
    auto simd = cast<SIMDType>(op.getInput().getType());
    const uint64_t size = *simd.getResolvedSize();

    Type i32Type = rewriter.getIntegerType(32);
    Type ui8Type = rewriter.getIntegerType(8);
    Type f32Type = rewriter.getF32Type();

    StringRef intrinsicStr = [&]() -> StringRef {
      if (fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FNUZ ||
          fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN) {
        if (size == 1)
          return "llvm.amdgcn.cvt.f32.fp8";

        return "llvm.amdgcn.cvt.pk.f32.fp8";
      }

      if (fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2FNUZ ||
          fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2) {
        if (size == 1)
          return "llvm.amdgcn.cvt.f32.bf8";

        return "llvm.amdgcn.cvt.pk.f32.bf8";
      }

      llvm_unreachable("conversion format not supported on AMDGPU");
    }();

    assert(llvm::isPowerOf2_64(size) && "SIMD size must be a power of 2");
    VectorType v2f32Type = VectorType::get(2, rewriter.getF32Type());
    Value input = LLVM::UndefOp::create(rewriter, op.getLoc(),
                                        VectorType::get(4, ui8Type));

    if (size == 1) {
      Value src = LLVM::InsertElementOp::create(
          rewriter, loc, input, value,
          createConstant<uint32_t>(rewriter, loc, 0));
      src = LLVM::BitcastOp::create(rewriter, loc, i32Type, src);
      SmallVector<Value> operands{src,
                                  createConstant<uint32_t>(rewriter, loc, 0)};

      return LLVM::CallIntrinsicOp::create(rewriter, loc, f32Type,
                                           rewriter.getStringAttr(intrinsicStr),
                                           operands)
          .getResult(0);
    }

    if (size == 2) {
      Value src = LLVM::InsertElementOp::create(
          rewriter, loc, input,
          extractElement(rewriter, loc, ui8Type, value, 0),
          createConstant<uint32_t>(rewriter, loc, 0));

      src = LLVM::InsertElementOp::create(
          rewriter, loc, src, extractElement(rewriter, loc, ui8Type, value, 1),
          createConstant<uint32_t>(rewriter, loc, 1));

      src = LLVM::BitcastOp::create(rewriter, loc, i32Type, src);
      SmallVector<Value> operands{
          src, LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                        rewriter.getBoolAttr(false))};
      return LLVM::CallIntrinsicOp::create(rewriter, loc, v2f32Type,
                                           rewriter.getStringAttr(intrinsicStr),
                                           operands)
          .getResult(0);
    }

    // SIMD size >= 4
    Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), outType);
    for (uint64_t i = 0; i < size; i += 4) {
      Value src = input;
      for (uint64_t j = 0; j < 4; ++j) {
        src = LLVM::InsertElementOp::create(
            rewriter, loc, src,
            extractElement(rewriter, loc, ui8Type, value, i + j),
            createConstant<uint32_t>(rewriter, loc, j));
      }
      src = LLVM::BitcastOp::create(rewriter, loc, i32Type, src);

      SmallVector<Value> operands{
          src, LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(),
                                        rewriter.getBoolAttr(false))};
      Value res0 = LLVM::CallIntrinsicOp::create(
                       rewriter, loc, v2f32Type,
                       rewriter.getStringAttr(intrinsicStr), operands)
                       .getResult(0);

      operands[1] = LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));
      Value res1 = LLVM::CallIntrinsicOp::create(
                       rewriter, loc, v2f32Type,
                       rewriter.getStringAttr(intrinsicStr), operands)
                       .getResult(0);

      for (uint64_t j = 0; j < 4; ++j) {
        Value res = j < 2 ? res0 : res1;
        result = LLVM::InsertElementOp::create(
            rewriter, loc, result,
            extractElement(rewriter, loc, f32Type, res, j % 2),
            createConstant<uint32_t>(rewriter, loc, i + j));
      }
    }
    return result;
  }

  LogicalResult
  convertF8ToF32OnAMDGPU(DirectRewriter &rewriter, POP::CastOp op, Value value,
                         APFloat::Semantics fromFloatSemantics,
                         APFloat::Semantics toFloatSemantics) const {

    Value converted = convertF8ToF32OnAMDGPUHelper(
        rewriter, op, value, convertType(op.getType()), fromFloatSemantics,
        toFloatSemantics);

    rewriter.replaceOp(op, converted);
    return success();
  }

  /// Convert a `pop.cast` into optimized sequence of asm instructions that
  /// are known to be more efficient for a target than general LLVM's
  /// conversion.
  LogicalResult convertToTargetSpecificCast(DirectRewriter &rewriter,
                                            POP::CastOp cast,
                                            Value value) const {
    TargetInfoAttr target = getTypeConverter()->getTarget();
    if (!target)
      return failure();

    KGENDType fromDType = *cast.getInput().getType().getResolvedDType();
    KGENDType toDType = *cast.getOutput().getType().getResolvedDType();

    if (!fromDType.isFloat() || !toDType.isFloat())
      return failure();

    auto getFltSemantics = [](KGENDType dtype) {
      return APFloat::SemanticsToEnum(*dtype.getFloatSemantics());
    };

    APFloat::Semantics fromFloatSemantics = getFltSemantics(fromDType);
    APFloat::Semantics toFloatSemantics = getFltSemantics(toDType);

    auto simd = dyn_cast<SIMDType>(cast.getInput().getType());

    // Convert F32 to BF16
    if (fromFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle &&
        toFloatSemantics == llvm::APFloat::Semantics::S_BFloat) {
      // post gfx950 targets have native support for f32->bf16
      if (target.getTriple().isAMDGPU() && isAMDGPU(target, {"gfx942"}) &&
          POP::FastmathFlagsInterface(cast).isFast()) {
        return convertF32ToBF16OnAMDGPU(rewriter, cast, value,
                                        fromFloatSemantics, toFloatSemantics);
      }
      return failure();
    }

    // Convert F32 to F8 (either e4m3fn or e5m2)
    if (fromFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle &&
        (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2)) {
      // This might not be ideal to check the targeted GPU by the name, but it's
      // what stdlib does for now. Might be better to use approach similar to
      // NVPTX backend of getting SM version and expecting targeted GPU has at
      // least that version.
      if (simd) {
        if (target.getTriple().isAMDGPU()) {
          return convertF32ToF8OnAMDGPU(rewriter, cast, value,
                                        fromFloatSemantics, toFloatSemantics);
        }
      }
      return failure();
    }

    // Convert F32 to F8 (either e4m3fnuz or e5m2fnuz)
    if (fromFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle &&
        (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FNUZ ||
         toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2FNUZ)) {
      if (simd && target.getTriple().isAMDGPU()) {
        return convertF32ToF8OnAMDGPU(rewriter, cast, value, fromFloatSemantics,
                                      toFloatSemantics);
      }
      return failure();
    }

    // Convert F8 (either e4m3fn or e5m2) to F32
    if ((fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2) &&
        toFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle) {
      // This might not be ideal to check the targeted GPU by the name, but it's
      // what stdlib does for now. Might be better to use approach similar to
      // NVPTX backend of getting SM version and expecting targeted GPU has at
      // least that version.
      if (simd) {
        if (target.getTriple().isAMDGPU()) {
          // Convert F8 (either e4m3fn or e5m2) to F32 on amd mi355x and newer.
          return convertF8ToF32OnAMDGPU(rewriter, cast, value,
                                        fromFloatSemantics, toFloatSemantics);
        }
      }
      return failure();
    }

    // Convert F8 (either e4m3fnuz or e5m2fnuz) to F32 on amd mi300x.
    if ((fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FNUZ ||
         fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2FNUZ) &&
        toFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle) {

      if (simd && target.getTriple().isAMDGPU()) {
        return convertF8ToF32OnAMDGPU(rewriter, cast, value, fromFloatSemantics,
                                      toFloatSemantics);
      }

      return failure();
    }
    return failure();
  }
};

class AMDGPULowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &AMDGPUTraits::get(); }

  mlir::Type lowerKernelArgToMemory(mlir::Type type) const override {
    return lowerAggregateArgToMemory(type);
  }

  bool needsVerification() const override { return true; }

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return verifyNoGPUHeapAllocation(op);
  }

  void populateLowerPOPToLLVMPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::LLVMTypeConverter &converter,
                                      TargetInfoAttr target) const override {
    patterns.add<AMDGPUCastToLLVM>(converter, mlir::PatternBenefit(2));
  }

  void markExportedKernel(mlir::Operation *func) const override {
    // TODO: Need to set the NVVM-equivalent kernel func attr for AMDGPU too?
    cast<LLVM::LLVMFuncOp>(func).setCConv(LLVM::cconv::CConv::AMDGPU_KERNEL);
  }

  bool isExportedKernel(mlir::Operation *func) const override {
    return cast<LLVM::LLVMFuncOp>(func).getCConv() ==
           LLVM::cconv::CConv::AMDGPU_KERNEL;
  }

  llvm::StringRef getKernelByValArgAttrName() const override {
    return LLVM::LLVMDialect::getByRefAttrName();
  }

  /// Promotes AMDGPU kernel `byref` args whose pointee is a struct from flat
  /// (addrspace 0) to the kernarg/constant address space (addrspace 4), and
  /// inserts an `addrspacecast` back to flat at the entry block so the body
  /// keeps operating on flat pointers.
  ///
  /// This matches Clang's HIP/OpenCL lowering and lets
  /// `AMDGPUPromoteKernelArguments` + `InferAddressSpaces` recognize loads
  /// through the kernarg struct as kernarg-origin pointers, enabling
  /// `s_load_dwordx2` for the kernarg itself and `global_load_*` for
  /// downstream dereferences instead of falling back to flat memory ops.
  ///
  /// Must run AFTER the KGEN->LLVM dialect conversion finishes -- mutating
  /// block-arg types mid-conversion would race with the driver's recorded
  /// type mapping and surface as stray `unrealized_conversion_cast` ops in
  /// downstream uses.
  void finalizeConvertedFunction(mlir::Operation *op) const override {
    auto func = cast<LLVM::LLVMFuncOp>(op);
    if (func.getCConv() != LLVM::cconv::CConv::AMDGPU_KERNEL)
      return;

    MLIRContext *ctx = func.getContext();
    auto flatPtrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/0);
    auto kernArgPtrTy = LLVM::LLVMPointerType::get(
        ctx, /*addressSpace=*/llvm::AMDGPUAS::CONSTANT_ADDRESS);
    auto byrefName =
        StringAttr::get(ctx, LLVM::LLVMDialect::getByRefAttrName());

    LLVM::LLVMFunctionType origFnTy = func.getFunctionType();
    SmallVector<Type> newInputs(origFnTy.getParams().begin(),
                                origFnTy.getParams().end());
    Block *entry = func.getBody().empty() ? nullptr : &func.getBody().front();
    bool anyChanged = false;

    for (unsigned i = 0, e = newInputs.size(); i != e; ++i) {
      // Only promote byref kernargs whose pointee is a struct.
      auto byrefAttr =
          dyn_cast_or_null<TypeAttr>(func.getArgAttr(i, byrefName));
      if (!byrefAttr)
        continue;
      if (!isa<LLVM::LLVMStructType>(byrefAttr.getValue()))
        continue;
      if (!isa<LLVM::LLVMPointerType>(newInputs[i]))
        continue;

      newInputs[i] = kernArgPtrTy;
      anyChanged = true;
      if (entry) {
        BlockArgument arg = entry->getArgument(i);
        arg.setType(kernArgPtrTy);
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(entry);
        auto cast = LLVM::AddrSpaceCastOp::create(builder, func.getLoc(),
                                                  flatPtrTy, arg);
        arg.replaceAllUsesExcept(cast.getResult(), cast);
      }
    }

    if (anyChanged) {
      auto newFnTy = LLVM::LLVMFunctionType::get(
          origFnTy.getReturnType(), newInputs, origFnTy.isVarArg());
      func.setFunctionType(newFnTy);
    }
  }

  mlir::LogicalResult
  mapFuncMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                  mlir::NamedAttrList &funcAttrs,
                  llvm::SmallVectorImpl<mlir::Attribute> &passthrough,
                  bool &handled) const override {
    handled = false;
    if (!isa<ROCDL::ROCDLDialect>(attr.getNameDialect()))
      return success();
    // FIXME: Remove when integer conversion to StringAttr is done by the
    // ROCDL -> LLVM IR lowering.
    if (attr.getName() != ROCDL::ROCDLDialect::getFlatWorkGroupSizeAttrName())
      return success();
    ErrorOr<std::pair<int64_t, int64_t>> valuesOr =
        getWorkGroupSizeRange(attr.getValue());
    if (valuesOr.isError())
      return emitError(func->getLoc(), "unsupported `")
             << attr.getName() << "`: " << valuesOr.takeError();
    std::pair<int64_t, int64_t> values = valuesOr.get();
    Builder b(func->getContext());
    funcAttrs.append(attr.getName().strref(),
                     b.getStringAttr(std::to_string(values.first) + ", " +
                                     std::to_string(values.second)));
    handled = true;
    return success();
  }

  void populateLegalConversions(
      TargetInfoAttr target,
      llvm::function_ref<void(KGENDType from, KGENDType to)> addConversion)
      const override {
    addConversion(KGENDType::f8e5m2, KGENDType::f32);
    addConversion(KGENDType::f8e5m2fnuz, KGENDType::f32);
    addConversion(KGENDType::f8e4m3fn, KGENDType::f32);
    addConversion(KGENDType::f8e4m3fnuz, KGENDType::f32);
    addConversion(KGENDType::f32, KGENDType::f8e5m2);
    addConversion(KGENDType::f32, KGENDType::f8e5m2fnuz);
    addConversion(KGENDType::f32, KGENDType::f8e4m3fn);
    addConversion(KGENDType::f32, KGENDType::f8e4m3fnuz);
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<AMDGPULowering> registerAMDGPULowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
