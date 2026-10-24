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
// MLIR-lowering policy for NVIDIA GPU (nvptx) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"
#include "Target/max_license/NVPTX/NVPTXTraits.h"

#include "KGENToLLVM/LLVMLoweringUtils.h"
#include "KGENToLLVM/Target/max_license/GPUCommon.h"
#include "Mojo/KGENDialect/DebugInfoEncoding.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"

#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;

namespace M::KGEN {
namespace {

/// Returns whether `target` is an NVPTX target whose arch is one of
/// `allowedGPUs` matching with or without a trailing "a" arch suffix (e.g.
/// "sm_90a").
bool isNVPTX(TargetInfoAttr target,
             llvm::ArrayRef<llvm::StringRef> allowedGPUs) {
  if (!target || !target.getTriple().isNVPTX())
    return false;
  llvm::StringRef arch = target.getArch();
  if (llvm::is_contained(allowedGPUs, arch))
    return true;
  return arch.ends_with("a") &&
         llvm::is_contained(allowedGPUs, arch.drop_back());
}

/// Returns whether `target` is NVPTX Hopper (sm_90) or above.
bool isHopperAndAbove(TargetInfoAttr target) {
  return isNVPTX(target, {"sm_90", "sm_100", "sm_101", "sm_103", "sm_120"});
}

//===----------------------------------------------------------------------===//
// NVPTXCastToLLVM
//===----------------------------------------------------------------------===//

/// Lowers `pop.cast`s that have a faster NVPTX-specific instruction sequence
/// (fp8/fp16 conversions on Hopper and above). Registered with a higher benefit
/// than the generic `ConvertPOPCast`, so it runs first; when the cast isn't one
/// it handles, it fails and the generic pattern takes over.
struct NVPTXCastToLLVM : public ConvertPOPToLLVMPattern<POP::CastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(POP::CastOp op, POP::CastOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    TargetInfoAttr target = getTypeConverter()->getTarget();
    // These conversions rely on NVPTX instructions available on Hopper+.
    if (!target || !isHopperAndAbove(target))
      return failure();

    KGENDType fromDType = *op.getInput().getType().getResolvedDType();
    KGENDType toDType = *op.getOutput().getType().getResolvedDType();
    if (!fromDType.isFloat() || !toDType.isFloat())
      return failure();

    auto getFltSemantics = [](KGENDType dtype) {
      return APFloat::SemanticsToEnum(*dtype.getFloatSemantics());
    };
    APFloat::Semantics fromFloatSemantics = getFltSemantics(fromDType);
    APFloat::Semantics toFloatSemantics = getFltSemantics(toDType);

    // The NVPTX conversions below operate on SIMD vectors/scalars.
    if (!isa<SIMDType>(op.getInput().getType()))
      return failure();

    Value value = adaptor.getInput();

    // Convert F32 to F8 (either e4m3fn or e5m2).
    if (fromFloatSemantics == llvm::APFloat::Semantics::S_IEEEsingle &&
        (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2)) {
      return convertF32ToF8OnNVPTX(rewriter, op, value, fromFloatSemantics,
                                   toFloatSemantics);
    }

    // Convert F8 (either e4m3fn or e5m2) to F16.
    if ((fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2) &&
        toFloatSemantics == llvm::APFloat::Semantics::S_IEEEhalf) {
      return convertF8ToF16OnNVPTX(rewriter, op, value, fromFloatSemantics,
                                   toFloatSemantics);
    }

    return failure();
  }

private:
  /// Convert vector of FP32 type to vector of FP8 (F8E4M3FN or F8E5M2) on NVPTX
  /// The conversion relies on NVPTX-specific instructions to perform the
  /// conversion, therefore no need to rely on `fast` attribute on a `pop.cast`.
  LogicalResult
  convertF32ToF8OnNVPTX(DirectRewriter &rewriter, POP::CastOp op, Value value,
                        APFloat::Semantics fromFloatSemantics,
                        APFloat::Semantics toFloatSemantics) const {
    assert(isHopperAndAbove(getTypeConverter()->getTarget()) &&
           "lowering of f32 to f8 is only supported on NVIDIA Hopper "
           "architectures or above");
    Location loc = op.getLoc();
    auto simd = cast<SIMDType>(op.getInput().getType());
    const uint64_t size = *simd.getResolvedSize();

    Type f32Type = convertType(rewriter.getF32Type());
    Type f8Type = convertType(op.getType());

    StringRef asmStr =
        toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN
            ? "cvt.rn.satfinite.e4m3x2.f32"
            : "cvt.rn.satfinite.e5m2x2.f32";

    assert(llvm::isPowerOf2_64(size) && "SIMD size must be a power of 2");
    if (size > 1) {
      Value res = LLVM::UndefOp::create(
          rewriter, op.getLoc(),
          VectorType::get(size / 2, rewriter.getIntegerType(16)));
      for (uint64_t i = 0; i < size; i += 2) {
        Value firstFp = extractElement(rewriter, loc, f32Type, value, i + 1);
        Value secondFp = extractElement(rewriter, loc, f32Type, value, i);
        Value converted =
            createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1, $2;",
                            "=h,f,f", rewriter.getIntegerType(16),
                            {firstFp, secondFp})
                .getResult(0);
        res = LLVM::InsertElementOp::create(
            rewriter, loc, res, converted,
            createConstant<uint32_t>(rewriter, loc, i / 2));
      }

      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, f8Type, res);
    } else {
      Value converted =
          createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1, $2;",
                          "=h,f,f", rewriter.getIntegerType(16),
                          {createConstant(rewriter, loc, APFloat(0.0f)), value})
              .getResult(0);
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, f8Type, converted);
    }
    return success();
  }

  /// Helper function to convert vector of F8 (F8E4M3FN or F8E5M2) to F16 on
  /// NVPTX without replacing the operation
  Value convertF8ToF16OnNVPTXHelper(DirectRewriter &rewriter, POP::CastOp op,
                                    Value value,
                                    APFloat::Semantics fromFloatSemantics,
                                    APFloat::Semantics toFloatSemantics) const {
    assert(isHopperAndAbove(getTypeConverter()->getTarget()) &&
           "lowering of f8 to f16 or f32 is only supported on NVIDIA Hopper "
           "architectures or above");
    Location loc = op.getLoc();
    auto simd = cast<SIMDType>(op.getInput().getType());
    const uint64_t size = *simd.getResolvedSize();

    Type f16Type = Float16Type::get(rewriter.getContext());

    StringRef asmStr =
        fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN
            ? "cvt.rn.f16x2.e4m3x2"
            : "cvt.rn.f16x2.e5m2x2";

    Type ui16Type = rewriter.getIntegerType(16);
    assert(llvm::isPowerOf2_64(size) && "SIMD size must be a power of 2");
    if (size > 1) {
      // Bitcast the value to a vector of i16 as NVPTX instruction expects
      // packed f8 as i16.
      value = LLVM::BitcastOp::create(
          rewriter, loc, VectorType::get(size / 2, ui16Type), value);
      // Create a vector of I32 to hold the result of the conversion. At the end
      // it will be bitcasted to f16
      Value res = LLVM::UndefOp::create(
          rewriter, op.getLoc(),
          VectorType::get(size / 2, rewriter.getIntegerType(32)));
      for (uint64_t i = 0, e = size / 2; i < e; ++i) {
        Value converted =
            createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1;", "=r,h",
                            rewriter.getIntegerType(32),
                            {extractElement(rewriter, loc, ui16Type, value, i)})
                .getResult(0);
        res = LLVM::InsertElementOp::create(
            rewriter, loc, res, converted,
            createConstant<uint32_t>(rewriter, loc, i));
      }
      return LLVM::BitcastOp::create(rewriter, loc,
                                     VectorType::get(size, f16Type), res);
    } else {
      Type ui8Type = rewriter.getIntegerType(8);
      Value ui16 = LLVM::ZExtOp::create(
          rewriter, loc, ui16Type,
          LLVM::BitcastOp::create(rewriter, loc, ui8Type, value));
      Value converted =
          createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1;", "=r,h",
                          rewriter.getIntegerType(32), {ui16})
              .getResult(0);
      // At this point result contains two elements, while we're only interested
      // in lower one.
      converted = LLVM::TruncOp::create(rewriter, loc, ui16Type, converted);
      return LLVM::BitcastOp::create(rewriter, loc, f16Type, converted);
    }
  }

  /// Convert scalar or vector of FP8 (F8E4M3FN or F8E5M2) to a scalar or vector
  /// of F16 on NVPTX. The conversion relies on NVPTX-specific instructions to
  /// perform the conversion, therefore no need to rely on `fast` attribute on a
  /// `pop.cast`.
  LogicalResult
  convertF8ToF16OnNVPTX(DirectRewriter &rewriter, POP::CastOp op, Value value,
                        APFloat::Semantics fromFloatSemantics,
                        APFloat::Semantics toFloatSemantics) const {
    rewriter.replaceOp(op, convertF8ToF16OnNVPTXHelper(rewriter, op, value,
                                                       fromFloatSemantics,
                                                       toFloatSemantics));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NVPTXAbsToLLVM
//===----------------------------------------------------------------------===//

/// Lowers `pop.abs` of f16/bf16 to the NVVM `llvm.nvvm.fabs` intrinsic.
/// Registered with a higher benefit than the generic `ConvertPOPAbs`; for any
/// other dtype it fails and the generic pattern takes over.
struct NVPTXAbsToLLVM : public ConvertPOPToLLVMPattern<POP::AbsOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(POP::AbsOp op, POP::AbsOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    TargetInfoAttr target = getTypeConverter()->getTarget();
    if (!target || !target.getTriple().isNVPTX())
      return failure();

    KGENDType dtype = *op.getType().getResolvedDType();
    // Only f16/bf16 have a faster NVVM fabs; the generic pattern handles the
    // rest (including integer/bool/f32 abs).
    if (!(dtype == DType::f16 || dtype == DType::bf16))
      return failure();

    Type type = convertType(op.getType());
    unsigned size = *op.getType().getResolvedSize();
    assert(llvm::isPowerOf2_32(size) && "Unexpected vector width");

    StringRef nvvmIntr = "llvm.nvvm.fabs";
    if (size == 1) {
      rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
          op, type, rewriter.getStringAttr(nvvmIntr),
          SmallVector<Value>{adaptor.getOperand()});
      return success();
    }

    Location loc = op.getLoc();
    // Work down the vector, extracting consecutive two-element subvectors,
    // calling the appropriate 'abs' intrinsic on them, and re-inserting them
    // into the result vector.
    auto res = LLVM::PoisonOp::create(rewriter, op.getLoc(), type).getResult();
    auto subvecTy = VectorType::get(2, cast<VectorType>(type).getElementType());
    for (unsigned i = 0; i < size; i += 2) {
      auto idx = LLVM::ConstantOp::create(rewriter, loc,
                                          rewriter.getIntegerType(64), i);
      auto subvec = LLVM::CallIntrinsicOp::create(
                        rewriter, loc, subvecTy,
                        rewriter.getStringAttr("llvm.vector.extract"),
                        {adaptor.getOperand(), idx})
                        .getResult(0);
      auto abs = LLVM::CallIntrinsicOp::create(rewriter, loc, subvecTy,
                                               rewriter.getStringAttr(nvvmIntr),
                                               {subvec})
                     .getResult(0);
      res = LLVM::CallIntrinsicOp::create(
                rewriter, loc, type,
                rewriter.getStringAttr("llvm.vector.insert"), {res, abs, idx})
                .getResult(0);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NVPTXLowering
//===----------------------------------------------------------------------===//

class NVPTXLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &NVPTXTraits::get(); }

  mlir::Type lowerKernelArgToMemory(mlir::Type type) const override {
    return lowerAggregateArgToMemory(type);
  }

  bool needsVerification() const override { return true; }

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return verifyNoGPUHeapAllocation(op);
  }

  bool isConvergentOp(mlir::Operation *op) const override {
    // TODO: Use trait when it's available in upstream
    return mlir::isa<mlir::NVVM::BarrierOp>(op);
  }

  void populateLowerPOPToLLVMPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::LLVMTypeConverter &converter,
                                      TargetInfoAttr target) const override {
    patterns.add<NVPTXCastToLLVM, NVPTXAbsToLLVM>(converter,
                                                  mlir::PatternBenefit(2));
    // NVVM ops appear only in NVPTX modules; lower them to LLVM here.
    mlir::populateNVVMToLLVMConversionPatterns(patterns);
  }

  void markExportedKernel(mlir::Operation *func) const override {
    func->setAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName(),
                  mlir::UnitAttr::get(func->getContext()));
  }

  bool isExportedKernel(mlir::Operation *func) const override {
    return func->hasAttr(mlir::NVVM::NVVMDialect::getKernelFuncAttrName());
  }

  llvm::StringRef getKernelByValArgAttrName() const override {
    return mlir::LLVM::LLVMDialect::getByValAttrName();
  }

  void mapKernelArgMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                            mlir::NamedAttrList &argAttrs) const override {
    // NVPTX applies nvvm.grid_constant; any other annotation is dropped.
    if (attr.getName() != mlir::NVVM::NVVMDialect::getGridConstantAttrName())
      return;
    mlir::Builder b(func->getContext());
    argAttrs.set(attr.getName(), b.getUnitAttr());
    // FIXME(MOCO-1342): Remove this, once we are able to explicitly pass
    // alignment.
    argAttrs.set(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                 mlir::IntegerAttr::get(b.getI32Type(), 64));
  }

  void populateLegalConversions(
      TargetInfoAttr target,
      llvm::function_ref<void(KGENDType from, KGENDType to)> addConversion)
      const override {
    // The fp8 conversions rely on NVPTX instructions available on Hopper+.
    if (!isHopperAndAbove(target))
      return;
    addConversion(KGENDType::f8e5m2, KGENDType::f16);
    addConversion(KGENDType::f8e4m3fn, KGENDType::f16);
    addConversion(KGENDType::f32, KGENDType::f8e5m2);
    addConversion(KGENDType::f32, KGENDType::f8e4m3fn);
  }

  std::optional<std::string> getDTypeDebugName(KGENDType dtype) const override {
    // cuda-gdb / NVPTX debug info spells dtypes with their CUDA C++ names.
    return DebugInfoEncoding::getKGENDTypeAsCppString(dtype);
  }

  DebugInfo::DIType buildDebugTypeForDType(mlir::MLIRContext *ctx,
                                           KGENDType dtype) const override {
    // cuda-gdb expects non-standard data types to be structure types with
    // special names, e.g.
    // https://docs.nvidia.com/cuda/cuda-math-api/struct____nv__bfloat16.html
    StringRef name;
    if (dtype == KGENDType::f8e5m2)
      name = "__nv_fp8_e5m2";
    else if (dtype == KGENDType::f8e4m3fn)
      name = "__nv_fp8_e4m3";
    else if (dtype == KGENDType::bf16)
      name = "__nv_bfloat16";
    else if (dtype == KGENDType::f16)
      name = "__half";
    else
      return {};

    // The structure contains a single `__x` field holding the raw bits.
    unsigned width = DType(dtype.getValue()).getWidthInBits();
    DType storageDType = *DType::getInt(width, /*isSigned=*/false);
    uint32_t align =
        llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT)) * CHAR_BIT;
    align = std::min(align, width);
    uint64_t size = llvm::alignTo(width, align);
    KGENDType storage(storageDType);
    std::string baseName = getDTypeDebugName(storage).value_or(
        DebugInfoEncoding::getKGENDTypeAsString(storage));
    auto baseType = DebugInfo::DIBasicUIntType::get(ctx, baseName, size, align);
    auto memberType = DebugInfo::DIMemberType::get("__x", baseType);
    return DebugInfo::DIStructType::get(StringAttr::get(ctx, name),
                                        {memberType});
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<NVPTXLowering> registerNVPTXLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
