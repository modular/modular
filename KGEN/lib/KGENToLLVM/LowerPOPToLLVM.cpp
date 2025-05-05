//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLVMLoweringUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;
namespace NVVM = mlir::NVVM;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERGLOBALPOPTOLLVM
#define GEN_PASS_DEF_LOWERPOPTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// POP dialect fastmath flags match the LLVM ones.
static LLVM::FastmathFlagsAttr
convertFastmathFlags(FastmathFlags fmf, ConversionPatternRewriter &rewriter) {
  return rewriter.getAttr<LLVM::FastmathFlagsAttr>(
      static_cast<LLVM::FastmathFlags>(fmf));
}

//===----------------------------------------------------------------------===//
// OneToOneFloatOrIntConversion
//===----------------------------------------------------------------------===//

/// This patterns converts a scalar POP dialect operation to either an integer
/// or floating point LLVM operation one-to-one.
template <typename Op, typename FloatOp, typename SIntOp,
          typename UIntOp = SIntOp>
struct OneToOneFloatOrIntConversion : public ConvertPOPToLLVMPattern<Op> {
  using ConvertPOPToLLVMPattern<Op>::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = this->convertType(op.getType());

    if (dtype.isBool() || dtype.isInt() || dtype.isIndex()) {
      if (std::is_same_v<SIntOp, UIntOp> || dtype.isSInt() || dtype.isIndex())
        rewriter.replaceOpWithNewOp<SIntOp>(op, type, adaptor.getLhs(),
                                            adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<UIntOp>(op, type, adaptor.getLhs(),
                                            adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<FloatOp>(
          op, type, adaptor.getLhs(), adaptor.getRhs(), LLVM_FASTMATH_FLAGS);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPNeg
//===----------------------------------------------------------------------===//

/// Convert an integer pop.neg(x) -> 0 - x
/// and float pop.neg(x) -> llvm.fneg(x)
struct ConvertPOPNeg : public ConvertPOPToLLVMPattern<NegOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isInt() && !dtype.isIndex()) {
      rewriter.replaceOpWithNewOp<LLVM::FNegOp>(op, adaptor.getOperand(),
                                                LLVM_FASTMATH_FLAGS);
      return success();
    }

    Type type = adaptor.getOperand().getType();
    Value zero;
    if (auto vec = dyn_cast<VectorType>(type)) {
      auto intType = dyn_cast<IntegerType>(vec.getElementType());
      if (!intType)
        return op.emitError("could not handle integer type");
      auto apZero = APInt::getZero(intType.getWidth());
      zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), DenseIntElementsAttr::get(vec, apZero));
    } else {
      zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, 0);
    }

    rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPShr
//===----------------------------------------------------------------------===//

/// Lower to `llvm.ashr` if the result dtype is signed and `llvm.lshr`
/// otherwise.
struct ConvertPOPShr : public ConvertPOPToLLVMPattern<ShrOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, ShrOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isSInt() || dtype.isIndex())
      rewriter.replaceOpWithNewOp<LLVM::AShrOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<LLVM::LShrOp>(op, adaptor.getLhs(),
                                                adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFMA
//===----------------------------------------------------------------------===//

/// Convert integer pop.fma(x, y, z) -> x * y + z
/// and float pop.fma(x, y, a) -> llvm.intr.fma(x, y, z)
struct ConvertPOPFMA : public ConvertPOPToLLVMPattern<FMAOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FMAOp op, FMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isInt() || dtype.isIndex()) {
      auto lhs = rewriter.create<LLVM::MulOp>(op.getLoc(), adaptor.getA(),
                                              adaptor.getB());
      rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, lhs, adaptor.getC());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FMAOp>(
          op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
          convertFastmathFlags(op.getFastmathFlags(), rewriter));
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCmp
//===----------------------------------------------------------------------===//

class ConvertPOPCmp : public ConvertPOPToLLVMPattern<CmpOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, CmpOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getLhs().getType().getResolvedDType();
    if (dtype.isBool() || dtype.isInt() || dtype.isIndex() ||
        dtype.isAddress()) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, getICmpPredicate(op.getPred(), dtype.isSInt()), adaptor.getLhs(),
          adaptor.getRhs());
    } else {
      assert(dtype.isFloat());
      Type i1Type = rewriter.getI1Type();
      if (auto simd = dyn_cast<SIMDType>(op.getLhs().getType())) {
        auto size = *simd.getResolvedSize();
        // Vectors of size 1 should remain scalars
        if (size != 1)
          i1Type = VectorType::get(size, i1Type);
      }
      rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
          op, i1Type, getFCmpPredicate(op.getPred()), adaptor.getLhs(),
          adaptor.getRhs(), LLVM_FASTMATH_FLAGS);
    }
    return success();
  }

private:
  /// Convert the integer comparison predicate to the LLVM predicate based on
  /// the signedness.
  static LLVM::ICmpPredicate getICmpPredicate(CmpPredicate pred,
                                              bool isSigned) {
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
  }

  /// Convert the float comparison predicate to the LLVM predicate based on the
  /// signedness.
  static LLVM::FCmpPredicate getFCmpPredicate(CmpPredicate pred) {
    switch (pred) {
    case CmpPredicate::EQ:
      return LLVM::FCmpPredicate::oeq;
    case CmpPredicate::NE:
      return LLVM::FCmpPredicate::one;
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
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCast
//===----------------------------------------------------------------------===//

struct ConvertPOPCast : public ConvertPOPToLLVMPattern<CastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, CastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Target-specific lowering that are known to be better than what LLVM can
    // generate for a generic conversion.
    if (succeeded(
            convertToTargetSpecificCast(rewriter, op, adaptor.getInput())))
      return success();

    KGENDType inDType = *op.getInput().getType().getResolvedDType();
    KGENDType outDType = *op.getOutput().getType().getResolvedDType();

    int64_t inByteCount = getDTypeSizeInBytes(inDType);
    int64_t outByteCount = getDTypeSizeInBytes(outDType);

    // Select the element-wise cast to perform. LLVM integer types are signless,
    // but the signedness semantics of the operation's input and output types
    // affect which casts are selected. `bool` is `i1`.
    StringRef opName;
    if (inDType.isBool() || inDType.isInt() || inDType.isIndex()) {
      if (outDType.isBool() || outDType.isInt() || outDType.isIndex()) {
        // A bool should still become a cast as the bool is only 1 bit but
        // appears as 1 byte here.
        if (outByteCount > inByteCount || inDType.isBool()) {
          // Sign or zero extend.
          opName = inDType.isSInt() ? LLVM::SExtOp::getOperationName()
                                    : LLVM::ZExtOp::getOperationName();
        } else if (outByteCount < inByteCount || outDType.isBool()) {
          // Truncate.
          opName = LLVM::TruncOp::getOperationName();
        }
      } else {
        // Cast from an integer to a float.
        opName = inDType.isSInt() ? LLVM::SIToFPOp::getOperationName()
                                  : LLVM::UIToFPOp::getOperationName();
      }
    } else if (outDType.isBool() || outDType.isInt() || outDType.isIndex()) {
      // Cast from a float to an integer.
      opName = outDType.isSInt() ? LLVM::FPToSIOp::getOperationName()
                                 : LLVM::FPToUIOp::getOperationName();
    } else if (outByteCount > inByteCount) {
      // Extend
      opName = LLVM::FPExtOp::getOperationName();
    } else if (outByteCount < inByteCount) {
      // Truncate.
      opName = LLVM::FPTruncOp::getOperationName();
    } else if (outDType != inDType) {
      // FIXME: Unclear how to cast between `bf16` and `f16`.
      return rewriter.notifyMatchFailure(
          op, "casts between 'bf16' and 'f16' unsupported");
    }

    // If no cast was selected, this is a no-op conversion between equivalent
    // types.
    if (opName.empty()) {
      rewriter.replaceOp(op, adaptor.getInput());
      return success();
    }

    // Create the cast.
    OperationState state(op.getLoc(), opName);
    state.addOperands(adaptor.getInput());
    state.addTypes(convertType(op.getOutput().getType()));
    Operation *cast = rewriter.create(state);
    rewriter.replaceOp(op, cast->getResults());
    return success();
  }

private:
  int64_t getDTypeSizeInBytes(KGENDType dtype) const {
    if (dtype.isIndex())
      return getTypeConverter()->getIndexTypeBitwidth() / CHAR_BIT;
    return dtype.getSizeInBytes();
  }

  LLVM::InlineAsmOp createInlineAsm(ConversionPatternRewriter &rewriter,
                                    Location loc, StringRef asmStr,
                                    StringRef asmConstraints, Type resultType,
                                    SmallVector<Value> operands) const {
    const auto asmDialectAttr = LLVM::AsmDialectAttr::get(
        rewriter.getContext(), LLVM::AsmDialect::AD_ATT);
    return rewriter.create<LLVM::InlineAsmOp>(
        loc, resultType,
        /*operands=*/operands,
        /*asm_string=*/asmStr,
        /*constraints=*/asmConstraints, /*has_side_effects=*/false,
        /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/mlir::ArrayAttr());
  }

  /// Helper function to create a 32-bit signless constant.
  template <typename intType>
  Value createConstant(ConversionPatternRewriter &rewriter, Location loc,
                       uint64_t value) const {
    return rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(sizeof(intType) * 8), value);
  }

  Value createConstant(ConversionPatternRewriter &rewriter, Location loc,
                       APFloat value) const {
    return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF32Type(), value);
  }

  Type getConvertedScalarType(SIMDType simd) const {
    return convertType(
        SIMDType::get(simd.getContext(), /*size=*/1, *simd.getResolvedDType()));
  }

  Type convertKGENDType(MLIRContext *ctx, KGENDType dtype) const {
    return getConvertedScalarType(SIMDType::get(ctx, /*size=*/1, dtype));
  }

  Value extractElement(ConversionPatternRewriter &rewriter, Location loc,
                       Type resType, Value value, unsigned index) const {
    return rewriter.create<LLVM::ExtractElementOp>(
        loc, resType, value, createConstant<uint32_t>(rewriter, loc, index));
  }

  /// Fast conversion of f32 to bf16 on AMDGPU that is not supported by LLVM and
  /// has different handling of NaNs.
  /// The generated sequence has been moved from stdlib (see reference
  /// implementation in PR##54249) to compiler in order:
  ///   - remove boilerplate code from stdlib
  ///   - reduce compile time as stdlib's code won't be parsed and will simply
  ///   be represented by `pop.cast`
  LogicalResult
  convertF32ToBF16OnAMDGPU(ConversionPatternRewriter &rewriter, CastOp op,
                           Value value, APFloat::Semantics fromFloatSemantics,
                           APFloat::Semantics toFloatSemantics) const {
    assert(getTypeConverter()->getTarget().getTriple().isAMDGPU() &&
           "fast lowering of f32 to bf16 is only supported on AMDGPU");
    assert(op.getFastAttr() && "`fast` attribute must be set on a `pop.cast`");

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
                          "=v,v,v,v", vecI32, {value, lsb, roundedBias})
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
          rewriter.create<LLVM::LShrOp>(loc, floatBits, mantissaDiff);

      shifted = rewriter.create<LLVM::TruncOp>(loc, rewriter.getIntegerType(16),
                                               shifted);

      return rewriter.create<LLVM::BitcastOp>(loc, bf16Type, shifted);
    };

    Value res;
    if (size > 1) {
      res = rewriter.create<LLVM::UndefOp>(op.getLoc(),
                                           VectorType::get(size, bf16Type));
      for (uint32_t i = 0; i < size; ++i) {
        Value element = extractElement(rewriter, loc, f32Type, value, i);
        Value converted = convertSingleValue(element);
        res = rewriter.create<LLVM::InsertElementOp>(
            loc, res, converted, createConstant<uint32_t>(rewriter, loc, i));
      }
    } else {
      res = convertSingleValue(value);
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  /// Convert vector of FP32 type to vector of FP8 (F8E4M3FN or F8E5M2) on NVPTX
  /// The conversion relies on NVPTX-specific instructions to perform the
  /// conversion, therefore no need to rely on `fast` attribute on a `pop.cast`.
  LogicalResult
  convertF32ToF8OnNVPTX(ConversionPatternRewriter &rewriter, CastOp op,
                        Value value, APFloat::Semantics fromFloatSemantics,
                        APFloat::Semantics toFloatSemantics) const {
    assert(isNVPTX_HopperAndAbove(getTypeConverter()->getTarget()) &&
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
      Value res = rewriter.create<LLVM::UndefOp>(
          op.getLoc(), VectorType::get(size / 2, rewriter.getIntegerType(16)));
      for (uint64_t i = 0; i < size; i += 2) {
        Value firstFp = extractElement(rewriter, loc, f32Type, value, i + 1);
        Value secondFp = extractElement(rewriter, loc, f32Type, value, i);
        Value converted =
            createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1, $2;",
                            "=h,f,f", rewriter.getIntegerType(16),
                            {firstFp, secondFp})
                .getResult(0);
        res = rewriter.create<LLVM::InsertElementOp>(
            loc, res, converted,
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
  Value convertF8ToF16OnNVPTXHelper(ConversionPatternRewriter &rewriter,
                                    CastOp op, Value value,
                                    APFloat::Semantics fromFloatSemantics,
                                    APFloat::Semantics toFloatSemantics) const {
    assert(isNVPTX_HopperAndAbove(getTypeConverter()->getTarget()) &&
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
      value = rewriter.create<LLVM::BitcastOp>(
          loc, VectorType::get(size / 2, ui16Type), value);
      // Create a vector of I32 to hold the result of the conversion. At the end
      // it will be bitcasted to f16
      Value res = rewriter.create<LLVM::UndefOp>(
          op.getLoc(), VectorType::get(size / 2, rewriter.getIntegerType(32)));
      for (uint64_t i = 0, e = size / 2; i < e; ++i) {
        Value converted =
            createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1;", "=r,h",
                            rewriter.getIntegerType(32),
                            {extractElement(rewriter, loc, ui16Type, value, i)})
                .getResult(0);
        res = rewriter.create<LLVM::InsertElementOp>(
            loc, res, converted, createConstant<uint32_t>(rewriter, loc, i));
      }
      return rewriter.create<LLVM::BitcastOp>(
          loc, VectorType::get(size, f16Type), res);
    } else {
      Type ui8Type = rewriter.getIntegerType(8);
      Value ui16 = rewriter.create<LLVM::ZExtOp>(
          loc, ui16Type, rewriter.create<LLVM::BitcastOp>(loc, ui8Type, value));
      Value converted =
          createInlineAsm(rewriter, loc, asmStr.str() + " $0, $1;", "=r,h",
                          rewriter.getIntegerType(32), {ui16})
              .getResult(0);
      // At this point result contains two elements, while we're only interested
      // in lower one.
      converted = rewriter.create<LLVM::TruncOp>(loc, ui16Type, converted);
      return rewriter.create<LLVM::BitcastOp>(loc, f16Type, converted);
    }
  }

  /// Convert scalar or vector of FP8 (F8E4M3FN or F8E5M2) to a scalar or vector
  /// of F16 on NVPTX. The conversion relies on NVPTX-specific instructions to
  /// perform the conversion, therefore no need to rely on `fast` attribute on a
  /// `pop.cast`.
  LogicalResult
  convertF8ToF16OnNVPTX(ConversionPatternRewriter &rewriter, CastOp op,
                        Value value, APFloat::Semantics fromFloatSemantics,
                        APFloat::Semantics toFloatSemantics) const {
    rewriter.replaceOp(op, convertF8ToF16OnNVPTXHelper(rewriter, op, value,
                                                       fromFloatSemantics,
                                                       toFloatSemantics));
    return success();
  }

  /// Convert scalar or vector of FP8 (F8E4M3FN or F8E5M2) to a scalar or vector
  /// of F32 on NVPTX. The conversion relies on NVPTX-specific instructions to
  /// perform the conversion, therefore no need to rely on `fast` attribute on a
  /// `pop.cast`.
  LogicalResult
  convertF8ToF32OnNVPTX(ConversionPatternRewriter &rewriter, CastOp op,
                        Value value, APFloat::Semantics fromFloatSemantics,
                        APFloat::Semantics toFloatSemantics) const {
    Value res = convertF8ToF16OnNVPTXHelper(
        rewriter, op, value, fromFloatSemantics, toFloatSemantics);
    rewriter.replaceOpWithNewOp<LLVM::FPExtOp>(op, convertType(op.getType()),
                                               res);
    return success();
  }

  /// Convert scalar or vector of FP8 (F8E4M3FN or F8E5M2) to a scalar or vector
  /// of BF16 on NVPTX.
  /// Since NVPTX has no instruction to do this directly, do this with a
  /// sequence FP8 -> F32 -> BF16
  LogicalResult
  convertF8ToBF16OnNVPTX(ConversionPatternRewriter &rewriter, CastOp op,
                         Value value, APFloat::Semantics fromFloatSemantics,
                         APFloat::Semantics toFloatSemantics) const {
    Value f16Result = convertF8ToF16OnNVPTXHelper(
        rewriter, op, value, fromFloatSemantics, toFloatSemantics);
    auto simdF32 =
        SIMDType::get(rewriter.getContext(),
                      /*size=*/*op.getType().getResolvedSize(), KGENDType::f32);
    Value f32Result = rewriter.create<LLVM::FPExtOp>(
        op.getLoc(), convertType(simdF32), f16Result);

    rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, convertType(op.getType()),
                                                 f32Result);
    return success();
  }

  /// Convert scalar or vector of BF16 to a scalar or vector of FP8 (F8E4M3FN or
  /// F8E5M2) on NVPTX. Since NVPTX has no instruction to do this directly, do
  /// this with a sequence BF16 -> F32 -> FP8
  LogicalResult
  convertBF16toF8OnNVPTX(ConversionPatternRewriter &rewriter, CastOp op,
                         Value value, APFloat::Semantics fromFloatSemantics,
                         APFloat::Semantics toFloatSemantics) const {
    auto simdF32 =
        SIMDType::get(rewriter.getContext(),
                      /*size=*/*op.getType().getResolvedSize(), KGENDType::f32);
    Value f32Result = rewriter.create<LLVM::FPExtOp>(
        op.getLoc(), convertType(simdF32), value);

    return convertF32ToF8OnNVPTX(rewriter, op, f32Result, fromFloatSemantics,
                                 toFloatSemantics);
  }

  /// Convert a `pop.cast` into optimized sequence of asm instructions that are
  /// known to be more efficient for a target than general LLVM's conversion.
  LogicalResult convertToTargetSpecificCast(ConversionPatternRewriter &rewriter,
                                            CastOp cast, Value value) const {
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
      if (target.getTriple().isAMDGPU() && cast.getFastAttr()) {
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
      if (simd && isNVPTX_HopperAndAbove(target)) {
        return convertF32ToF8OnNVPTX(rewriter, cast, value, fromFloatSemantics,
                                     toFloatSemantics);
      }
      return failure();
    }
    // Convert F8 (either e4m3fn or e5m2) to F16
    if ((fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2) &&
        toFloatSemantics == llvm::APFloat::Semantics::S_IEEEhalf) {
      // This might not be ideal to check the targeted GPU by the name, but it's
      // what stdlib does for now. Might be better to use approach similar to
      // NVPTX backend of getting SM version and expecting targeted GPU has at
      // least that version.
      if (simd && isNVPTX_HopperAndAbove(target)) {
        return convertF8ToF16OnNVPTX(rewriter, cast, value, fromFloatSemantics,
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
      if (simd && isNVPTX_HopperAndAbove(target)) {
        return convertF8ToF32OnNVPTX(rewriter, cast, value, fromFloatSemantics,
                                     toFloatSemantics);
      }
      return failure();
    }

    // Convert F8 (either e4m3fn or e5m2) to BF16
    if ((fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         fromFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2) &&
        toFloatSemantics == llvm::APFloat::Semantics::S_BFloat) {
      // This might not be ideal to check the targeted GPU by the name, but it's
      // what stdlib does for now. Might be better to use approach similar to
      // NVPTX backend of getting SM version and expecting targeted GPU has at
      // least that version.
      if (simd && isNVPTX_HopperAndAbove(target)) {
        return convertF8ToBF16OnNVPTX(rewriter, cast, value, fromFloatSemantics,
                                      toFloatSemantics);
      }
      return failure();
    }

    // Convert BF16 to F8 (either e4m3fn or e5m2)
    if (fromFloatSemantics == llvm::APFloat::Semantics::S_BFloat &&
        (toFloatSemantics == llvm::APFloat::Semantics::S_Float8E4M3FN ||
         toFloatSemantics == llvm::APFloat::Semantics::S_Float8E5M2)) {
      // This might not be ideal to check the targeted GPU by the name, but it's
      // what stdlib does for now. Might be better to use approach similar to
      // NVPTX backend of getting SM version and expecting targeted GPU has at
      // least that version.
      if (simd && isNVPTX_HopperAndAbove(target)) {
        return convertBF16toF8OnNVPTX(rewriter, cast, value, fromFloatSemantics,
                                      toFloatSemantics);
      }
      return failure();
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDSelect
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDSelect : public ConvertPOPToLLVMPattern<SIMDSelectOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDSelectOp op, SIMDSelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue(), LLVM_FASTMATH_FLAGS);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDSplat
//===----------------------------------------------------------------------===//

/// Convert a SIMD splat to an `insertelement` into an `undef` and then a
/// zero-initialized `shufflevector`.
struct ConvertPOPSIMDSplat : public ConvertPOPToLLVMPattern<SIMDSplatOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDSplatOp op, SIMDSplatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the vector is size 1, skip the shuffle.
    if (op.getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getScalar());
      return success();
    }

    SIMDType simdType = op.getType();
    int64_t size = *simdType.getResolvedSize();
    Value undef =
        rewriter.create<LLVM::UndefOp>(op.getLoc(), convertType(simdType));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0));
    Value vector = rewriter.create<LLVM::InsertElementOp>(
        op.getLoc(), undef, adaptor.getScalar(), zero);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
        op, vector, undef, /*mask=*/SmallVector<int32_t>(size, 0));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDInsertElement
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDInsertElement
    : public ConvertPOPToLLVMPattern<SIMDInsertElementOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDInsertElementOp op, SIMDInsertElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVector().getType().isScalar()) {
      // If the vector is size 1, return the value as is - it's a scalar.
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        op, convertType(op.getType()), adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDShuffle
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDShuffle : public ConvertPOPToLLVMPattern<SIMDShuffleOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDShuffleOp op, SIMDShuffleOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mask = cast<POP::ArrayAttr>(adaptor.getMask());
    SmallVector<int32_t> maskValues;
    for (TypedAttr maskElement : mask.getValues())
      maskValues.push_back(cast<IntegerAttr>(maskElement).getInt());

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto inputSize = *op.getLhs().getType().getResolvedSize();
    if (inputSize != 1) {
      // Both LHS and RHS are vectors - generate LLVM ShuffleVector
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, lhs, rhs, rewriter.getDenseI32ArrayAttr(maskValues));

      return success();
    }
    // Special handling for inputs consisting of just 1 element - instead of
    // converting them to vectors and generating shufflevector for them, we will
    // instead generate a sequence of insertelement operations.  Since there are
    // just two elements to pick from, mask should only contain 0s and 1s. If it
    // contains a different value, the behavior is undefined - we will simply
    // treat such a case as value 1.
    KGENDType dtype = *op.getType().getResolvedDType();
    auto llvmVecType = VectorType::get(
        mask.getValues().size(),
        *getMLIRTypeForDType(op.getType().getContext(), dtype,
                             getTypeConverter()->getIndexTypeBitwidth()));
    Value result = rewriter.create<LLVM::UndefOp>(op.getLoc(), llvmVecType);
    int idx = 0;
    for (int32_t maskElement : maskValues) {
      Value pos = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32IntegerAttr(idx));
      result = rewriter.create<LLVM ::InsertElementOp>(
          op.getLoc(), result, maskElement == 0 ? lhs : rhs, pos);
      idx++;
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDExtractElement
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDExtractElement
    : public ConvertPOPToLLVMPattern<SIMDExtractElementOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDExtractElementOp op, SIMDExtractElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special handling for scalars
    if (op.getVector().getType().isScalar()) {
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        op, convertType(op.getType()), adaptor.getVector(),
        adaptor.getPosition());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPOffset
//===----------------------------------------------------------------------===//

struct ConvertPOPOffset : public ConvertPOPToLLVMPattern<OffsetOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(OffsetOp op, OffsetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type elementType =
        typeConverter->convertType(op.getPtr().getType().getElementType());

    // Set the address space if specified.
    unsigned addrSpace = 0;
    if (auto addrSpaceAttr =
            cast_or_null<IntegerAttr>(op.getPtr().getType().getAddressSpace()))
      addrSpace = addrSpaceAttr.getInt();

    // Coerce the index to the same type as the pointer which is required by the
    // address space.
    Type intPtrType = getIntPtrType(addrSpace);
    size_t intPtrTypeSize = intPtrType.getIntOrFloatBitWidth();
    size_t indexTypeSize = adaptor.getIndex().getType().getIntOrFloatBitWidth();

    Value offset;
    if (intPtrTypeSize == indexTypeSize) {
      offset = adaptor.getIndex();
    } else if (intPtrTypeSize < indexTypeSize) {
      offset = rewriter.createOrFold<LLVM::TruncOp>(op.getLoc(), intPtrType,
                                                    adaptor.getIndex());
    } else {
      offset = rewriter.createOrFold<LLVM::SExtOp>(op.getLoc(), intPtrType,
                                                   adaptor.getIndex());
    }
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, /*resultType=*/adaptor.getPtr().getType(),
        /*elementType=*/elementType,
        /*basePtr=*/adaptor.getPtr(),
        /*indices=*/ValueRange{offset},
        /*noWrapFlags=*/LLVM::GEPNoWrapFlags::inbounds);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSelect
//===----------------------------------------------------------------------===//

struct ConvertPOPSelect : public ConvertPOPToLLVMPattern<SelectOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, SelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue(), LLVM_FASTMATH_FLAGS);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocation
//===----------------------------------------------------------------------===//

/// A `pop.stack_allocation` is lowered by converting it to an `llvm.alloca`
/// with lifetime markers and hoisting it to the top of the enclosing
/// function.
class ConvertPOPStackAllocation
    : public ConvertPOPToLLVMPattern<StackAllocationOp> {
public:
  explicit ConvertPOPStackAllocation(mlir::LLVMTypeConverter &typeConverter,
                                     TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(typeConverter), target(target) {}

  LogicalResult
  matchAndRewrite(StackAllocationOp op, StackAllocationOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// The target info.
  TargetInfoAttr target;
  mlir::DominanceInfo domInfo;

  static unsigned resolveAlignment(std::optional<TypedAttr> alignment) {
    if (!alignment)
      return 0;
    return cast<IntegerAttr>(*alignment).getInt();
  }
};

/// Generate the LLVM IR to materialize an alloca with the given LLVM type and
/// count. The alloca is created at the top of the given block, and lifetime
/// markers are inserted at the end of the given operation's block.
static Value materializeLLVMAlloca(OpBuilder &b, TargetInfoAttr target,
                                   Type elementType, int64_t count,
                                   Operation *op, int64_t typeAllocSize,
                                   int64_t align) {
  unsigned addressSpace = 0;
  auto alloca = dyn_cast<StackAllocationOp>(op);
  if (alloca) {
    if (auto addrSpaceAttr =
            cast_or_null<IntegerAttr>(alloca.getType().getAddressSpace()))
      addressSpace = addrSpaceAttr.getInt();
  }

  bool needAddrSpaceCast = false;
  if (addressSpace == 0) {
    addressSpace = target.getDataLayout().getAllocaAddrSpace();
    needAddrSpaceCast = addressSpace != 0;
  }

  Value countVal =
      b.create<LLVM::ConstantOp>(op->getLoc(), b.getI64IntegerAttr(count));
  Value ptr = b.create<LLVM::AllocaOp>(
      op->getLoc(), LLVM::LLVMPointerType::get(b.getContext(), addressSpace),
      elementType, countVal, align);

  if (alloca && alloca.getMarkedLifetimes()) {
    // If this alloca has marked lifetimes, it always begins as dead.
    b.create<LLVM::LifetimeEndOp>(op->getLoc(), typeAllocSize * count, ptr);
  } else {
    // Insert lifetime markers starting from the op to the end of its block.
    b.setInsertionPoint(op);
    auto start = b.create<LLVM::LifetimeStartOp>(op->getLoc(),
                                                 typeAllocSize * count, ptr);
    b.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
    b.create<LLVM::LifetimeEndOp>(op->getLoc(), typeAllocSize * count, ptr);
    b.setInsertionPointAfter(start);
  }

  if (needAddrSpaceCast) {
    ptr = b.create<LLVM::AddrSpaceCastOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/0), ptr);
  }

  return ptr;
}

LogicalResult ConvertPOPStackAllocation::matchAndRewrite(
    StackAllocationOp op, StackAllocationOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  PointerType ptrType = cast<PointerType>(op.getType());
  Type elementType = convertType(ptrType.getElementType());
  if (!elementType)
    return op.emitError("could not lower pointer element type");

  // Compute the bytecount of the allocated buffer.
  std::optional<int64_t> typeAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, ptrType.getElementType());
  if (!typeAllocSize)
    return op.emitError("could not get size of variadic element");

  // Check to see if this stack allocation has a single pop.store to it and
  // some number of pop.loads.  If so, we know the store will dominate the loads
  // so we can just completely eliminate this.  This is a form of guaranteed
  // optimization, and it also matters for LLVM intrinsic propagation.
  StoreOp theStore;
  SmallVector<LoadOp> loads;
  bool allSimple = true;
  for (Operation *user : op->getUsers()) {
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
      continue;
    // If this is the first store to the stack allocation, remember it.
    if (auto storeOp = dyn_cast<StoreOp>(user))
      if (storeOp.getOperand(1) == op.getResult() && !theStore) {
        theStore = storeOp;
        continue;
      }
    // Remember all the loads.
    if (auto loadOp = dyn_cast<LoadOp>(user)) {
      loads.push_back(loadOp);
      continue;
    }
    allSimple = false;
    break;
  }

  // If all the accesses are simple, we can just remove this entirely.
  if (allSimple && theStore) {
    bool dominates = true;
    for (auto loadOp : loads) {
      if (!domInfo.dominates(theStore, loadOp)) {
        dominates = false;
        break;
      }
    }
    if (dominates) {
      for (auto load : loads) {
        load.replaceAllUsesWith(theStore.getOperand(0));
        rewriter.eraseOp(load);
      }
    }
  }

  Value alloca = materializeLLVMAlloca(
      rewriter, target, elementType, cast<IntegerAttr>(op.getCount()).getInt(),
      op, *typeAllocSize, resolveAlignment(op.getAlignment()));
  rewriter.replaceOp(op, alloca);
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocLifetimeStart
//===----------------------------------------------------------------------===//

template <typename OpT>
static void lowerLifetimeMarker(Operation *op, ValueRange values,
                                TargetInfoAttr target,
                                ConversionPatternRewriter &b) {
  for (auto [ptr, values] : llvm::zip(op->getOperands(), values)) {
    int64_t typeAllocSize = *DataLayoutInterface::getTypeAllocSize(
        target, cast<PointerType>(ptr.getType()).getElementType());
    auto alloc = ptr.template getDefiningOp<StackAllocationOp>();
    assert(alloc && "expected a parent stack allocation");
    int64_t count = cast<IntegerAttr>(alloc.getCountAttr()).getInt();
    b.create<OpT>(op->getLoc(), typeAllocSize * count, values);
  }
  b.eraseOp(op);
}

class ConvertPOPStackAllocLifetimeStart
    : public ConvertPOPToLLVMPattern<StackAllocLifetimeStartOp> {
public:
  explicit ConvertPOPStackAllocLifetimeStart(mlir::LLVMTypeConverter &tc,
                                             TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(tc), target(target) {}

  LogicalResult matchAndRewrite(StackAllocLifetimeStartOp op,
                                StackAllocLifetimeStartOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    lowerLifetimeMarker<LLVM::LifetimeStartOp>(op, adaptor.getValues(), target,
                                               b);
    return success();
  }

private:
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPStackAllocLifetimeEnd
//===----------------------------------------------------------------------===//

class ConvertPOPStackAllocLifetimeEnd
    : public ConvertPOPToLLVMPattern<StackAllocLifetimeEndOp> {
public:
  explicit ConvertPOPStackAllocLifetimeEnd(mlir::LLVMTypeConverter &tc,
                                           TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(tc), target(target) {}

  LogicalResult matchAndRewrite(StackAllocLifetimeEndOp op,
                                StackAllocLifetimeEndOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    lowerLifetimeMarker<LLVM::LifetimeEndOp>(op, adaptor.getValues(), target,
                                             b);
    return success();
  }

private:
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayCreate
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayCreate : public ConvertPOPToLLVMPattern<ArrayCreateOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, ArrayCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    if (!type)
      return op.emitError("failed to convert array type");

    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
    for (auto [idx, val] : llvm::enumerate(adaptor.getOperands()))
      array =
          rewriter.create<LLVM::InsertValueOp>(op.getLoc(), array, val, idx);
    rewriter.replaceOp(op, array);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayRepeat
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayRepeat : public ConvertPOPToLLVMPattern<ArrayRepeatOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayRepeatOp op, ArrayRepeatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    if (!type)
      return op.emitError("failed to convert array type");

    Value array = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
    // Fill the consecutive elements of the array by cycling through the
    // operands until the array is filled.
    for (unsigned i = 0, size = *op.getType().getResolvedSize(); i < size;) {
      for (auto it = adaptor.getOperands().begin(),
                e = adaptor.getOperands().end();
           it != e && i < size; ++it, ++i) {
        array =
            rewriter.create<LLVM::InsertValueOp>(op.getLoc(), array, *it, i);
      }
    }
    rewriter.replaceOp(op, array);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayGet
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayGet : public ConvertPOPToLLVMPattern<ArrayGetOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayGetOp op, ArrayGetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getArray(), cast<IntegerAttr>(op.getIndex()).getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayReplace
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayReplace : public ConvertPOPToLLVMPattern<ArrayReplaceOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayReplaceOp op, ArrayReplaceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getArray(), adaptor.getValue(),
        cast<IntegerAttr>(op.getIndex()).getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayGEP
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayGEP : public ConvertPOPToLLVMPattern<ArrayGEPOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayGEPOp op, ArrayGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrType = convertType(op.getType());
    Type elementType = convertType(op.getArray().getType().getElementType());
    if (!ptrType)
      return op.emitError("failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrType, elementType, adaptor.getArray(),
        ArrayRef<LLVM::GEPArg>{0, adaptor.getIndex()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// getAlignment, getAtomicOrdering
//===----------------------------------------------------------------------===//

static unsigned getAlignment(const POPToLLVMTypeConverter *tc,
                             PointerType ptrType,
                             TypedAttr alignmentAttr = {}) {
  // If we have the alignment attribute, use it.
  if (alignmentAttr)
    return cast<IntegerAttr>(alignmentAttr).getInt();

  return tc->getTypeABIAlign(tc->convertType(ptrType.getElementType()));
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

//===----------------------------------------------------------------------===//
// ConvertPOPLoad
//===----------------------------------------------------------------------===//

struct ConvertPOPLoad : ConvertPOPToLLVMPattern<LoadOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, LoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptrType = cast<PointerType>(op.getPtr().getType());
    Type elementType = typeConverter->convertType(ptrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), ptrType, adaptor.getAlignmentAttr());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, elementType, adaptor.getPtr(), /*alignment=*/alignment,
        /*isVolatile=*/adaptor.getIsVolatile(), /*isNonTemporal=*/false,
        /*isInvariant=*/adaptor.getIsInvariant(),
        /*isInvariantGroup=*/false,
        /*ordering=*/getAtomicOrdering(adaptor.getOrdering()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStore
//===----------------------------------------------------------------------===//

struct ConvertPOPStore : ConvertPOPToLLVMPattern<StoreOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ptrType = cast<PointerType>(op.getPtr().getType());
    unsigned alignment =
        getAlignment(getTypeConverter(), ptrType, adaptor.getAlignmentAttr());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getArg(), adaptor.getPtr(), /*alignment=*/alignment,
        /*isVolatile=*/adaptor.getIsVolatile(), /*isNonTemporal=*/false,
        /*isInvariantGroup=*/false,
        /*ordering=*/getAtomicOrdering(adaptor.getOrdering()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariadicCreate
//===----------------------------------------------------------------------===//

/// Converts a `pop.variadic.create` to:
/// 1. An `alloca`, to allocate space for a sequence of elements on the stack.
/// 2. Zero or more GEP and `store`, to insert elements of the variadic sequence
///    into the allocated space.
/// 3. A struct that holds the pointer to allocated sequence, and the number of
///    elements.
static LogicalResult convertVariadicCreate(VariadicType resultType,
                                           ValueRange operands, Operation *op,
                                           ConversionPatternRewriter &rewriter,
                                           const TypeConverter *typeConverter,
                                           TargetInfoAttr target) {

  // 1. Allocate space for an array of elements.
  Type opElementType = resultType.getElementType();
  std::optional<int64_t> typeAllocSize =
      DataLayoutInterface::getTypeAllocSize(target, opElementType);
  std::optional<int64_t> typeABIAlign =
      DataLayoutInterface::getTypeABIAlign(target, opElementType);
  if (!typeAllocSize || !typeABIAlign)
    return op->emitError("failed to get element type size and alignment");

  Type elementType = typeConverter->convertType(opElementType);
  if (!elementType)
    return op->emitError("failed to convert element type");

  size_t count = operands.size();
  Value ptr = materializeLLVMAlloca(rewriter, target, elementType, count, op,
                                    *typeAllocSize, *typeABIAlign);

  // 2. Store elements of the sequence into the allocated space.
  Type indexType = typeConverter->convertType(rewriter.getIndexType());
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  for (auto [index, operand] : llvm::enumerate(operands)) {
    Value indexConstant = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(indexType, index));
    auto destination = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), /*resultType=*/opaquePtr,
        /*basePtrType=*/elementType, /*basePtr=*/ptr,
        ArrayRef<LLVM::GEPArg>{indexConstant});
    rewriter.create<LLVM::StoreOp>(op->getLoc(), operand, destination);
  }

  // 3. Replace the `pop.variadic.create` op with a struct containing the
  //    pointer & the size of the sequence.
  Type structType = typeConverter->convertType(resultType);
  if (!structType)
    return op->emitError("failed to convert variadic type");
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  Value container = materializeLLVMStruct(
      b, structType,
      ValueRange{ptr,
                 rewriter.create<LLVM::ConstantOp>(
                     op->getLoc(), rewriter.getIntegerAttr(indexType, count))

      });
  rewriter.replaceOp(op, container);
  return success();
}

class ConvertPOPVariadicCreate
    : public ConvertPOPToLLVMPattern<VariadicCreateOp> {
public:
  explicit ConvertPOPVariadicCreate(mlir::LLVMTypeConverter &typeConverter,
                                    TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(typeConverter), target(target) {}

  LogicalResult
  matchAndRewrite(VariadicCreateOp op, VariadicCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return convertVariadicCreate(op.getType(), adaptor.getOperands(), op,
                                 rewriter, typeConverter, target);
  }

private:
  /// The target info.
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariadicSplat
//===----------------------------------------------------------------------===//

/// Converts a `pop.variadic.splat` to the same machinery as
/// `pop.variadic.create`.
class ConvertPOPVariadicSplat
    : public ConvertPOPToLLVMPattern<VariadicSplatOp> {
public:
  explicit ConvertPOPVariadicSplat(mlir::LLVMTypeConverter &typeConverter,
                                   TargetInfoAttr target)
      : ConvertPOPToLLVMPattern(typeConverter), target(target) {}

  LogicalResult
  matchAndRewrite(VariadicSplatOp op, VariadicSplatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto numElements = dyn_cast<IntegerAttr>(adaptor.getNumElements());
    if (!numElements)
      return op.emitError("pop.variadic.splat has parametric # elements");

    SmallVector<Value> operands(numElements.getInt(), adaptor.getOperand());
    return convertVariadicCreate(op.getType(), operands, op, rewriter,
                                 typeConverter, target);
  }

private:
  /// The enclosing function body.
  /// The target info.
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariadicGet
//===----------------------------------------------------------------------===//

/// Converts a `pop.variadic.get` into LLVM ops that load one of the elements of
/// the underlying struct that represents the `!kgen.variadic` type.
struct ConvertPOPVariadicGet : public ConvertPOPToLLVMPattern<VariadicGetOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariadicGetOp op, VariadicGetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrElement =
        typeConverter->convertType(op.getVariadic().getType().getElementType());
    Value ptr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(),
                                                      adaptor.getVariadic(), 0);
    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), ptr.getType(), ptrElement, ptr, adaptor.getIndex());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, ptrElement, gep);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariadicSize
//===----------------------------------------------------------------------===//

/// Converts a `pop.variadic.size` into LLVM ops that load the size member
/// of the underlying struct representing the `!kgen.variadic` type.
struct ConvertPOPVariadicSize : public ConvertPOPToLLVMPattern<VariadicSizeOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariadicSizeOp op, VariadicSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getOperand(),
                                                      1);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastToBuiltin
//===----------------------------------------------------------------------===//

struct ConvertPOPCastToBuiltin : ConvertPOPToLLVMPattern<CastToBuiltinOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastToBuiltinOp op, CastToBuiltinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastFromBuiltin
//===----------------------------------------------------------------------===//

struct ConvertPOPCastFromBuiltin : ConvertPOPToLLVMPattern<CastFromBuiltinOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastFromBuiltinOp op, CastFromBuiltinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPInlineAsm
//===----------------------------------------------------------------------===//

// Given %20 with something like:
//     %19 = builtin.unrealized_conversion_cast %18 : !llvm.struct<(i8, i1)> to
//     !kgen.struct<(scalar<ui8>, i1)> %20 = builtin.unrealized_conversion_cast
//     %19 : !kgen.struct<(scalar<ui8>, i1)> to !llvm.struct<(i8, i1)>
// Return the input %18.
static Value squashPointlessCasts(Value v) {
  auto cast1Op = v.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast1Op || cast1Op.getNumOperands() != 1 || cast1Op.getNumResults() != 1)
    return v;

  auto cast2Op =
      cast1Op.getOperand(0).getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast2Op || cast1Op.getNumOperands() != 1 ||
      cast1Op.getNumResults() != 1 ||
      cast2Op.getOperand(0).getType() != v.getType())
    return v;

  return squashPointlessCasts(cast2Op.getOperand(0));
}

/// Expand one level of structs so kgen.pack elements are passed as individual
/// values instead of as a kgen.struct.
static SmallVector<Value> expandOperands(ConversionPatternRewriter &rewriter,
                                         Location loc, ValueRange args) {
  SmallVector<Value> operands;
  operands.reserve(args.size());
  for (auto value : args) {
    // Squash pointless conversion casts that will get in the way of folds.
    value = squashPointlessCasts(value);

    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(value.getType())) {
      // Unpack each of the elements.
      for (size_t i = 0, e = structTy.getBody().size(); i != e; ++i) {
        auto elt = rewriter.createOrFold<LLVM::ExtractValueOp>(loc, value, i);
        operands.push_back(elt);
      }
    } else {
      operands.push_back(value);
    }
  }
  return operands;
}

struct ConvertPOPInlineAsm : ConvertPOPToLLVMPattern<InlineAsmOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(InlineAsmOp op, InlineAsmOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> types;
    if (op.getNumResults()) {
      types.push_back(
          getTypeConverter()->packFunctionResults(op->getResultTypes()));
      if (!types.back())
        return failure();
    }

    auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
        op.getLoc(), types,
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands()),
        cast<StringAttr>(adaptor.getAssembly()),
        cast<StringAttr>(adaptor.getConstraints()),
        adaptor.getHasSideEffectsAttr(), adaptor.getIsStackAlignedAttr(),
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        adaptor.getOperandAttrsAttr());

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, asmOp);
      return success();
    }
    // Unpack the results.
    SmallVector<Value> results;
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), asmOp.getResult(0), i));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicCmpXchg
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicCmpXchg
    : public ConvertPOPToLLVMPattern<AtomicCmpXchgOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicCmpXchgOp op, AtomicCmpXchgOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
        op, adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal(),
        getAtomicOrdering(op.getSuccessOrdering()),
        getAtomicOrdering(op.getFailureOrdering()),
        adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                               : StringRef(),
        resolveAlignment(adaptor));
    return success();
  }

  static unsigned resolveAlignment(AtomicCmpXchgOpAdaptor adaptor) {
    if (auto alignment = adaptor.getAlignment())
      return cast<IntegerAttr>(*alignment).getInt();

    // If alignment is not set on the op, use the alignment of the pointer.
    Value ptr = adaptor.getPtr();
    if (!ptr.getDefiningOp()->hasAttr("alignment"))
      return 0;

    return cast<IntegerAttr>(ptr.getDefiningOp()->getAttr("alignment"))
        .getInt();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicRMW
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicRMW : public ConvertPOPToLLVMPattern<AtomicRMWOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicRMWOp op, AtomicRMWOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *cast<SIMDType>(op.getType()).getResolvedDType();
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        op, getAtomicBinOp(dtype, adaptor.getBinOp()), adaptor.getPtr(),
        adaptor.getVal(), getAtomicOrdering(op.getOrdering()),
        adaptor.getSyncscope() ? cast<StringAttr>(*adaptor.getSyncscope())
                               : StringRef());
    return success();
  }

private:
  static LLVM::AtomicBinOp getAtomicBinOp(KGENDType dtype, AtomicBinOp binOp) {
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
    llvm_unreachable("unknown atomic ordering");
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFence
//===----------------------------------------------------------------------===//

class ConvertPOPFence : public ConvertPOPToLLVMPattern<FenceOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(FenceOp op, FenceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, getAtomicOrdering(adaptor.getOrdering()),
        adaptor.getSyncscopeAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStringAddress
//===----------------------------------------------------------------------===//

struct ConvertPOPStringAddress
    : public ConvertPOPToLLVMPattern<StringAddressOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StringAddressOp op, StringAddressOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // The first operand is a !kgen.string lowered to
    // !llvm.struct<(ptr<i8>, index)>, grab the the first field: the address
    // of the string.
    Value extractedAddr =
        b.create<LLVM::ExtractValueOp>(adaptor.getOperands().front(), 0);
    rewriter.replaceOp(op, extractedAddr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStringSize
//===----------------------------------------------------------------------===//

struct ConvertPOPStringSize : public ConvertPOPToLLVMPattern<StringSizeOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(StringSizeOp op, StringSizeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // The first operand is a !kgen.string lowered to
    // !llvm.struct<(ptr<i8>, index)>, grab the the second field: the size
    // of the string.
    Value extractedAddr =
        b.create<LLVM::ExtractValueOp>(adaptor.getOperands().front(), 1);
    rewriter.replaceOp(op, extractedAddr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPDTypeToUI8
//===----------------------------------------------------------------------===//

struct ConvertPOPDTypeToUI8 : public ConvertPOPToLLVMPattern<DTypeToUI8> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(DTypeToUI8 op, DTypeToUI8Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, type, adaptor.getDType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPDTypeFromUI8
//===----------------------------------------------------------------------===//

struct ConvertPOPDTypeFromUI8 : public ConvertPOPToLLVMPattern<DTypeFromUI8> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(DTypeFromUI8 op, DTypeFromUI8Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCallLLVMIntrinsic
//===----------------------------------------------------------------------===//

struct ConvertPOPCallLLVMIntrinsic
    : public ConvertPOPToLLVMPattern<CallLLVMIntrinsicOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallLLVMIntrinsicOp op, CallLLVMIntrinsicOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, types, cast<StringAttr>(op.getIntrin()),
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands()),
        convertFastmathFlags(op.getFastmathFlags(), rewriter));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPPointerBitcast
//===----------------------------------------------------------------------===//

struct ConvertPOPPointerBitcast
    : public ConvertPOPToLLVMPattern<PointerBitcastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(PointerBitcastOp op, PointerBitcastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy)
      return failure();

    // The LLVMPointerType doesn't maintain an element type, just an address
    // space.  Insert an address space cast if needed.
    auto srcVal = adaptor.getOperands()[0];
    if (srcVal.getType() != resultTy)
      rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(op, resultTy, srcVal);
    else
      rewriter.replaceOp(op, srcVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionBitcast
//===----------------------------------------------------------------------===//

struct ConvertPOPUnionBitcast : public ConvertPOPToLLVMPattern<UnionBitcastOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionBitcastOp op,
                                UnionBitcastOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    b.replaceOp(op, adaptor.getValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionWrap
//===----------------------------------------------------------------------===//

struct ConvertPOPUnionWrap : public ConvertPOPToLLVMPattern<UnionWrapOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionWrapOp op, UnionWrapOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    auto variantType =
        dyn_cast_or_null<LLVM::LLVMArrayType>(convertType(op.getType()));
    if (!variantType)
      return failure();

    VariantHelper helper(b, op.getLoc(), *getTypeConverter());
    Value result = helper.materializeLLVMUnion(variantType, adaptor.getValue());
    if (!result)
      return failure();
    b.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPUnionUnwrap
//===----------------------------------------------------------------------===//

struct ConvertPOPUnionUnwrap : public ConvertPOPToLLVMPattern<UnionUnwrapOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(UnionUnwrapOp op, UnionUnwrapOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Type valueType = convertType(op.getType());
    if (!valueType)
      return failure();
    auto contentType = cast<LLVM::LLVMArrayType>(adaptor.getValue().getType());

    SmallVector<Value> storageValues;
    for (unsigned i = 0, e = contentType.getNumElements(); i != e; ++i) {
      storageValues.push_back(
          b.create<LLVM::ExtractValueOp>(op.getLoc(), adaptor.getValue(), i));
    }

    VariantHelper helper(b, op.getLoc(), *getTypeConverter());
    ArrayRef<Value>::iterator valueIt = storageValues.begin();
    unsigned storageOffset = 0;
    unsigned offset = 0;
    Value result =
        helper.walkAndExtractVariant(valueIt, storageOffset, offset, valueType);

    b.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertPOPAnd = mlir::OneToOneConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using ConvertPOPOr = mlir::OneToOneConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using ConvertPOPXOr = mlir::OneToOneConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;
using ConvertPOPSIMDAnd =
    mlir::OneToOneConvertToLLVMPattern<SIMDAndOp, LLVM::AndOp>;
using ConvertPOPSIMDOr =
    mlir::OneToOneConvertToLLVMPattern<SIMDOrOp, LLVM::OrOp>;
using ConvertPOPSIMDXOr =
    mlir::OneToOneConvertToLLVMPattern<SIMDXOrOp, LLVM::XOrOp>;
using ConvertPOPAdd =
    OneToOneFloatOrIntConversion<AddOp, LLVM::FAddOp, LLVM::AddOp>;
using ConvertPOPSub =
    OneToOneFloatOrIntConversion<SubOp, LLVM::FSubOp, LLVM::SubOp>;
using ConvertPOPMul =
    OneToOneFloatOrIntConversion<MulOp, LLVM::FMulOp, LLVM::MulOp>;
using ConvertPOPDiv = OneToOneFloatOrIntConversion<DivOp, LLVM::FDivOp,
                                                   LLVM::SDivOp, LLVM::UDivOp>;
using ConvertPOPRem = OneToOneFloatOrIntConversion<RemOp, LLVM::FRemOp,
                                                   LLVM::SRemOp, LLVM::URemOp>;
using ConvertPOPMax = OneToOneFloatOrIntConversion<MaxOp, LLVM::MaxNumOp,
                                                   LLVM::SMaxOp, LLVM::UMaxOp>;
using ConvertPOPMin = OneToOneFloatOrIntConversion<MinOp, LLVM::MinNumOp,
                                                   LLVM::SMinOp, LLVM::UMinOp>;
using ConvertPOPBitcast =
    mlir::OneToOneConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using ConvertPOPShl = mlir::OneToOneConvertToLLVMPattern<ShlOp, LLVM::ShlOp>;
using ConvertPOPPointerToIndex =
    mlir::OneToOneConvertToLLVMPattern<PointerToIndexOp, LLVM::PtrToIntOp>;

//===----------------------------------------------------------------------===//
// ConvertNVVMWGMAMMAAsync
//===----------------------------------------------------------------------===//

static FailureOr<NVVM::WGMMATypesAttr> getMMAType(MLIRContext *ctx, Type type) {
  if (isa<Float8E4M3Type, Float8E4M3FNType>(type))
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::e4m3);
  if (isa<Float8E5M2Type>(type))
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::e5m2);
  if (type.isBF16())
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::bf16);
  if (type.isF16())
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::f16);
  if (type.isF32())
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::f32);
  if (type.isTF32())
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::tf32);
  if (type.isSignedInteger(8))
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::s8);
  if (type.isUnsignedInteger(8))
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::u8);
  if (type.isSignedInteger(32))
    return NVVM::WGMMATypesAttr::get(ctx, NVVM::WGMMATypes::s32);
  return failure();
}

static FailureOr<NVVM::MMALayoutAttr> getMMALayout(MLIRContext *ctx,
                                                   StringRef layoutStr) {
  if (layoutStr == "row")
    return NVVM::MMALayoutAttr::get(ctx, NVVM::MMALayout::row);
  if (layoutStr == "col")
    return NVVM::MMALayoutAttr::get(ctx, NVVM::MMALayout::col);
  return failure();
}

// Converts pop.nvvm.wgmma.mma_async to nvvm.wgmma.mma_async operation.
struct ConvertPoPNVVMWGMAMMAAsync
    : public ConvertPOPToLLVMPattern<NVVMWGMAMMAAsyncOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(NVVMWGMAMMAAsyncOp mmaOp,
                                NVVMWGMAMMAAsyncOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {

    MLIRContext *ctx = mmaOp.getContext();
    Location loc = mmaOp->getLoc();

    int64_t shapeM = cast<IntegerAttr>(mmaOp.getShapeM()).getInt();
    int64_t shapeN = cast<IntegerAttr>(mmaOp.getShapeN()).getInt();
    int64_t shapeK = cast<IntegerAttr>(mmaOp.getShapeK()).getInt();

    auto vecType = cast<VectorType>(adaptor.getRegC().getType());

    auto regStructType = LLVM::LLVMStructType::getLiteral(
        ctx,
        SmallVector<Type>(vecType.getNumElements(), vecType.getElementType()));

    Value inputOperand = b.create<LLVM::UndefOp>(loc, regStructType);

    // Insert elements in the struct
    for (int i = 0, e = vecType.getNumElements(); i < e; ++i) {
      Value idx = b.create<LLVM::ConstantOp>(mmaOp.getLoc(), b.getI32Type(), i);
      Value element = b.create<LLVM::ExtractElementOp>(mmaOp.getLoc(),
                                                       adaptor.getRegC(), idx);
      inputOperand =
          b.create<LLVM::InsertValueOp>(loc, inputOperand, element, i);
    }

    int64_t scaleD = resolveScaleOut(mmaOp.getScaleD());
    int64_t scaleA = resolveScaleIn(mmaOp.getScaleA());
    int64_t scaleB = resolveScaleIn(mmaOp.getScaleB());

    auto scaleDAttr = NVVM::WGMMAScaleOutAttr::get(
        ctx,
        scaleD == 0 ? NVVM::WGMMAScaleOut::zero : NVVM::WGMMAScaleOut::one);
    auto scaleAAttr = NVVM::WGMMAScaleInAttr::get(
        ctx, scaleA == -1 ? NVVM::WGMMAScaleIn::neg : NVVM::WGMMAScaleIn::one);
    auto scaleBAttr = NVVM::WGMMAScaleInAttr::get(
        ctx, scaleB == -1 ? NVVM::WGMMAScaleIn::neg : NVVM::WGMMAScaleIn::one);
    auto overflowAttr =
        NVVM::MMAIntOverflowAttr::get(ctx, NVVM::MMAIntOverflow::wrapped);

    FailureOr<NVVM::WGMMATypesAttr> typeA = getMMAType(ctx, mmaOp.getTypeA());
    FailureOr<NVVM::WGMMATypesAttr> typeB = getMMAType(ctx, mmaOp.getTypeB());
    FailureOr<NVVM::WGMMATypesAttr> typeC = getMMAType(ctx, mmaOp.getTypeC());

    assert((!failed(typeA) || !failed(typeB) || !failed(typeC)) &&
           "Unsupported operand types");

    FailureOr<NVVM::MMALayoutAttr> layoutA =
        getMMALayout(ctx, cast<StringAttr>(adaptor.getLayoutA()));
    FailureOr<NVVM::MMALayoutAttr> layoutB =
        getMMALayout(ctx, cast<StringAttr>(adaptor.getLayoutB()));

    assert((!failed(layoutA) || !failed(layoutB)) &&
           "Unsupported operand layouts");

    Value descA = mmaOp.getDescriptorA();
    Value descB = mmaOp.getDescriptorB();

    auto instShape = NVVM::MMAShapeAttr::get(ctx, shapeM, shapeN, shapeK);
    Value resStruct = b.create<NVVM::WgmmaMmaAsyncOp>(
        mmaOp.getLoc(), inputOperand.getType(), inputOperand, descA, descB,
        instShape, typeA.value(), typeB.value(), typeC.value(), scaleDAttr,
        scaleAAttr, scaleBAttr, layoutA.value(), layoutB.value(), overflowAttr);

    Value result = b.create<LLVM::UndefOp>(mmaOp.getLoc(), vecType);
    for (int i = 0, e = vecType.getNumElements(); i < e; ++i) {
      auto idx = b.create<LLVM::ConstantOp>(mmaOp.getLoc(), b.getI32Type(), i);
      auto val = b.create<LLVM::ExtractValueOp>(loc, resStruct, i);
      result = b.create<LLVM::InsertElementOp>(loc, result, val, idx);
    }

    b.replaceOp(mmaOp, result);

    return success();
  }

private:
  int64_t resolveScaleOut(TypedAttr scaleOutAttr) const {
    int64_t scaleOut = cast<IntegerAttr>(scaleOutAttr).getInt();
    assert(scaleOut == 0 || scaleOut == 1 && "Invalid scale out value");
    return scaleOut;
  }

  int64_t resolveScaleIn(TypedAttr scaleInAttr) const {
    int64_t scaleIn = cast<IntegerAttr>(scaleInAttr).getInt();
    assert(scaleIn == -1 || scaleIn == 1 && "Invalid scale in value");
    return scaleIn;
  }
};

struct ConvertPoPNVVMWGMAMMAAsyncInlineArray
    : public ConvertPOPToLLVMPattern<NVVMWGMAMMAAsyncOpInlineArray> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(NVVMWGMAMMAAsyncOpInlineArray mmaOp,
                                NVVMWGMAMMAAsyncOpInlineArrayAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {

    MLIRContext *ctx = mmaOp.getContext();
    Location loc = mmaOp->getLoc();

    int64_t shapeM = cast<IntegerAttr>(mmaOp.getShapeM()).getInt();
    int64_t shapeN = cast<IntegerAttr>(mmaOp.getShapeN()).getInt();
    int64_t shapeK = cast<IntegerAttr>(mmaOp.getShapeK()).getInt();

    Type elementType = convertType(mmaOp.getRegC().getType().getElementType());
    if (!elementType)
      return failure();
    std::optional<int64_t> numElements =
        mmaOp.getRegC().getType().getResolvedSize();
    if (!numElements)
      return failure();

    auto regStructType = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(*numElements, elementType));

    Value inputOperand = b.create<LLVM::UndefOp>(loc, regStructType);

    // Insert elements in the struct
    for (int i = 0, e = *numElements; i < e; ++i) {
      Value element =
          b.create<LLVM::ExtractValueOp>(mmaOp.getLoc(), adaptor.getRegC(), i);
      inputOperand =
          b.create<LLVM::InsertValueOp>(loc, inputOperand, element, i);
    }

    int64_t scaleD = resolveScaleOut(mmaOp.getScaleD());
    int64_t scaleA = resolveScaleIn(mmaOp.getScaleA());
    int64_t scaleB = resolveScaleIn(mmaOp.getScaleB());

    auto scaleDAttr = NVVM::WGMMAScaleOutAttr::get(
        ctx,
        scaleD == 0 ? NVVM::WGMMAScaleOut::zero : NVVM::WGMMAScaleOut::one);
    auto scaleAAttr = NVVM::WGMMAScaleInAttr::get(
        ctx, scaleA == -1 ? NVVM::WGMMAScaleIn::neg : NVVM::WGMMAScaleIn::one);
    auto scaleBAttr = NVVM::WGMMAScaleInAttr::get(
        ctx, scaleB == -1 ? NVVM::WGMMAScaleIn::neg : NVVM::WGMMAScaleIn::one);
    auto overflowAttr =
        NVVM::MMAIntOverflowAttr::get(ctx, NVVM::MMAIntOverflow::wrapped);

    FailureOr<NVVM::WGMMATypesAttr> typeA = getMMAType(ctx, mmaOp.getTypeA());
    FailureOr<NVVM::WGMMATypesAttr> typeB = getMMAType(ctx, mmaOp.getTypeB());
    FailureOr<NVVM::WGMMATypesAttr> typeC = getMMAType(ctx, mmaOp.getTypeC());

    assert((!failed(typeA) || !failed(typeB) || !failed(typeC)) &&
           "Unsupported operand types");

    FailureOr<NVVM::MMALayoutAttr> layoutA =
        getMMALayout(ctx, cast<StringAttr>(adaptor.getLayoutA()));
    FailureOr<NVVM::MMALayoutAttr> layoutB =
        getMMALayout(ctx, cast<StringAttr>(adaptor.getLayoutB()));

    assert((!failed(layoutA) || !failed(layoutB)) &&
           "Unsupported operand layouts");

    Value descA = mmaOp.getDescriptorA();
    Value descB = mmaOp.getDescriptorB();

    auto instShape = NVVM::MMAShapeAttr::get(ctx, shapeM, shapeN, shapeK);
    Value resStruct = b.create<NVVM::WgmmaMmaAsyncOp>(
        mmaOp.getLoc(), inputOperand.getType(), inputOperand, descA, descB,
        instShape, typeA.value(), typeB.value(), typeC.value(), scaleDAttr,
        scaleAAttr, scaleBAttr, layoutA.value(), layoutB.value(), overflowAttr);

    auto arrayType = convertType(mmaOp.getRegC().getType());
    Value resultArray = b.create<LLVM::UndefOp>(mmaOp.getLoc(), arrayType);

    for (int i = 0, e = *numElements; i < e; ++i) {
      auto val = b.create<LLVM::ExtractValueOp>(loc, resStruct, i);
      resultArray = b.create<LLVM::InsertValueOp>(loc, resultArray, val, i);
    }

    b.replaceOp(mmaOp, resultArray);

    return success();
  }

private:
  int64_t resolveScaleOut(TypedAttr scaleOutAttr) const {
    int64_t scaleOut = cast<IntegerAttr>(scaleOutAttr).getInt();
    assert(scaleOut == 0 || scaleOut == 1 && "Invalid scale out value");
    return scaleOut;
  }

  int64_t resolveScaleIn(TypedAttr scaleInAttr) const {
    int64_t scaleIn = cast<IntegerAttr>(scaleInAttr).getInt();
    assert(scaleIn == -1 || scaleIn == 1 && "Invalid scale in value");
    return scaleIn;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populatePOPToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertPOPAdd,
      ConvertPOPAnd,
      ConvertPOPArrayCreate,
      ConvertPOPArrayGEP,
      ConvertPOPArrayGet,
      ConvertPOPArrayRepeat,
      ConvertPOPArrayReplace,
      ConvertPOPAtomicCmpXchg,
      ConvertPOPAtomicRMW,
      ConvertPOPBitcast,
      ConvertPOPCallLLVMIntrinsic,
      ConvertPOPCast,
      ConvertPOPCastFromBuiltin,
      ConvertPOPCastToBuiltin,
      ConvertPOPCmp,
      ConvertPOPDiv,
      ConvertPOPDTypeFromUI8,
      ConvertPOPDTypeToUI8,
      ConvertPOPFence,
      ConvertPOPFMA,
      ConvertPOPInlineAsm,
      ConvertPOPLoad,
      ConvertPOPMax,
      ConvertPOPMin,
      ConvertPOPMul,
      ConvertPOPNeg,
      ConvertPOPOffset,
      ConvertPOPOr,
      ConvertPOPPointerBitcast,
      ConvertPOPPointerToIndex,
      ConvertPOPRem,
      ConvertPOPSelect,
      ConvertPOPShl,
      ConvertPOPShr,
      ConvertPOPSIMDAnd,
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDOr,
      ConvertPOPSIMDSelect,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPSIMDXOr,
      ConvertPOPStore,
      ConvertPOPStringAddress,
      ConvertPOPStringSize,
      ConvertPOPSub,
      ConvertPOPUnionBitcast,
      ConvertPOPUnionUnwrap,
      ConvertPOPUnionWrap,
      ConvertPOPVariadicGet,
      ConvertPOPVariadicSize,
      ConvertPOPXOr,
      ConvertPoPNVVMWGMAMMAAsync,
      ConvertPoPNVVMWGMAMMAAsyncInlineArray
      // clang-format on
      >(typeConverter);
}

//===----------------------------------------------------------------------===//
// LowerPOPToLLVMPass
//===----------------------------------------------------------------------===//

namespace {
struct LowerPOPToLLVMPass
    : public KGEN::impl::LowerPOPToLLVMBase<LowerPOPToLLVMPass> {
  using LowerPOPToLLVMBase::LowerPOPToLLVMBase;

  void runOnOperation() override;

  /// Verify that the operation is a function and has no nested CFGs.
  FailureOr<mlir::FunctionOpInterface> validateOperation();
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

void LowerPOPToLLVMPass::runOnOperation() {
  FailureOr<mlir::FunctionOpInterface> func = validateOperation();
  if (failed(func))
    return signalPassFailure();

  // If the function body is empty, return.
  if (func->getFunctionBody().empty())
    return;

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<POPDialect>();
  target.addIllegalDialect<mlir::index::IndexDialect>();
  target.addLegalDialect<DebugInfo::DebugInfoDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // These ops are handled by other passes.
  target.addLegalOp<GlobalAllocOp>();
  target.addLegalOp<GlobalConstantOp>();
  target.addLegalOp<GlobalAddressOp>();
  target.addLegalOp<ExternalCallOp>();
  target.addLegalOp<ExternPointerSymbolOp>();
  target.addLegalOp<AlignedAllocOp>();
  target.addLegalOp<AlignedFreeOp>();

  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(*func);
  if (!targetInfo) {
    mlir::emitError(func->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);
  mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateNVVMToLLVMConversionPatterns(patterns);
  patterns.insert<ConvertPOPStackAllocation, ConvertPOPVariadicCreate,
                  ConvertPOPVariadicSplat, ConvertPOPStackAllocLifetimeStart,
                  ConvertPOPStackAllocLifetimeEnd>(typeConverter, targetInfo);

  if (failed(mlir::applyPartialConversion(*func, target, std::move(patterns))))
    return signalPassFailure();
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPExternalCall
//===----------------------------------------------------------------------===//

/// Expand one level of struct type from any operand types, these come from
/// !kgen.pack.
static SmallVector<Type> expandOperandTypes(TypeRange types) {
  SmallVector<Type> operandTypes;
  operandTypes.reserve(types.size());
  for (auto type : types) {
    if (auto structTy = dyn_cast<StructType>(type)) {
      operandTypes.append(structTy.getElementTypes().begin(),
                          structTy.getElementTypes().end());
    } else {
      operandTypes.push_back(type);
    }
  }
  return operandTypes;
}

/// Lower an external call. Add the callee to the symbol table.
struct ConvertPOPExternalCall : public ConvertSymbolOpToLLVM<ExternalCallOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(ExternalCallOp op, ExternalCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<FunctionType> funcType = op.getVariadicType();
    if (!funcType) {
      funcType = rewriter.getFunctionType(
          expandOperandTypes(op.getOperandTypes()), op.getResultTypes());
    }
    TypeConverter::SignatureConversion conversion(funcType->getNumInputs());
    Type signature = getTypeConverter()->convertFunctionSignature(
        *funcType, op.getVariadicType().has_value(),
        getTypeConverter()->getOptions().useBarePtrCallConv, conversion);

    // Get the passthrough attributes. Set the target passthrough attributes
    // early because all functions will have them.
    mlir::ArrayAttr passthrough = attachTargetPassthroughAttrs(
        rewriter, getTypeConverter()->getTarget(), op.getFuncAttrsAttr());
    mlir::ArrayAttr argAttrs = op.getArgAttrsAttr();
    mlir::ArrayAttr resAttrs = op.getResAttrsAttr();
    auto memory = dyn_cast_or_null<LLVM::MemoryEffectsAttr>(op.getMemoryAttr());

    // Lookup an existing function.
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(op.getCallee().getValue());
    if (func && func.getFunctionType() != signature) {
      return mlir::emitError(op.getLoc(),
                             "existing function with conflicting signature")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }
    if (func &&
        std::make_tuple(func.getPassthroughAttr(), func.getArgAttrsAttr(),
                        func.getResAttrsAttr(), func.getMemoryEffectsAttr()) !=
            std::make_tuple(passthrough, argAttrs, resAttrs, memory)) {
      return mlir::emitError(op.getLoc(),
                             "existing function with conflicting attributes")
                 .attachNote(func.getLoc())
             << "see function declaration here";
    }

    // Create the function declaration if necessary.
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();
      func = rewriter.create<LLVM::LLVMFuncOp>(
          mlir::UnknownLoc::get(getContext()), op.getCallee(), signature);
      func.setPassthroughAttr(passthrough);
      if (argAttrs)
        func.setArgAttrsAttr(argAttrs);
      if (resAttrs)
        func.setResAttrsAttr(resAttrs);
      if (memory)
        func.setMemoryEffectsAttr(memory);
      symtab.insert(func);
    }

    LLVM::CallOp call = createLLVMCall(
        rewriter, op.getLoc(), func,
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands()));
    replaceCallWithLLVMCall(rewriter, op, call);
    return success();
  }
};

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
                                ConversionPatternRewriter &b) const override {
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
      func = b.create<LLVM::LLVMFuncOp>(mlir::UnknownLoc::get(getContext()),
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
                                ConversionPatternRewriter &b) const override {
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
      func = b.create<LLVM::LLVMFuncOp>(mlir::UnknownLoc::get(getContext()),
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

    Value ptr = b.create<LLVM::BitcastOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(getContext()),
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

  LogicalResult matchAndRewrite(GlobalAllocOp op, GlobalAllocOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Set the alignment if specified. Otherwise use the natural alignment.
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto elementType = typeConverter->convertType(kgenPtrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), kgenPtrType, op.getAlignmentAttr());

    // Set the address space if specified.
    unsigned addrSpace = 0;
    if (auto addrSpaceAttr =
            cast_or_null<IntegerAttr>(op.getType().getAddressSpace()))
      addrSpace = addrSpaceAttr.getInt();

    // (HACK) Add a postfix to the name here so that we can identify
    // this type of variables in the llvm module.
    // This is a workaround to LLVM MLIR lowering doesn't allow
    // GlobalValues to have passthrough metadata.

    std::string name = cast<StringAttr>(adaptor.getName()).str();
    if (op.getMemoryType() == POP::GlobalAllocAddressSpace::GPU_SHARED)
      name += "._gpu_shared_mem";

    // Create the global.
    b.clearInsertionPoint();
    auto global = b.create<LLVM::GlobalOp>(
        op.getLoc(),
        LLVM::LLVMArrayType::get(elementType,
                                 cast<IntegerAttr>(op.getCount()).getInt()),
        /*isConstant=*/false, LLVM::Linkage::Internal, name,
        /*value=*/Attribute(), alignment, addrSpace);
    symtab.insert(global);

    // Replace the alloc op with an `addressof`.
    b.setInsertionPoint(op);
    auto opaquePtrType = LLVM::LLVMPointerType::get(getContext(), addrSpace);
    auto ptr = b.create<LLVM::AddressOfOp>(op.getLoc(), global);
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
  auto func = b.create<LLVM::LLVMFuncOp>(
      UnknownLoc::get(b.getContext()), name,
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
  b.create<LLVM::ReturnOp>(func.getLoc(),
                           body->addArgument(ptrType, func.getLoc()));

  return func;
}

struct ConvertPOPNoAliasPointerCast
    : ConvertSymbolOpToLLVM<NoAliasPointerCastOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(NoAliasPointerCastOp op,
                                NoAliasPointerCastOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
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
      mlir::LLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        constants(constants) {}

  LogicalResult
  matchAndRewrite(GlobalConstantOp op, GlobalConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

      LLVM::GlobalOp global = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), elementType, true, LLVM::Linkage::Internal,
          "global_constant", Attribute(),
          getAlignment(getTypeConverter(), kgenPtrType,
                       adaptor.getAlignmentAttr()));
      // Emit the constant using an initializer region.
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
      b.setInsertionPointToStart(global.getBody());
      ErrorOr<Value> value =
          convertParameterToLLVM(b, *getTypeConverter(), /*imc=*/nullptr,
                                 /*scope=*/nullptr, op.getValue());
      if (value.isError()) {
        b.emitError(value.getError());
        return failure();
      }
      b.create<LLVM::ReturnOp>(value.get());

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
                                ConversionPatternRewriter &b) const override {
    int64_t addressSpace =
        cast<IntegerAttr>(op.getResSymbol().getType().getAddressSpace())
            .getInt();
    Type resType = convertType(op.getResSymbol().getType().getElementType());
    unsigned align = getAlignment(
        getTypeConverter(), op.getResSymbol().getType(), op.getAlignmentAttr());

    b.clearInsertionPoint();
    auto global = b.create<LLVM::GlobalOp>(
        op.getLoc(), resType, /*constant=*/false, LLVM::Linkage::External,
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

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<DebugInfo::DebugInfoDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Convert external calls.
  target.addIllegalOp<GlobalAllocOp, ExternalCallOp, ExternPointerSymbolOp>();
  patterns.insert<ConvertPOPGlobalAlloc, ConvertPOPExternalCall,
                  ConvertExternPointerSymbol, ConvertPOPAlignedAlloc,
                  ConvertPOPAlignedFree, ConvertPOPNoAliasPointerCast>(
      typeConverter, symtab);

  // Convert global constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> constants;
  target.addIllegalOp<GlobalConstantOp>();
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter);

  // pop.compiler.* are all illegal.
  target.addIllegalOp<CompilerGlobalLoadOp, CompilerGlobalStoreOp>();

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
