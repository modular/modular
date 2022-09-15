//===- ConvertPOPToLLVM.cpp -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace {

//===----------------------------------------------------------------------===//
// OneToOneFloatOrIntConversion
//===----------------------------------------------------------------------===//

/// This patterns converts a scalar POP dialect operation to either an integer
/// or floating point LLVM operation one-to-one.
template <typename Op, typename FloatOp, typename SIntOp,
          typename UIntOp = SIntOp>
struct OneToOneFloatOrIntConversion : public mlir::ConvertOpToLLVMPattern<Op> {
  using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().template cast<DTypeInterface>().resolveDType();
    Type type = this->getTypeConverter()->convertType(op.getType());

    if (dtype.isInt()) {
      if (std::is_same_v<SIntOp, UIntOp> || dtype.isSInt())
        rewriter.replaceOpWithNewOp<SIntOp>(op, type, adaptor.getOperands(),
                                            op->getAttrs());
      else
        rewriter.replaceOpWithNewOp<UIntOp>(op, type, adaptor.getOperands(),
                                            op->getAttrs());
    } else {
      rewriter.replaceOpWithNewOp<FloatOp>(op, type, adaptor.getOperands(),
                                           op->getAttrs());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPNeg
//===----------------------------------------------------------------------===//

/// Convert an integer pop.neg(x) -> 0 - x
/// and float pop.neg(x) -> llvm.fneg(x)
struct ConvertPOPNeg : public mlir::ConvertOpToLLVMPattern<NegOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NegOp op, NegOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      Type type = adaptor.getOperand().getType();
      Value zero;
      if (auto vec = type.dyn_cast<VectorType>())
        zero = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), DenseIntElementsAttr::get(vec, 0));
      else
        zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type, 0);
      rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getOperand());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FNegOp>(op, adaptor.getOperand());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAbs
//===----------------------------------------------------------------------===//

/// Convert integer pop.abs x -> llvm.abs
struct ConvertPOPAbs : public mlir::ConvertOpToLLVMPattern<AbsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, AbsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      Type type = adaptor.getOperand().getType();
      auto zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getBoolAttr(false));
      rewriter.replaceOpWithNewOp<LLVM::AbsOp>(op, type, adaptor.getOperand(),
                                               zero);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FAbsOp>(op, adaptor.getOperand());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPShr
//===----------------------------------------------------------------------===//

/// Lower to `llvm.ashr` if the result dtype is signed and `llvm.lshr`
/// otherwise.
struct ConvertPOPShr : public mlir::ConvertOpToLLVMPattern<ShrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ShrOp op, ShrOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    if (dtype.isSInt())
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
struct ConvertPOPFMA : public mlir::ConvertOpToLLVMPattern<FMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FMAOp op, FMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      auto lhs = rewriter.create<LLVM::MulOp>(op.getLoc(), adaptor.getA(),
                                              adaptor.getB());
      rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, lhs, adaptor.getC());
    } else {
      rewriter.replaceOpWithNewOp<LLVM::FMAOp>(op, adaptor.getOperands());
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCmp
//===----------------------------------------------------------------------===//

class ConvertPOPCmp : public mlir::ConvertOpToLLVMPattern<CmpOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, CmpOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getLhs().getType().cast<DTypeInterface>().resolveDType();
    if (dtype.isInt()) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, getICmpPredicate(op.getPred(), dtype.isSInt()), adaptor.getLhs(),
          adaptor.getRhs());
    } else {
      Type i1Type = rewriter.getI1Type();
      if (auto simd = op.getLhs().getType().dyn_cast<SIMDType>())
        i1Type = VectorType::get(*simd.resolveSize(), i1Type);
      rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
          op, i1Type, getFCmpPredicate(op.getPred()), adaptor.getLhs(),
          adaptor.getRhs());
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
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCast
//===----------------------------------------------------------------------===//

struct ConvertPOPCast : public mlir::ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, CastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType inDType =
        op.getInput().getType().cast<DTypeInterface>().resolveDType();
    DType outDType =
        op.getOutput().getType().cast<DTypeInterface>().resolveDType();

    // Select the element-wise cast to perform. LLVM integer types are signless,
    // but the signedness semantics of the operation's input and output types
    // affect which casts are selected. `bool` is `i1`.
    StringRef opName;
    if (inDType.isBool() || inDType.isInt()) {
      if (outDType.isBool() || outDType.isInt()) {
        if (outDType.getWidthInBits() > inDType.getWidthInBits()) {
          // Sign or zero extend.
          opName = inDType.isSInt() ? LLVM::SExtOp::getOperationName()
                                    : LLVM::ZExtOp::getOperationName();
        } else if (outDType.getWidthInBits() < inDType.getWidthInBits()) {
          // Truncate.
          opName = LLVM::TruncOp::getOperationName();
        }
      } else {
        // Cast from an integer to a float.
        opName = inDType.isSInt() ? LLVM::SIToFPOp::getOperationName()
                                  : LLVM::UIToFPOp::getOperationName();
      }
    } else if (outDType.isBool() || outDType.isInt()) {
      // Cast from a float to an integer.
      opName = outDType.isSInt() ? LLVM::FPToSIOp::getOperationName()
                                 : LLVM::FPToUIOp::getOperationName();
    } else {
      if (outDType.getWidthInBits() > inDType.getWidthInBits()) {
        // Extend
        opName = LLVM::FPExtOp::getOperationName();
      } else if (outDType.getWidthInBits() < inDType.getWidthInBits()) {
        // Truncate.
        opName = LLVM::FPTruncOp::getOperationName();
      } else if (outDType != inDType) {
        // FIXME: Unclear how to cast between `bf16` and `f16`.
        return rewriter.notifyMatchFailure(
            op, "casts between 'bf16' and 'f16' unsupported");
      }
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
    state.addTypes(getTypeConverter()->convertType(op.getOutput().getType()));
    Operation *cast = rewriter.create(state);
    rewriter.replaceOp(op, cast->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDSplat
//===----------------------------------------------------------------------===//

/// Convert a SIMD splat to an `insertelement` into an `undef` and then a
/// zero-initialized `shufflevector`.
struct ConvertPOPSIMDSplat : public mlir::ConvertOpToLLVMPattern<SIMDSplatOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDSplatOp op, SIMDSplatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType().cast<SIMDType>();
    Value undef = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), getTypeConverter()->convertType(type));
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32IntegerAttr(0));
    Value vector = rewriter.create<LLVM::InsertElementOp>(
        op.getLoc(), undef, adaptor.getScalar(), zero);
    // If the vector is size 1, skip the shuffle.
    int64_t size = type.getSize().cast<IntegerAttr>().getInt();
    if (size == 1) {
      rewriter.replaceOp(op, vector);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(
          op, vector, undef, /*mask=*/SmallVector<int32_t>(size, 0));
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPOffset
//===----------------------------------------------------------------------===//

struct ConvertPOPOffset : public mlir::ConvertOpToLLVMPattern<OffsetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OffsetOp op, OffsetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, adaptor.getPtr().getType(), adaptor.getPtr(), adaptor.getIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPBufferStackAllocationOp
//===----------------------------------------------------------------------===//

struct ConvertPOPBufferStackAllocationOp
    : public mlir::ConvertOpToLLVMPattern<BufferStackAllocationOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BufferStackAllocationOp op,
                  BufferStackAllocationOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op->getContext();
    BufferDescriptorBuilder buffer(op.getResult(), op.getLoc(), rewriter,
                                   *getTypeConverter());
    DType dtype = buffer.getDType();
    Type elemType = *getMLIRTypeForDType(ctx, dtype);
    Type ptrType = LLVM::LLVMPointerType::get(elemType);

    Value size = buffer.emitGetSize(op.getResult());
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, ptrType, elemType, size);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertPOPConstant =
    mlir::OneToOneConvertToLLVMPattern<ConstantOp, LLVM::ConstantOp>;
using ConvertPOPCopySign =
    mlir::OneToOneConvertToLLVMPattern<CopySignOp, LLVM::CopySignOp>;
using ConvertPOPAdd =
    OneToOneFloatOrIntConversion<AddOp, LLVM::FAddOp, LLVM::AddOp>;
using ConvertPOPSub =
    OneToOneFloatOrIntConversion<SubOp, LLVM::FSubOp, LLVM::SubOp>;
using ConvertPOPMul =
    OneToOneFloatOrIntConversion<MulOp, LLVM::FMulOp, LLVM::MulOp>;
using ConvertPOPDiv = OneToOneFloatOrIntConversion<DivOp, LLVM::FDivOp,
                                                   LLVM::SDivOp, LLVM::UDivOp>;
using ConvertPOPMax = OneToOneFloatOrIntConversion<MaxOp, LLVM::MaxNumOp,
                                                   LLVM::SMaxOp, LLVM::UMaxOp>;
using ConvertPOPMin = OneToOneFloatOrIntConversion<MinOp, LLVM::MinNumOp,
                                                   LLVM::SMinOp, LLVM::UMinOp>;
using ConvertPOPBitCast =
    mlir::OneToOneConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using ConvertPOPShl = mlir::OneToOneConvertToLLVMPattern<ShlOp, LLVM::ShlOp>;
using ConvertPOPSelect =
    mlir::OneToOneConvertToLLVMPattern<SelectOp, LLVM::SelectOp>;
using ConvertPOPSIMDExtractElement =
    mlir::OneToOneConvertToLLVMPattern<SIMDExtractElementOp,
                                       LLVM::ExtractElementOp>;
using ConvertPOPSIMDInsertElement =
    mlir::OneToOneConvertToLLVMPattern<SIMDInsertElementOp,
                                       LLVM::InsertElementOp>;
using ConvertPOPSIMDShuffle =
    mlir::OneToOneConvertToLLVMPattern<SIMDShuffleOp, LLVM::ShuffleVectorOp>;
using ConvertPOPLoad = mlir::OneToOneConvertToLLVMPattern<LoadOp, LLVM::LoadOp>;
using ConvertPOPStore =
    mlir::OneToOneConvertToLLVMPattern<StoreOp, LLVM::StoreOp>;

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populatePOPToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertPOPAbs,
      ConvertPOPAdd,
      ConvertPOPBitCast,
      ConvertPOPBufferStackAllocationOp,
      ConvertPOPCast,
      ConvertPOPCmp,
      ConvertPOPConstant,
      ConvertPOPCopySign,
      ConvertPOPDiv,
      ConvertPOPFMA,
      ConvertPOPLoad,
      ConvertPOPMax,
      ConvertPOPMin,
      ConvertPOPMul,
      ConvertPOPNeg,
      ConvertPOPOffset,
      ConvertPOPSelect,
      ConvertPOPShl,
      ConvertPOPShr,
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPStore,
      ConvertPOPSub
      // clang-format on
      >(typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertPOPToLLVMPass
    : public ConvertPOPToLLVMBase<ConvertPOPToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertPOPToLLVMPass::runOnOperation() {
  Operation *func = getOperation();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<POPDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(func->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyPartialConversion(func, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createConvertPOPToLLVMPass() {
  return std::make_unique<ConvertPOPToLLVMPass>();
}
