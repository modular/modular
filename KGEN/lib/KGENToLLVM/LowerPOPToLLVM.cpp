//===- LowerPOPToLLVM.cpp -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

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
// ConvertPOPStackAllocation
//===----------------------------------------------------------------------===//

/// A `pop.stack_allocation` is lowered by converting it to an `llvm.alloca`
/// with lifetime markers and hoisting it to the top of the enclosing function.
class ConvertPOPStackAllocation
    : public mlir::ConvertOpToLLVMPattern<StackAllocationOp> {
public:
  explicit ConvertPOPStackAllocation(mlir::LLVMTypeConverter &typeConverter,
                                     Block *body)
      : ConvertOpToLLVMPattern(typeConverter), body(body) {}

  LogicalResult
  matchAndRewrite(StackAllocationOp op, StackAllocationOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  /// The enclosing function body.
  Block *body;
};

LogicalResult ConvertPOPStackAllocation::matchAndRewrite(
    StackAllocationOp op, StackAllocationOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  IntegerAttr size =
      rewriter.getI64IntegerAttr(op.getSize().cast<IntegerAttr>().getInt());
  Type ptrType = getTypeConverter()->convertType(op.getType());
  if (!ptrType)
    return op.emitOpError("could not lower pointer element type");

  // Hoist the alloca to the top of the enclosing function body.
  rewriter.setInsertionPointToStart(body);
  Value sizeVal = rewriter.create<LLVM::ConstantOp>(op.getLoc(), size);
  Value ptr = rewriter.create<LLVM::AllocaOp>(op.getLoc(), ptrType, sizeVal);

  // Insert lifetime markers starting from the op to the end of its block.
  rewriter.setInsertionPoint(op);
  rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), size, ptr);
  rewriter.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
  rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), size, ptr);
  rewriter.replaceOp(op, ptr);
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDReduceAdd
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDReduceAdd
    : public mlir::ConvertOpToLLVMPattern<SIMDReduceAddOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceAddOp op, SIMDReduceAddOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    Type eltType =
        adaptor.getOperand().getType().cast<VectorType>().getElementType();
    if (dtype.isInt()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_add>(
          op, eltType, adaptor.getOperand());
      return success();
    }
    // Handle the floating point case.
    // To ignore the start value, we pass in negative zero (-0.0) as it is
    // the neutral value of floating point addition.
    Value negZero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), eltType, rewriter.getFloatAttr(eltType, -0.0));
    rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fadd>(op, eltType, negZero,
                                                          adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDReduceMul
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDReduceMul
    : public mlir::ConvertOpToLLVMPattern<SIMDReduceMulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceMulOp op, SIMDReduceMulOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DType dtype = op.getType().cast<DTypeInterface>().resolveDType();
    Type eltType =
        adaptor.getOperand().getType().cast<VectorType>().getElementType();
    if (dtype.isInt()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_mul>(
          op, eltType, adaptor.getOperand());
      return success();
    }
    // Handle the floating point case.
    // To ignore the start value, one (1.0) is used, as it is the neutral
    // value of floating point multiplication.
    Value one = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), eltType, rewriter.getFloatAttr(eltType, 1.0));
    rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmul>(op, eltType, one,
                                                          adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStructConstruct
//===----------------------------------------------------------------------===//

struct ConvertPOPStructConstruct
    : mlir::ConvertOpToLLVMPattern<StructConstructOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructConstructOp op, StructConstructOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type structType = getTypeConverter()->convertType(op.getType());
    if (!structType)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "failed to convert struct type");
    Value container = rewriter.create<LLVM::UndefOp>(op.getLoc(), structType);
    for (auto &element : llvm::enumerate(adaptor.getOperands()))
      container = rewriter.create<LLVM::InsertValueOp>(
          op.getLoc(), container, element.value(), element.index());
    rewriter.replaceOp(op, container);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPReplaceElement
//===----------------------------------------------------------------------===//

struct ConvertPOPReplaceElement
    : mlir::ConvertOpToLLVMPattern<ReplaceElementOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReplaceElementOp op, ReplaceElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getContainer(), adaptor.getValue(),
        op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPGetElement
//===----------------------------------------------------------------------===//

struct ConvertPOPGetElement : mlir::ConvertOpToLLVMPattern<GetElementOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetElementOp op, GetElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getContainer(), op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPTypeLowerCast
//===----------------------------------------------------------------------===//

struct ConvertPOPTypeLowerCast : mlir::ConvertOpToLLVMPattern<TypeLowerOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeLowerOp op, TypeLowerOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPTypeRaiseCast
//===----------------------------------------------------------------------===//

struct ConvertPOPTypeRaiseCast : mlir::ConvertOpToLLVMPattern<TypeRaiseOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeRaiseOp op, TypeRaiseOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
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
using ConvertPOPBitcast =
    mlir::OneToOneConvertToLLVMPattern<BitcastOp, LLVM::BitcastOp>;
using ConvertPOPPointerBitcast =
    mlir::OneToOneConvertToLLVMPattern<PointerBitcastOp, LLVM::BitcastOp>;
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
using ConvertPOPSIMDReduceMax =
    OneToOneFloatOrIntConversion<SIMDReduceMaxOp, LLVM::vector_reduce_fmax,
                                 LLVM::vector_reduce_smax,
                                 LLVM::vector_reduce_umax>;
using ConvertPOPSIMDReduceMin =
    OneToOneFloatOrIntConversion<SIMDReduceMinOp, LLVM::vector_reduce_fmin,
                                 LLVM::vector_reduce_smin,
                                 LLVM::vector_reduce_umin>;
using ConvertPOPLoad = mlir::OneToOneConvertToLLVMPattern<LoadOp, LLVM::LoadOp>;
using ConvertPOPStore =
    mlir::OneToOneConvertToLLVMPattern<StoreOp, LLVM::StoreOp>;
using ConvertPOPIndexToPointer =
    mlir::OneToOneConvertToLLVMPattern<IndexToPointerOp, LLVM::IntToPtrOp>;
using ConvertPOPPointerToIndex =
    mlir::OneToOneConvertToLLVMPattern<PointerToIndexOp, LLVM::PtrToIntOp>;

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
      ConvertPOPBitcast,
      ConvertPOPCast,
      ConvertPOPCmp,
      ConvertPOPConstant,
      ConvertPOPCopySign,
      ConvertPOPDiv,
      ConvertPOPFMA,
      ConvertPOPGetElement,
      ConvertPOPIndexToPointer,
      ConvertPOPLoad,
      ConvertPOPMax,
      ConvertPOPMin,
      ConvertPOPMul,
      ConvertPOPNeg,
      ConvertPOPOffset,
      ConvertPOPPointerBitcast,
      ConvertPOPPointerToIndex,
      ConvertPOPReplaceElement,
      ConvertPOPSelect,
      ConvertPOPShl,
      ConvertPOPShr,
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDReduceAdd,
      ConvertPOPSIMDReduceMax,
      ConvertPOPSIMDReduceMin,
      ConvertPOPSIMDReduceMul,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPStore,
      ConvertPOPStructConstruct,
      ConvertPOPSub,
      ConvertPOPTypeLowerCast,
      ConvertPOPTypeRaiseCast
      // clang-format on
      >(typeConverter);
}

//===----------------------------------------------------------------------===//
// LowerPOPToLLVMPass
//===----------------------------------------------------------------------===//

namespace {
struct LowerPOPToLLVMPass : public LowerPOPToLLVMBase<LowerPOPToLLVMPass> {
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
  diag.attachNote() << "try running it before lower-scf-to-llvm";
  return diag;
}

void LowerPOPToLLVMPass::runOnOperation() {
  FailureOr<mlir::FunctionOpInterface> func = validateOperation();
  if (failed(func))
    return signalPassFailure();

  // If the function body is empty, return.
  if (func->getBody().empty())
    return;

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<POPDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ExternalCallOp>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(func->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);
  patterns.insert<ConvertPOPStackAllocation>(typeConverter,
                                             &func->getBody().front());

  if (failed(mlir::applyPartialConversion(*func, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createLowerPOPToLLVMPass() {
  return std::make_unique<LowerPOPToLLVMPass>();
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPExternalCall
//===----------------------------------------------------------------------===//

/// Lower an external call. Add the callee to the symbol table.
class ConvertPOPExternalCall
    : public mlir::ConvertOpToLLVMPattern<ExternalCallOp> {
public:
  ConvertPOPExternalCall(SymbolTable &symtab,
                         mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter), symtab(symtab) {}

  LogicalResult
  matchAndRewrite(ExternalCallOp op, ExternalCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Optional<FunctionType> funcType = op.getVariadicType();
    if (!funcType)
      funcType =
          rewriter.getFunctionType(op.getOperandTypes(), op.getResultTypes());
    TypeConverter::SignatureConversion conversion(funcType->getNumInputs());
    Type signature = getTypeConverter()->convertFunctionSignature(
        *funcType, op.getVariadicType().has_value(), conversion);

    // Lookup an existing function.
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(op.getFuncAttr().getAttr());
    if (func && func.getFunctionType() != signature)
      return op.emitError("existing function with conflicting signature")
                 .attachNote(func.getLoc())
             << "see function declaration here";

    // Create the function declaration if necessary.
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();
      func = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), op.getFunc(),
                                               signature);
      symtab.insert(func);
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, adaptor.getOperands());
    return success();
  }

private:
  /// The symbol table.
  SymbolTable &symtab;
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalConstant
//===----------------------------------------------------------------------===//

/// Lower a global constant. Unique the constant value.
class ConvertPOPGlobalConstant
    : public mlir::ConvertOpToLLVMPattern<GlobalConstantOp> {
public:
  ConvertPOPGlobalConstant(SymbolTable &symtab,
                           DenseMap<TypedAttr, LLVM::GlobalOp> &constants,
                           mlir::LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMPattern(typeConverter), symtab(symtab),
        constants(constants) {}

  LogicalResult
  matchAndRewrite(GlobalConstantOp op, GlobalConstantOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Unique the constant.
    auto [it, inserted] = constants.try_emplace(op.getValue(), nullptr);
    if (inserted) {
      // If the constant doesn't exist, create it and insert it in the module.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();
      Type type = getTypeConverter()->convertType(op.getType());
      if (!type)
        return rewriter.notifyMatchFailure(
            op.getLoc(), "failed to convert constant result type");
      it->second = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), type.cast<LLVM::LLVMPointerType>().getElementType(),
          true, LLVM::Linkage::Internal, "global_constant", op.getValue());
      symtab.insert(it->second);
    }

    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, it->second);
    return success();
  }

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// Uniqued constants.
  DenseMap<TypedAttr, LLVM::GlobalOp> &constants;
};

//===----------------------------------------------------------------------===//
// LowerGlobalPOPToLLVMPass
//===----------------------------------------------------------------------===//

class LowerGlobalPOPToLLVMPass
    : public LowerGlobalPOPToLLVMBase<LowerGlobalPOPToLLVMPass> {
public:
  void runOnOperation() override;
};

} // namespace

void LowerGlobalPOPToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable symtab(theModule);

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalOp<ExternalCallOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(theModule->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Convert external calls.
  patterns.insert<ConvertPOPExternalCall>(symtab, typeConverter);

  // Convert global constants.
  DenseMap<TypedAttr, LLVM::GlobalOp> constants;
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter);

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createLowerGlobalPOPToLLVM() {
  return std::make_unique<LowerGlobalPOPToLLVMPass>();
}
