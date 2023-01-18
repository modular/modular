//===----------------------------------------------------------------------===//
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
#include "Support/Compiler/SymbolTableAnalysis.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;
namespace LLVM = mlir::LLVM;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERGLOBALPOPTOLLVM
#define GEN_PASS_DEF_LOWERPOPTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

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
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = this->getTypeConverter()->convertType(op.getType());

    if (dtype.isInt() || dtype.isIndex()) {
      if (std::is_same_v<SIntOp, UIntOp> || dtype.isSInt() || dtype.isIndex())
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
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isInt() || dtype.isIndex()) {
      Type type = adaptor.getOperand().getType();
      Value zero;
      if (auto vec = dyn_cast<VectorType>(type))
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
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isUInt()) {
      rewriter.replaceOp(op, adaptor.getOperand());
    } else if (dtype.isSInt()) {
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
struct ConvertPOPFMA : public mlir::ConvertOpToLLVMPattern<FMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FMAOp op, FMAOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (dtype.isInt() || dtype.isIndex()) {
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
    KGENDType dtype = *op.getLhs().getType().getResolvedDType();
    if (dtype.isBool() || dtype.isInt() || dtype.isIndex()) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          op, getICmpPredicate(op.getPred(), dtype.isSInt()), adaptor.getLhs(),
          adaptor.getRhs());
    } else {
      Type i1Type = rewriter.getI1Type();
      if (auto simd = dyn_cast<SIMDType>(op.getLhs().getType())) {
        auto size = *simd.getResolvedSize();
        // Vectors of size 1 should remain scalars
        if (size != 1)
          i1Type = VectorType::get(size, i1Type);
      }
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
        if (outByteCount > inByteCount) {
          // Sign or zero extend.
          opName = inDType.isSInt() ? LLVM::SExtOp::getOperationName()
                                    : LLVM::ZExtOp::getOperationName();
        } else if (outByteCount < inByteCount) {
          // Truncate.
          opName = LLVM::TruncOp::getOperationName();
        }
      } else {
        // Cast from an integer to a float.
        opName = inDType.isSInt() ? LLVM::SIToFPOp::getOperationName()
                                  : LLVM::UIToFPOp::getOperationName();
      }
    } else if (outDType.isBool() || outDType.isInt() || inDType.isIndex()) {
      // Cast from a float to an integer.
      opName = outDType.isSInt() ? LLVM::FPToSIOp::getOperationName()
                                 : LLVM::FPToUIOp::getOperationName();
    } else {
      if (outByteCount > inByteCount) {
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

private:
  int64_t getDTypeSizeInBytes(KGENDType dtype) const {
    if (dtype.isIndex())
      return getTypeConverter()->getIndexTypeBitwidth() / CHAR_BIT;
    return dtype.getSizeInBytes();
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
    // If the vector is size 1, skip the shuffle.
    if (isSIMDSizeOneType(op.getType())) {
      rewriter.replaceOp(op, adaptor.getScalar());
      return success();
    }

    SIMDType simdType = op.getType();
    int64_t size = *simdType.getResolvedSize();
    Value undef = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), getTypeConverter()->convertType(simdType));
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
    : public mlir::ConvertOpToLLVMPattern<SIMDInsertElementOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDInsertElementOp op, SIMDInsertElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isSIMDSizeOneType(op.getVector().getType())) {
      // If the vector is size 1, return the value as is - it's a scalar.
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
        op, getTypeConverter()->convertType(op.getType()),
        adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDShuffle
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDShuffle
    : public mlir::ConvertOpToLLVMPattern<SIMDShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDShuffleOp op, SIMDShuffleOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto mask = cast<ListAttr>(adaptor.getMask());
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
    auto llvmVecType = LLVM::getFixedVectorType(
        *getMLIRTypeForDType(op.getType().getContext(), dtype,
                             getTypeConverter()->getIndexTypeBitwidth()),
        mask.getValues().size());
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
    : public mlir::ConvertOpToLLVMPattern<SIMDExtractElementOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDExtractElementOp op, SIMDExtractElementOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special handling for scalars
    if (isSIMDSizeOneType(op.getVector().getType())) {
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
        op, getTypeConverter()->convertType(op.getType()), adaptor.getVector(),
        adaptor.getPosition());
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

  unsigned resolveAlignment(std::optional<TypedAttr> alignment) const {
    if (!alignment)
      return 0;
    return alignment->cast<IntegerAttr>().getInt();
  }
};

LogicalResult ConvertPOPStackAllocation::matchAndRewrite(
    StackAllocationOp op, StackAllocationOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type ptrType = getTypeConverter()->convertType(op.getType());
  if (!ptrType)
    return op.emitOpError("could not lower pointer element type");

  // Hoist the alloca to the top of the enclosing function body.
  rewriter.setInsertionPointToStart(body);
  int64_t count = cast<IntegerAttr>(op.getCount()).getInt();
  Value sizeVal = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64IntegerAttr(count));
  Value ptr = rewriter.create<LLVM::AllocaOp>(
      op.getLoc(), ptrType,
      ptrType.cast<LLVM::LLVMPointerType>().getElementType(), sizeVal,
      resolveAlignment(op.getAlignment()));

  // Compute the bytecount of the allocated buffer.
  std::optional<int64_t> byteCount = DataLayoutInterface::getTypeSizeInBytes(
      TargetInfoAttr::getForHost(op.getContext()),
      cast<PointerType>(op.getType()).getResolvedElementType());
  if (!byteCount)
    return op.emitError("could not get size of pointer element size");

  // Insert lifetime markers starting from the op to the end of its block.
  rewriter.setInsertionPoint(op);
  rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), *byteCount * count, ptr);
  rewriter.setInsertionPoint(op->getBlock(), --op->getBlock()->end());
  rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), *byteCount * count, ptr);
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
    // Handle 1 element vector, i.e. scalar, case
    if (isSIMDSizeOneType(op.getOperand().getType())) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    KGENDType dtype = *op.getType().getResolvedDType();
    Type eltType =
        adaptor.getOperand().getType().cast<VectorType>().getElementType();
    if (dtype.isInt() || dtype.isIndex()) {
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
    // Handle 1 element vector, i.e. scalar, case
    if (isSIMDSizeOneType(op.getOperand().getType())) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    KGENDType dtype = *op.getType().getResolvedDType();
    Type eltType =
        adaptor.getOperand().getType().cast<VectorType>().getElementType();
    if (dtype.isInt() || dtype.isIndex()) {
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
// ConvertPOPSIMDReduceMax
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDReduceMax
    : public mlir::ConvertOpToLLVMPattern<SIMDReduceMaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceMaxOp op, SIMDReduceMaxOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Handle 1 element vector, i.e. scalar, case
    if (isSIMDSizeOneType(op.getOperand().getType())) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = getTypeConverter()->convertType(op.getType());
    if (dtype.isFloat()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmax>(
          op, type, adaptor.getOperands(), op->getAttrs());
      return success();
    }
    if (dtype.isSInt()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smax>(
          op, type, adaptor.getOperands(), op->getAttrs());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umax>(
        op, type, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDReduceMin
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDReduceMin
    : public mlir::ConvertOpToLLVMPattern<SIMDReduceMinOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDReduceMinOp op, SIMDReduceMinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Handle 1 element vector, i.e. scalar, case
    if (isSIMDSizeOneType(op.getOperand().getType())) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = getTypeConverter()->convertType(op.getType());
    if (dtype.isFloat()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_fmin>(
          op, type, adaptor.getOperands(), op->getAttrs());
      return success();
    }
    if (dtype.isSInt()) {
      rewriter.replaceOpWithNewOp<LLVM::vector_reduce_smin>(
          op, type, adaptor.getOperands(), op->getAttrs());
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::vector_reduce_umin>(
        op, type, adaptor.getOperands(), op->getAttrs());
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
// ConvertPOPStructReplace
//===----------------------------------------------------------------------===//

struct ConvertPOPStructReplace : mlir::ConvertOpToLLVMPattern<StructReplaceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructReplaceOp op, StructReplaceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getContainer(), adaptor.getValue(),
        op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStructGet
//===----------------------------------------------------------------------===//

struct ConvertPOPStructGet : mlir::ConvertOpToLLVMPattern<StructGetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StructGetOp op, StructGetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, adaptor.getContainer(), op.getIndexAttr().getInt());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStructGEP
//===----------------------------------------------------------------------===//

struct ConvertPOPStructGEP : mlir::ConvertOpToLLVMPattern<POP::StructGEPOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(POP::StructGEPOp op, POP::StructGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrType = getTypeConverter()->convertType(op.getType());
    if (!ptrType)
      return op.emitError("failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrType, adaptor.getContainer(),
        ArrayRef<LLVM::GEPArg>{
            0, static_cast<int32_t>(op.getIndexAttr().getInt())});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPArrayCreate
//===----------------------------------------------------------------------===//

struct ConvertPOPArrayCreate
    : public mlir::ConvertOpToLLVMPattern<ArrayCreateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayCreateOp op, ArrayCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
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

struct ConvertPOPArrayRepeat
    : public mlir::ConvertOpToLLVMPattern<ArrayRepeatOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayRepeatOp op, ArrayRepeatOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
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

struct ConvertPOPArrayGet : public mlir::ConvertOpToLLVMPattern<ArrayGetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

struct ConvertPOPArrayReplace
    : public mlir::ConvertOpToLLVMPattern<ArrayReplaceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

struct ConvertPOPArrayGEP : public mlir::ConvertOpToLLVMPattern<ArrayGEPOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArrayGEPOp op, ArrayGEPOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrType = getTypeConverter()->convertType(op.getType());
    if (!ptrType)
      return op.emitError("failed to convert result type");
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrType, adaptor.getArray(),
        ArrayRef<LLVM::GEPArg>{0, adaptor.getIndex()});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// getAlignment
//===----------------------------------------------------------------------===//

static unsigned getAlignment(const llvm::DataLayout &dataLayout, Type ptrType,
                             std::optional<TypedAttr> alignmentAttr = {}) {
  // If we have the alignment attribute, use it.
  if (alignmentAttr)
    return alignmentAttr->cast<IntegerAttr>().getInt();

  // Otherwise, get the preferred alignment for the type.
  llvm::LLVMContext llvmContext;
  return LLVM::TypeToLLVMIRTranslator(llvmContext)
      .getPreferredAlignment(
          ptrType.cast<LLVM::LLVMPointerType>().getElementType(), dataLayout);
}

//===----------------------------------------------------------------------===//
// ConvertPOPLoad
//===----------------------------------------------------------------------===//

struct ConvertPOPLoad : mlir::ConvertOpToLLVMPattern<LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LoadOp op, LoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, adaptor.getPtr(),
        getAlignment(getTypeConverter()->getDataLayout(),
                     adaptor.getPtr().getType(), adaptor.getAlignment()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPStore
//===----------------------------------------------------------------------===//

struct ConvertPOPStore : mlir::ConvertOpToLLVMPattern<StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getArg(), adaptor.getPtr(),
        getAlignment(getTypeConverter()->getDataLayout(),
                     adaptor.getPtr().getType(), adaptor.getAlignment()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDGather
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDGather : mlir::ConvertOpToLLVMPattern<SIMDGatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDGatherOp op, SIMDGatherOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    Type ptrType = LLVM::LLVMPointerType::get(
        adaptor.getPassthrough().getType().cast<VectorType>().getElementType());
    rewriter.replaceOpWithNewOp<LLVM::masked_gather>(
        op, type, adaptor.getBase(), adaptor.getMask(),
        adaptor.getPassthrough(),
        getAlignment(getTypeConverter()->getDataLayout(), ptrType));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPSIMDScatter
//===----------------------------------------------------------------------===//

struct ConvertPOPSIMDScatter : mlir::ConvertOpToLLVMPattern<SIMDScatterOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SIMDScatterOp op, SIMDScatterOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type ptrType = LLVM::LLVMPointerType::get(
        adaptor.getValue().getType().cast<VectorType>().getElementType());
    rewriter.replaceOpWithNewOp<LLVM::masked_scatter>(
        op, adaptor.getValue(), adaptor.getBase(), adaptor.getMask(),
        getAlignment(getTypeConverter()->getDataLayout(), ptrType));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPPrefetch
//===----------------------------------------------------------------------===//

struct ConvertPOPPrefetch : mlir::ConvertOpToLLVMPattern<PrefetchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PrefetchOp op, PrefetchOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert enum attributes to LLVM encoding.
    auto [rwInt, cacheBankInt] = unpackRwAndCacheBank(op.getCacheTag());

    // Create LLVM constants for the encoded attributes.
    Value rw = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(rwInt));

    Value locality = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(),
        rewriter.getI32IntegerAttr(static_cast<int>(op.getLocality())));

    Value cacheBank = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(cacheBankInt));

    rewriter.replaceOpWithNewOp<LLVM::Prefetch>(op, adaptor.getPtr(), rw,
                                                locality, cacheBank);
    return success();
  }

private:
  /// Unpacks the tag following the doc at:
  /// https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic
  static std::pair<int, int> unpackRwAndCacheBank(PrefetchTag tag) {
    switch (tag) {
    case PrefetchTag::ReadDCache:
      return {0, 1};
    case PrefetchTag::ReadICache:
      return {0, 0};
    case PrefetchTag::WriteDCache:
      return {1, 1};
    }
    llvm_unreachable("unknown prefetch tag");
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariantCreate
//===----------------------------------------------------------------------===//

struct ConvertPOPVariantCreate
    : public mlir::ConvertOpToLLVMPattern<VariantCreateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantCreateOp op, VariantCreateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto variantType = dyn_cast_if_present<LLVM::LLVMStructType>(
        getTypeConverter()->convertType(op.getType()));
    if (!variantType)
      return failure();

    VariantHelper helper(rewriter, op.getLoc());
    Value result = helper.materializeLLVMVariant(
        variantType, adaptor.getOperand(),
        *op.getType().getTypeIndex(op.getOperand().getType()));
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariantIs
//===----------------------------------------------------------------------===//

/// Lower `pop.variant.is` to an extract and integer compare.
struct ConvertPOPVariantIs : public mlir::ConvertOpToLLVMPattern<VariantIsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantIsOp op, VariantIsOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value discr = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 1);
    auto variantType =
        adaptor.getVariant().getType().cast<LLVM::LLVMStructType>();
    Value discrVal = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), variantType.getBody().back(),
        *op.getVariant().getType().getTypeIndex(op.getTestType()));
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq,
                                              discr, discrVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariantGet
//===----------------------------------------------------------------------===//

struct ConvertPOPVariantGet : mlir::ConvertOpToLLVMPattern<VariantGetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VariantGetOp op, VariantGetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type valueType = getTypeConverter()->convertType(op.getType());
    if (!valueType)
      return failure();
    auto variantType =
        adaptor.getVariant().getType().cast<LLVM::LLVMStructType>();
    auto contentType = cast<LLVM::LLVMArrayType>(variantType.getBody().front());

    // Extract the content and put it in the block of memory.
    Value content = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 0);

    SmallVector<Value> storageValues;
    for (unsigned i = 0, e = contentType.getNumElements(); i != e; ++i)
      storageValues.push_back(
          rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), content, i));

    VariantHelper helper(rewriter, op.getLoc());
    ArrayRef<Value>::iterator valueIt = storageValues.begin();
    unsigned storageOffset = 0;
    unsigned offset = 0;
    Value result =
        helper.walkAndExtractVariant(valueIt, storageOffset, offset, valueType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPIndirectCall
//===----------------------------------------------------------------------===//

struct ConvertPOPIndirectCall : mlir::ConvertOpToLLVMPattern<IndirectCallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IndirectCallOp op, IndirectCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the result types.
    SmallVector<Type> types;
    if (op.getNumResults()) {
      types.assign(
          {getTypeConverter()->packFunctionResults(op.getResultTypes())});
      if (!types.back())
        return emitError(op.getLoc(), "failed to convert call result types");
    }

    // Create the LLVM call operation.
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), types, FlatSymbolRefAttr(), adaptor.getOperands());

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, llvmCall.getResults());
      return success();
    }

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op.getLoc(), llvmCall.getResult(), i));
    }

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastToBuiltin
//===----------------------------------------------------------------------===//

struct ConvertPOPCastToBuiltin : mlir::ConvertOpToLLVMPattern<CastToBuiltinOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

struct ConvertPOPCastFromBuiltin
    : mlir::ConvertOpToLLVMPattern<CastFromBuiltinOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastFromBuiltinOp op, CastFromBuiltinOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPMemcpy
//===----------------------------------------------------------------------===//

struct ConvertPOPMemcpy : mlir::ConvertOpToLLVMPattern<MemcpyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemcpyOp op, MemcpyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto isVolatile = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getBoolAttr(adaptor.getIsVolatile().has_value()));
    if (op.getIsInlined()) {
      rewriter.replaceOpWithNewOp<LLVM::MemcpyInlineOp>(
          op, adaptor.getDest(), adaptor.getSrc(), adaptor.getSize(),
          isVolatile);
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::MemcpyOp>(
        op, adaptor.getDest(), adaptor.getSrc(), adaptor.getSize(), isVolatile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPMemset
//===----------------------------------------------------------------------===//

struct ConvertPOPMemset : mlir::ConvertOpToLLVMPattern<MemsetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemsetOp op, MemsetOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto isVolatile = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getBoolAttr(adaptor.getIsVolatile().has_value()));
    rewriter.replaceOpWithNewOp<LLVM::MemsetOp>(op, adaptor.getDest(),
                                                adaptor.getValue(),
                                                adaptor.getSize(), isVolatile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPInlineAsm
//===----------------------------------------------------------------------===//

struct ConvertPOPInlineAsm : mlir::ConvertOpToLLVMPattern<InlineAsmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(InlineAsmOp op, InlineAsmOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 2> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op, types, adaptor.getOperands(), adaptor.getAssemblyAttr(),
        adaptor.getConstraintsAttr(), adaptor.getHasSideEffectsAttr(),
        adaptor.getIsStackAlignedAttr(),
        LLVM::AsmDialectAttr::get(op.getContext(), LLVM::AsmDialect::AD_ATT),
        adaptor.getOperandAttrsAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicCmpXchg
//===----------------------------------------------------------------------===//

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

class ConvertPOPAtomicCmpXchg
    : public mlir::ConvertOpToLLVMPattern<AtomicCmpXchgOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicCmpXchgOp op, AtomicCmpXchgOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
        op, type, adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal(),
        getAtomicOrdering(op.getSuccessOrdering()),
        getAtomicOrdering(op.getFailureOrdering()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicRMW
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicRMW : public mlir::ConvertOpToLLVMPattern<AtomicRMWOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicRMWOp op, AtomicRMWOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *cast<SIMDType>(op.getType()).getResolvedDType();
    Type type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        op, type, getAtomicBinOp(dtype, adaptor.getBinOp()), adaptor.getPtr(),
        adaptor.getVal(), getAtomicOrdering(op.getOrdering()));
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
      return dtype.isSInt() ? LLVM::AtomicBinOp::max : LLVM::AtomicBinOp::umax;
    case AtomicBinOp::MIN:
      return dtype.isSInt() ? LLVM::AtomicBinOp::min : LLVM::AtomicBinOp::umin;
    }
    llvm_unreachable("unknown atomic ordering");
  }
};

//===----------------------------------------------------------------------===//
// Trivial Conversions
//===----------------------------------------------------------------------===//

using ConvertPOPAnd = mlir::OneToOneConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using ConvertPOPOr = mlir::OneToOneConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using ConvertPOPXOr = mlir::OneToOneConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;
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
using ConvertPOPRem = OneToOneFloatOrIntConversion<RemOp, LLVM::FRemOp,
                                                   LLVM::SRemOp, LLVM::URemOp>;
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
using ConvertPOPIndexToPointer =
    mlir::OneToOneConvertToLLVMPattern<IndexToPointerOp, LLVM::IntToPtrOp>;
using ConvertPOPPointerToIndex =
    mlir::OneToOneConvertToLLVMPattern<PointerToIndexOp, LLVM::PtrToIntOp>;
using ConvertPOPCallLLVMIntrinsic =
    mlir::OneToOneConvertToLLVMPattern<CallLLVMIntrinsicOp,
                                       LLVM::CallIntrinsicOp>;

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
      ConvertPOPCopySign,
      ConvertPOPDiv,
      ConvertPOPFMA,
      ConvertPOPIndexToPointer,
      ConvertPOPIndirectCall,
      ConvertPOPInlineAsm,
      ConvertPOPLoad,
      ConvertPOPMax,
      ConvertPOPMemcpy,
      ConvertPOPMemset,
      ConvertPOPMin,
      ConvertPOPMul,
      ConvertPOPNeg,
      ConvertPOPOffset,
      ConvertPOPOr,
      ConvertPOPPointerBitcast,
      ConvertPOPPointerToIndex,
      ConvertPOPPrefetch,
      ConvertPOPRem,
      ConvertPOPSelect,
      ConvertPOPShl,
      ConvertPOPShr,
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDGather,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDReduceAdd,
      ConvertPOPSIMDReduceMax,
      ConvertPOPSIMDReduceMin,
      ConvertPOPSIMDReduceMul,
      ConvertPOPSIMDScatter,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPStore,
      ConvertPOPStructConstruct,
      ConvertPOPStructGEP,
      ConvertPOPStructGet,
      ConvertPOPStructReplace,
      ConvertPOPSub,
      ConvertPOPVariantCreate,
      ConvertPOPVariantGet,
      ConvertPOPVariantIs,
      ConvertPOPXOr
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
  diag.attachNote() << "try running it before lower-scf-to-llvm";
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
  target.addLegalDialect<LLVM::LLVMDialect>();

  // These ops are handled by other passes.
  target.addLegalOp<GlobalConstantOp>();
  target.addLegalOp<ExternalCallOp>();
  target.addLegalOp<VariantVisitOp>();
  target.addLegalOp<YieldOp>();
  target.addLegalOp<CoroutineHandleOp>();
  target.addLegalOp<CoroutineAwaitOp>();
  target.addLegalOp<CoroutinePromiseOp>();
  target.addLegalOp<CoroutineResumeOp>();
  target.addLegalOp<CoroutineDestroyOp>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  POPToLLVMTypeConverter typeConverter(func->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populatePOPToLLVMPatterns(typeConverter, patterns);
  patterns.insert<ConvertPOPStackAllocation>(typeConverter,
                                             &func->getFunctionBody().front());
  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(mlir::applyPartialConversion(*func, target, std::move(patterns))))
    return signalPassFailure();

  // If this function has debug info, update any unresolved pop types.
  if (DebugInfo::extractScope(*func)) {
    POPToLLVMDebugInfoTypeConverter debugTypeConverter(typeConverter);
    debugTypeConverter.applyRecursively(*func);
  }
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
    std::optional<FunctionType> funcType = op.getVariadicType();
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
      auto global = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), type.cast<LLVM::LLVMPointerType>().getElementType(),
          true, LLVM::Linkage::Internal, "global_constant", Attribute());

      // Emit the constant using an initializer region.
      global.getBodyRegion().push_back(new Block);
      ImplicitLocOpBuilder b(op.getLoc(), op.getContext());
      b.setInsertionPointToStart(global.getBody());
      Value value =
          convertParameterToLLVM(b, *getTypeConverter(), op.getValue());
      if (!value)
        return failure();
      b.create<LLVM::ReturnOp>(value);

      // Insert the global into the module.
      symtab.insert(it->second = global);
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

struct LowerGlobalPOPToLLVMPass
    : public KGEN::impl::LowerGlobalPOPToLLVMBase<LowerGlobalPOPToLLVMPass> {
  using LowerGlobalPOPToLLVMBase::LowerGlobalPOPToLLVMBase;

  void runOnOperation() override;
};

} // namespace

void LowerGlobalPOPToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  POPToLLVMTypeConverter typeConverter(theModule->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());

  // Convert external calls.
  target.addIllegalOp<ExternalCallOp>();
  patterns.insert<ConvertPOPExternalCall>(symtab, typeConverter);

  // Convert global constants.
  DenseMap<TypedAttr, LLVM::GlobalOp> constants;
  target.addIllegalOp<GlobalConstantOp>();
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter);

  // pop.compiler.* are all illegal.
  target.addIllegalOp<CompilerGlobalLoadOp, CompilerGlobalStoreOp>();

  DebugInfo::populateTypeConversionPatterns(patterns, typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
