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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"

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
struct OneToOneFloatOrIntConversion : public ConvertPOPToLLVMPattern<Op> {
  using ConvertPOPToLLVMPattern<Op>::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    Type type = this->convertType(op.getType());

    if (dtype.isInt() || dtype.isIndex()) {
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
      rewriter.replaceOpWithNewOp<LLVM::FMAOp>(op, adaptor.getA(),
                                               adaptor.getB(), adaptor.getC(),
                                               LLVM_FASTMATH_FLAGS);
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
    auto gep = rewriter.create<LLVM::GEPOp>(
        op.getLoc(), /*resultType=*/adaptor.getPtr().getType(),
        /*basePtrType=*/elementType,
        /*basePtr=*/adaptor.getPtr(), adaptor.getIndex());
    rewriter.replaceOp(op, gep);
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
/// with lifetime markers and hoisting it to the top of the enclosing function.
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

  static unsigned resolveAlignment(std::optional<TypedAttr> alignment) {
    if (!alignment)
      return 0;
    return cast<IntegerAttr>(*alignment).getInt();
  }
};

/// Generate the LLVM IR to materialize an alloca with the given LLVM type and
/// count. The alloca is created at the top of the given block, and lifetime
/// markers are inserted at the end of the given operation's block.
static Value materializeLLVMAlloca(OpBuilder &b, Type elementType,
                                   int64_t count, Operation *op,
                                   int64_t typeAllocSize, int64_t align) {
  unsigned addressSpace = 0;
  auto alloca = dyn_cast<StackAllocationOp>(op);
  if (alloca) {
    if (auto addrSpaceAttr =
            cast_or_null<IntegerAttr>(alloca.getAddressSpaceAttr()))
      addressSpace = addrSpaceAttr.getInt();
  }

  Value countVal =
      b.create<LLVM::ConstantOp>(op->getLoc(), b.getI64IntegerAttr(count));
  auto ptr = b.create<LLVM::AllocaOp>(
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

  Value alloca = materializeLLVMAlloca(
      rewriter, elementType, cast<IntegerAttr>(op.getCount()).getInt(), op,
      *typeAllocSize, resolveAlignment(op.getAlignment()));
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
// getAlignment
//===----------------------------------------------------------------------===//

static unsigned getAlignment(const POPToLLVMTypeConverter *tc,
                             PointerType ptrType,
                             TypedAttr alignmentAttr = {}) {
  // If we have the alignment attribute, use it.
  if (alignmentAttr)
    return cast<IntegerAttr>(alignmentAttr).getInt();

  return tc->getTypeABIAlign(tc->convertType(ptrType.getElementType()));
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
    auto loadOp = rewriter.create<LLVM::LoadOp>(op.getLoc(), elementType,
                                                adaptor.getPtr(), alignment);
    rewriter.replaceOp(op, loadOp);
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
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getArg(),
                                               adaptor.getPtr(), alignment,
                                               /*isVolatile=*/false);
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
  Value ptr = materializeLLVMAlloca(rewriter, elementType, count, op,
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
        op.getLoc(), types, adaptor.getOperands(),
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
    : public ConvertPOPToLLVMPattern<AtomicCmpXchgOp> {
public:
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult
  matchAndRewrite(AtomicCmpXchgOp op, AtomicCmpXchgOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
        op, adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal(),
        getAtomicOrdering(op.getSuccessOrdering()),
        getAtomicOrdering(op.getFailureOrdering()));
    return success();
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
// ConvertPOPStringAddress
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
        op, types, cast<StringAttr>(op.getIntrin()).getValue(),
        adaptor.getOperands(), convertFastmathFlags(op.getFastmathFlags()));
    return success();
  }

  /// POP dialect fastmath flags match the LLVM ones.
  static LLVM::FastmathFlags convertFastmathFlags(FastmathFlags fmf) {
    return static_cast<LLVM::FastmathFlags>(fmf);
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
      ConvertPOPSIMDExtractElement,
      ConvertPOPSIMDInsertElement,
      ConvertPOPSIMDSelect,
      ConvertPOPSIMDShuffle,
      ConvertPOPSIMDSplat,
      ConvertPOPStore,
      ConvertPOPStringAddress,
      ConvertPOPStringSize,
      ConvertPOPSub,
      ConvertPOPUnionBitcast,
      ConvertPOPUnionUnwrap,
      ConvertPOPUnionWrap,
      ConvertPOPVariadicGet,
      ConvertPOPVariadicSize,
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

  DebugInfoTypeConverter debugTypeConverter(typeConverter);
  DebugInfo::populateTypeConversionPatterns(patterns, debugTypeConverter,
                                            typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(mlir::applyPartialConversion(*func, target, std::move(patterns))))
    return signalPassFailure();

  // If this function has debug info, update any unresolved pop types.
  if (DebugInfo::extractScope(*func))
    debugTypeConverter.applyRecursively(*func);
}

namespace {

//===----------------------------------------------------------------------===//
// ConvertPOPExternalCall
//===----------------------------------------------------------------------===//

/// Lower an external call. Add the callee to the symbol table.
struct ConvertPOPExternalCall : public ConvertSymbolOpToLLVM<ExternalCallOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult
  matchAndRewrite(ExternalCallOp op, ExternalCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<FunctionType> funcType = op.getVariadicType();
    if (!funcType) {
      // Expand one level of struct type from any operand types, these come from
      // !kgen.pack.
      SmallVector<Type> operandTypes;
      operandTypes.reserve(op.getNumOperands());
      for (auto type : op.getOperandTypes()) {
        if (auto structTy = dyn_cast<StructType>(type)) {
          operandTypes.append(structTy.getElementTypes().begin(),
                              structTy.getElementTypes().end());
        } else {
          operandTypes.push_back(type);
        }
      }
      funcType = rewriter.getFunctionType(operandTypes, op.getResultTypes());
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

    // Expand one level of structs so kgen.pack elements are passed as
    // individual values instead of as a kgen.struct.
    SmallVector<Value> operands;
    operands.reserve(op.getNumOperands());
    for (auto value : adaptor.getOperands()) {
      if (auto structTy = dyn_cast<LLVM::LLVMStructType>(value.getType())) {
        // Unpack each of the elements.
        for (size_t i = 0, e = structTy.getBody().size(); i != e; ++i) {
          auto elt = rewriter.createOrFold<LLVM::ExtractValueOp>(op.getLoc(),
                                                                 value, i);
          operands.push_back(elt);
        }
      } else {
        operands.push_back(value);
      }
    }

    LLVM::CallOp call = createLLVMCall(rewriter, op.getLoc(), func, operands);
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
class ConvertPOPAlignedAlloc : public ConvertPOPToLLVMPattern<AlignedAllocOp> {
public:
  ConvertPOPAlignedAlloc(SymbolTable &symtab, StringRef allocFnName,
                         mlir::LLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        allocFnName(allocFnName),
        allocFnSig(LLVM::LLVMFunctionType::get(
            LLVM::LLVMPointerType::get(&typeConverter.getContext()),
            {typeConverter.getIndexType(), typeConverter.getIndexType()})) {}

  LogicalResult matchAndRewrite(AlignedAllocOp op,
                                AlignedAllocOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(allocFnName);
    if (func && func.getFunctionType() != allocFnSig) {
      // Fail if the signature does not match the expected signature.
      return mlir::emitError(op.getLoc(), "allocator function '")
             << allocFnName << "' signature " << func.getFunctionType()
             << " does not match expected signature " << allocFnSig;
    }
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();
      SmallVector<Attribute> passthrough;
      func = b.create<LLVM::LLVMFuncOp>(mlir::UnknownLoc::get(getContext()),
                                        allocFnName, allocFnSig);

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

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// The alloc function name.
  StringRef allocFnName;
  /// The expected function signature, saved in the pattern to reduce match
  /// overhead.
  LLVM::LLVMFunctionType allocFnSig;
};

//===----------------------------------------------------------------------===//
// ConvertPOPAlignedFree
//===----------------------------------------------------------------------===//

/// This pattern will generate the aligned free function with the appropriate
/// attributes to teach LLVM about the allocator. This would enable LLVM, for
/// example, to promote heap-to-stack among other optimizations. This enables
/// the aligned free function to receive similar treatment to `free`.
class ConvertPOPAlignedFree : public ConvertPOPToLLVMPattern<AlignedFreeOp> {
public:
  ConvertPOPAlignedFree(SymbolTable &symtab, StringRef freeFnName,
                        mlir::LLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        freeFnName(freeFnName),
        freeFnSig(LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(&typeConverter.getContext()),
            LLVM::LLVMPointerType::get(&typeConverter.getContext()))) {}

  LogicalResult matchAndRewrite(AlignedFreeOp op, AlignedFreeOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    // Try to find an existing function
    auto func = symtab.lookup<LLVM::LLVMFuncOp>(freeFnName);
    if (func && func.getFunctionType() != freeFnSig) {
      // Fail if the signature does not match the expected signature.
      return mlir::emitError(op.getLoc(), "free function '")
             << freeFnName << "' signature " << func.getFunctionType()
             << " does not match expected signature " << freeFnSig;
    }
    if (!func) {
      // No function found. Create one with the appropriate attributes.
      OpBuilder::InsertionGuard guard(b);
      b.clearInsertionPoint();
      SmallVector<Attribute> passthrough;
      func = b.create<LLVM::LLVMFuncOp>(mlir::UnknownLoc::get(getContext()),
                                        freeFnName, freeFnSig);

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

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// The free function name.
  StringRef freeFnName;
  /// The expected function signature, saved in the pattern to reduce match
  /// overhead.
  LLVM::LLVMFunctionType freeFnSig;
};

//===----------------------------------------------------------------------===//
// ConvertPOPGlobalAlloc
//===----------------------------------------------------------------------===//

struct ConvertPOPGlobalAlloc : public ConvertSymbolOpToLLVM<GlobalAllocOp> {
  using ConvertSymbolOpToLLVM::ConvertSymbolOpToLLVM;

  LogicalResult matchAndRewrite(GlobalAllocOp op, GlobalAllocOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    auto func = op->getParentOfType<mlir::FunctionOpInterface>();
    b.setInsertionPoint(func);

    // Set the alignment if specified. Otherwise use the natural alignment.
    auto kgenPtrType = cast<PointerType>(op.getType());
    auto elementType = typeConverter->convertType(kgenPtrType.getElementType());
    unsigned alignment =
        getAlignment(getTypeConverter(), kgenPtrType, op.getAlignmentAttr());

    // Set the address space if specified.
    unsigned addrSpace = 0;
    if (auto addrSpaceAttr =
            dyn_cast_or_null<IntegerAttr>(op.getAddressSpaceAttr()))
      addrSpace = addrSpaceAttr.getInt();

    // Mangle the name according to the contained function.
    std::string name = (func.getName() + "_global_alloc").str();

    // Create the global.
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
      Value value =
          convertParameterToLLVM(b, *getTypeConverter(), /*imc=*/nullptr,
                                 /*scope=*/nullptr, op.getValue());
      if (!value)
        return failure();
      b.create<LLVM::ReturnOp>(value);

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
class ConvertExternPointerSymbol
    : public ConvertPOPToLLVMPattern<ExternPointerSymbolOp> {
public:
  ConvertExternPointerSymbol(SymbolTable &symtab,
                             DenseMap<Value, LLVM::GlobalOp> &externPtrs,
                             mlir::LLVMTypeConverter &typeConverter)
      : ConvertPOPToLLVMPattern(typeConverter), symtab(symtab),
        externPtrs(externPtrs) {}

  LogicalResult
  matchAndRewrite(ExternPointerSymbolOp op,
                  ExternPointerSymbolOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t addressSpace =
        cast<IntegerAttr>(op.getResSymbol().getType().getAddressSpace())
            .getInt();
    // Unique the external symbols.
    auto [it, inserted] = externPtrs.try_emplace(op.getResSymbol(), nullptr);

    if (inserted) {
      // If the constant doesn't exist, create it and insert it in the module.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.clearInsertionPoint();

      Type resType = convertType(op.getResSymbol().getType().getElementType());
      auto global = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), resType, /*constant=*/false, LLVM::Linkage::External,
          "extern_ptr_syml", /*value=*/nullptr,
          /*alignment=*/
          getAlignment(getTypeConverter(), op.getResSymbol().getType(),
                       op.getAlignmentAttr()),
          addressSpace,
          /*dso_local=*/true);

      symtab.insert(it->second = global);
    }

    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, LLVM::LLVMPointerType::get(getContext(), addressSpace),
        FlatSymbolRefAttr::get(it->second.getSymNameAttr()));
    return success();
  }

private:
  /// The symbol table.
  SymbolTable &symtab;
  /// Uniqued symbols.
  DenseMap<Value, LLVM::GlobalOp> &externPtrs;
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
  target.addIllegalOp<ExternalCallOp>();
  patterns.insert<ConvertPOPGlobalAlloc, ConvertPOPExternalCall>(typeConverter,
                                                                 symtab);
  patterns.insert<ConvertPOPAlignedAlloc>(symtab, allocFnName, typeConverter);
  patterns.insert<ConvertPOPAlignedFree>(symtab, freeFnName, typeConverter);

  // Convert global constants.
  DenseMap<std::pair<TypedAttr, TypedAttr>, LLVM::GlobalOp> constants;
  target.addIllegalOp<GlobalConstantOp>();
  patterns.insert<ConvertPOPGlobalConstant>(symtab, constants, typeConverter);

  // pop.compiler.* are all illegal.
  target.addIllegalOp<CompilerGlobalLoadOp, CompilerGlobalStoreOp>();

  // Convert external ptr symbol
  target.addIllegalOp<ExternPointerSymbolOp>();
  DenseMap<Value, LLVM::GlobalOp> externalPtrs;
  patterns.insert<ConvertExternPointerSymbol>(symtab, externalPtrs,
                                              typeConverter);

  DebugInfoTypeConverter debugTypeConverter(typeConverter);
  DebugInfo::populateTypeConversionPatterns(patterns, debugTypeConverter,
                                            typeConverter);
  target.addDynamicallyLegalDialect<DebugInfo::DebugInfoDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  if (failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    return signalPassFailure();
}
