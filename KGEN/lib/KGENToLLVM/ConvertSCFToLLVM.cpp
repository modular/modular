//===- ConvertSCFToLLVM.cpp -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "LLVMLoweringUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
namespace scf = mlir::scf;
namespace LLVM = mlir::LLVM;

namespace {

/// Materialize type conversions for the operands of `scf.yield` terminators.
static LogicalResult materializeTerminatorOperands(
    PatternRewriter &rewriter, TypeConverter &typeConverter,
    Operation *terminator, SmallVectorImpl<Value> &values) {
  for (Value operand : terminator->getOperands()) {
    Type type = typeConverter.convertType(operand.getType());
    if (!type)
      return rewriter.notifyMatchFailure(
          terminator->getLoc(), "could not convert terminator operand type");
    Value materialized = typeConverter.materializeTargetConversion(
        rewriter, terminator->getLoc(), type, operand);
    if (!materialized)
      return rewriter.notifyMatchFailure(
          terminator->getLoc(), "could not materialize source conversion");
    values.push_back(materialized);
  }
  return success();
}

// These patterns were copied from `ConvertSCFToControlFlow.cpp`. The main
// difference is the use of LLVM operations and the region type conversions,
// mainly to convert `index` and other `meta` dialect types to LLVM.

//===----------------------------------------------------------------------===//
// ConvertSCFForOp
//===----------------------------------------------------------------------===//

struct ConvertSCFForOp : mlir::ConvertOpToLLVMPattern<scf::ForOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, scf::ForOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertSCFForOp::matchAndRewrite(scf::ForOp op, scf::ForOpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Convert the induction variable and iteration variable types.
  if (failed(rewriter.convertRegionTypes(&op.getBodyRegion(),
                                         *getTypeConverter())))
    return rewriter.notifyMatchFailure(op.getLoc(),
                                       "could not convert region types");

  // Start by splitting the block containing the 'scf.for' into two parts.
  // The part before will get the init code, the part after will be the end
  // point.
  Block *initBlock = rewriter.getInsertionBlock();
  Block::iterator initPosition = rewriter.getInsertionPoint();
  Block *endBlock = rewriter.splitBlock(initBlock, initPosition);

  // Use the first block of the loop body as the condition block since it is the
  // block that has the induction variable and loop-carried values as arguments.
  // Split out all operations from the first block into a new block. Move all
  // body blocks from the loop body region to the region containing the loop.
  Block *conditionBlock = &op.getRegion().front();
  Block *firstBodyBlock =
      rewriter.splitBlock(conditionBlock, conditionBlock->begin());
  Block *lastBodyBlock = &op.getRegion().back();
  rewriter.inlineRegionBefore(op.getRegion(), endBlock);
  BlockArgument iv = conditionBlock->getArgument(0);

  // Append the induction variable stepping logic to the last body block and
  // branch back to the condition block. Loop-carried values are taken from
  // operands of the loop terminator.
  Operation *terminator = lastBodyBlock->getTerminator();
  rewriter.setInsertionPointToEnd(lastBodyBlock);
  Value step = adaptor.getStep();
  Value stepped = rewriter.create<LLVM::AddOp>(loc, iv, step);

  SmallVector<Value> loopCarried;
  loopCarried.push_back(stepped);
  if (failed(materializeTerminatorOperands(rewriter, *getTypeConverter(),
                                           terminator, loopCarried)))
    return failure();

  rewriter.create<LLVM::BrOp>(loc, loopCarried, conditionBlock);
  rewriter.eraseOp(terminator);

  // Compute loop bounds before branching to the condition.
  rewriter.setInsertionPointToEnd(initBlock);
  Value lowerBound = adaptor.getLowerBound();
  Value upperBound = adaptor.getUpperBound();

  // The initial values of loop-carried values is obtained from the operands
  // of the loop operation.
  SmallVector<Value> destOperands;
  destOperands.push_back(lowerBound);
  llvm::append_range(destOperands, adaptor.getOperands().drop_front(
                                       op.getNumControlOperands()));
  rewriter.create<LLVM::BrOp>(loc, destOperands, conditionBlock);

  // With the body block done, we can fill in the condition block.
  rewriter.setInsertionPointToEnd(conditionBlock);
  Value comparison = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::slt, iv, upperBound);

  rewriter.create<LLVM::CondBrOp>(loc, comparison, firstBodyBlock, ValueRange(),
                                  endBlock, ValueRange());
  // The result of the loop operation is the values of the condition block
  // arguments except the induction variable on the last iteration.
  rewriter.replaceOp(op, conditionBlock->getArguments().drop_front());
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertSCFIfOp
//===----------------------------------------------------------------------===//

struct ConvertSCFIfOp : public mlir::ConvertOpToLLVMPattern<scf::IfOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, scf::IfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertSCFIfOp::matchAndRewrite(scf::IfOp op, scf::IfOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Start by splitting the block containing the 'scf.if' into two parts.
  // The part before will contain the condition, the part after will be the
  // continuation point.
  Block *condBlock = rewriter.getInsertionBlock();
  Block::iterator opPosition = rewriter.getInsertionPoint();
  Block *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
  Block *continueBlock;
  if (op.getNumResults() == 0) {
    continueBlock = remainingOpsBlock;
  } else {
    continueBlock =
        rewriter.createBlock(remainingOpsBlock, op.getResultTypes(),
                             SmallVector<Location>(op.getNumResults(), loc));
    rewriter.create<LLVM::BrOp>(loc, ValueRange(), remainingOpsBlock);
  }

  // Move blocks from the "then" region to the region containing 'scf.if',
  // place it before the continuation block, and branch to it.
  Region &thenRegion = op.getThenRegion();
  Block *thenBlock = &thenRegion.front();
  Operation *thenTerminator = thenRegion.back().getTerminator();
  SmallVector<Value> thenTerminatorOperands;
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  if (failed(materializeTerminatorOperands(rewriter, *getTypeConverter(),
                                           thenTerminator,
                                           thenTerminatorOperands)))
    return failure();
  rewriter.create<LLVM::BrOp>(loc, thenTerminatorOperands, continueBlock);
  rewriter.eraseOp(thenTerminator);
  rewriter.inlineRegionBefore(thenRegion, continueBlock);

  // Move blocks from the "else" region (if present) to the region containing
  // 'scf.if', place it before the continuation block and branch to it.  It
  // will be placed after the "then" regions.
  Block *elseBlock = continueBlock;
  Region &elseRegion = op.getElseRegion();
  if (!elseRegion.empty()) {
    elseBlock = &elseRegion.front();
    Operation *elseTerminator = elseRegion.back().getTerminator();
    SmallVector<Value> elseTerminatorOperands;
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    if (failed(materializeTerminatorOperands(rewriter, *getTypeConverter(),
                                             elseTerminator,
                                             elseTerminatorOperands)))
      return failure();
    rewriter.create<LLVM::BrOp>(loc, elseTerminatorOperands, continueBlock);
    rewriter.eraseOp(elseTerminator);
    rewriter.inlineRegionBefore(elseRegion, continueBlock);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<LLVM::CondBrOp>(loc, adaptor.getCondition(), thenBlock,
                                  /*trueArgs=*/ValueRange(), elseBlock,
                                  /*falseArgs=*/ValueRange());

  // Ok, we're done!
  rewriter.replaceOp(op, continueBlock->getArguments());
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateSCFToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertSCFForOp, ConvertSCFIfOp>(typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertSCFToLLVMPass
    : public ConvertSCFToLLVMBase<ConvertSCFToLLVMPass> {
public:
  void runOnOperation() override;
};
} // namespace

void ConvertSCFToLLVMPass::runOnOperation() {
  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  MetaToLLVMTypeConverter typeConverter(getOperation()->getLoc(), options);

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateSCFToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> M::KGEN::createConvertSCFToLLVMPass() {
  return std::make_unique<ConvertSCFToLLVMPass>();
}
