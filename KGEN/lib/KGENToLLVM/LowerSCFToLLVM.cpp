//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
namespace scf = mlir::scf;
namespace LLVM = mlir::LLVM;

namespace {

// These patterns were copied from `ConvertSCFToControlFlow.cpp`. The main
// difference is the use of LLVM operations and the region type conversions,
// mainly to convert `index` and other `pop` dialect types to LLVM.

//===----------------------------------------------------------------------===//
// ConvertSCFWhileOp
//===----------------------------------------------------------------------===//

struct ConvertSCFWhileOp : public mlir::ConvertOpToLLVMPattern<scf::WhileOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, scf::WhileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

LogicalResult
ConvertSCFWhileOp::matchAndRewrite(scf::WhileOp op, scf::WhileOpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  // Convert the loop body types.
  if (failed(
          rewriter.convertRegionTypes(&op.getBefore(), *getTypeConverter())) ||
      failed(rewriter.convertRegionTypes(&op.getAfter(), *getTypeConverter())))
    return rewriter.notifyMatchFailure(
        op.getLoc(), "failed to convert region argument types");

  // Split the current block before the WhileOp to create the inlining point.
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  // Inline both regions.
  Block *after = &op.getAfter().front();
  Block *afterLast = &op.getAfter().back();
  Block *before = &op.getBefore().front();
  Block *beforeLast = &op.getBefore().back();
  rewriter.inlineRegionBefore(op.getAfter(), continuation);
  rewriter.inlineRegionBefore(op.getBefore(), after);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<LLVM::BrOp>(loc, adaptor.getInits(), before);

  // Replace terminators with branches. Assuming bodies are SESE, which holds
  // given only the patterns from this file, we only need to look at the last
  // block. This should be reconsidered if we allow break/continue in SCF.
  rewriter.setInsertionPointToEnd(beforeLast);
  auto condOp = cast<scf::ConditionOp>(beforeLast->getTerminator());
  Value condition = rewriter.getRemappedValue(condOp.getCondition());
  if (!condition)
    return rewriter.notifyMatchFailure(condOp.getLoc(),
                                       "failed to convert condition");
  SmallVector<Value> args;
  if (failed(rewriter.getRemappedValues(condOp.getArgs(), args)))
    return rewriter.notifyMatchFailure(condOp.getLoc(),
                                       "failed to convert condition arguments");
  rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(condOp, condition, after, args,
                                              continuation, ValueRange());

  rewriter.setInsertionPointToEnd(afterLast);
  auto yieldOp = cast<scf::YieldOp>(afterLast->getTerminator());
  SmallVector<Value> results;
  if (failed(rewriter.getRemappedValues(yieldOp.getResults(), results)))
    return rewriter.notifyMatchFailure(yieldOp.getLoc(),
                                       "failed to convert yield results");
  rewriter.replaceOpWithNewOp<LLVM::BrOp>(yieldOp, results, before);

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(op, args);

  return success();
}

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
  if (failed(
          rewriter.getRemappedValues(terminator->getOperands(), loopCarried)))
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
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "could not convert result types");

    continueBlock =
        rewriter.createBlock(remainingOpsBlock, resultTypes,
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
  if (failed(rewriter.getRemappedValues(thenTerminator->getOperands(),
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
    if (failed(rewriter.getRemappedValues(elseTerminator->getOperands(),
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

//===----------------------------------------------------------------------===//
// ConvertSCFIndexSwitchOp
//===----------------------------------------------------------------------===//

struct ConvertSCFIndexSwitchOp
    : public mlir::ConvertOpToLLVMPattern<scf::IndexSwitchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, scf::IndexSwitchOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Split the block at the op.
    Block *condBlock = rewriter.getInsertionBlock();
    Block *continueBlock = rewriter.splitBlock(condBlock, Block::iterator(op));

    // Create the arguments on the continue block with which to replace the
    // results of the op.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      Type type = getTypeConverter()->convertType(resultType);
      if (!type)
        return rewriter.notifyMatchFailure(op.getLoc(),
                                           "could not convert result types");
      results.push_back(continueBlock->addArgument(type, op.getLoc()));
    }

    // Handle the regions.
    auto convertRegion = [&](Region &region) -> FailureOr<Block *> {
      Block *block = &region.front();

      // Convert the yield terminator to a branch to the continue block.
      auto yield = cast<scf::YieldOp>(block->getTerminator());
      rewriter.setInsertionPoint(yield);
      SmallVector<Value> operands;
      operands.reserve(yield.getNumOperands());
      if (failed(rewriter.getRemappedValues(yield.getOperands(), operands)))
        return yield.emitError("failed to get remapped operands");
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(yield, operands, continueBlock);

      // Inline the region.
      rewriter.inlineRegionBefore(region, continueBlock);
      return block;
    };

    // Convert the case regions.
    SmallVector<Block *> caseSuccessors;
    SmallVector<int32_t> caseValues;
    caseSuccessors.reserve(op.getCases().size());
    caseValues.reserve(op.getCases().size());
    for (auto [region, value] : llvm::zip(op.getCaseRegions(), op.getCases())) {
      FailureOr<Block *> block = convertRegion(region);
      if (failed(block))
        return failure();
      caseSuccessors.push_back(*block);
      caseValues.push_back(value);
    }

    // Convert the default region.
    FailureOr<Block *> defaultBlock = convertRegion(op.getDefaultRegion());
    if (failed(defaultBlock))
      return failure();

    // Create the LLVM switch.
    rewriter.setInsertionPointToEnd(condBlock);
    SmallVector<ValueRange> caseOperands(caseSuccessors.size(), {});
    rewriter.create<LLVM::SwitchOp>(op.getLoc(), adaptor.getArg(),
                                    *defaultBlock, ValueRange(), caseValues,
                                    caseSuccessors, caseOperands);
    rewriter.replaceOp(op, continueBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPVariantVisit
//===----------------------------------------------------------------------===//

struct ConvertPOPVariantVisit
    : mlir::ConvertOpToLLVMPattern<POP::VariantVisitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(POP::VariantVisitOp op, POP::VariantVisitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Store the contents into a block of memory.
    auto variantType =
        adaptor.getVariant().getType().cast<LLVM::LLVMStructType>();
    Value contentPtr =
        createAllocaAtEntry(op, variantType.getBody().front(), rewriter);
    // Compute the bytecount of the content.
    int64_t byteCount = getByteCount(variantType.getBody().front());
    Value content = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 0);
    rewriter.create<LLVM::LifetimeStartOp>(op.getLoc(), byteCount, contentPtr);
    rewriter.create<LLVM::StoreOp>(op.getLoc(), content, contentPtr);

    // Split the block at the op.
    Block *condBlock = rewriter.getInsertionBlock();
    Block *continueBlock = rewriter.splitBlock(condBlock, Block::iterator(op));

    // Create the arguments on the continue block with which to replace the
    // results of the op.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (Type resultType : op.getResultTypes()) {
      Type type = getTypeConverter()->convertType(resultType);
      if (!type)
        return op.emitError("failed to convert result type");
      results.push_back(continueBlock->addArgument(type, op.getLoc()));
    }

    // Rewrite a yield terminator.
    auto rewriteYield = [&](Block *block) -> LogicalResult {
      auto yield = cast<POP::YieldOp>(block->getTerminator());
      rewriter.setInsertionPoint(yield);
      SmallVector<Value> operands;
      operands.reserve(yield.getNumOperands());
      if (failed(rewriter.getRemappedValues(yield.getOperands(), operands)))
        return yield.emitError("failed to get remapped operands");
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(yield, operands, continueBlock);
      return success();
    };

    // Handle the case regions.
    SmallVector<Block *> successors;
    successors.reserve(op.getNumRegions());
    for (auto [caseType, region] : llvm::zip(op.getCases(), op.getRegions())) {
      Block *block = &region->front();

      // Load the content and replace the region argument with it.
      rewriter.setInsertionPointToStart(block);
      Type type = getTypeConverter()->convertType(caseType);
      Value valuePtr = rewriter.create<LLVM::BitcastOp>(
          op.getLoc(), LLVM::LLVMPointerType::get(type), contentPtr);
      Value value = rewriter.create<LLVM::LoadOp>(op.getLoc(), valuePtr);
      rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), byteCount, contentPtr);
      region->getArgument(0).replaceAllUsesWith(value);
      region->eraseArgument(0);

      // Inline the region.
      if (failed(rewriteYield(block)))
        return failure();
      successors.push_back(block);
      rewriter.inlineRegionBefore(*region, continueBlock);
    }

    // Handle the trailing default region if present.
    SmallVector<int32_t> caseValues;
    caseValues.reserve(op.getCases().size());
    for (Type caseType : op.getCases())
      caseValues.push_back(*op.getVariant().getType().getTypeIndex(caseType));
    if (op.getCases().size() != op.getNumRegions()) {
      Block *block = &op.getRegions().back()->front();
      // Insert a lifetime end marker on this control path.
      rewriter.setInsertionPointToStart(block);
      rewriter.create<LLVM::LifetimeEndOp>(op.getLoc(), byteCount, contentPtr);
      if (failed(rewriteYield(block)))
        return failure();
      successors.push_back(block);
      rewriter.inlineRegionBefore(*op.getRegions().back(), continueBlock);
    } else {
      caseValues.pop_back();
    }

    // Create the LLVM switch. If all cases were specified, pick the last block
    // as the default.
    rewriter.setInsertionPointToEnd(condBlock);
    Value discr = rewriter.create<LLVM::ExtractValueOp>(
        op.getLoc(), adaptor.getVariant(), 1);
    SmallVector<ValueRange> caseOperands(successors.size(), {});
    rewriter.create<LLVM::SwitchOp>(
        op.getLoc(), discr, successors.back(), ValueRange(), caseValues,
        ArrayRef<Block *>(successors).drop_back(), caseOperands);
    rewriter.replaceOp(op, continueBlock->getArguments());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertArithSelectOp
//===----------------------------------------------------------------------===//

/// The `scf.if` canonicalizer produces `arith.select` operations.
struct ConvertArithSelectOp
    : public mlir::ConvertOpToLLVMPattern<mlir::arith::SelectOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op,
                  mlir::arith::SelectOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, adaptor.getCondition(),
                                                adaptor.getTrueValue(),
                                                adaptor.getFalseValue());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateSCFToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                      mlir::RewritePatternSet &patterns) {
  patterns.insert<
      // clang-format off
      ConvertArithSelectOp,
      ConvertPOPVariantVisit,
      ConvertSCFForOp,
      ConvertSCFIfOp,
      ConvertSCFIndexSwitchOp,
      ConvertSCFWhileOp
      // clang-format on
      >(typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSCFTOLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSCFToLLVMPass
    : public KGEN::impl::LowerSCFToLLVMBase<LowerSCFToLLVMPass> {
  using LowerSCFToLLVMBase::LowerSCFToLLVMBase;

  void runOnOperation() override;
};
} // namespace

void LowerSCFToLLVMPass::runOnOperation() {
  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());

  // POP control-flow operations.
  target.addIllegalOp<POP::VariantVisitOp>();
  target.addIllegalOp<POP::YieldOp>();

  // SCF operations.
  target.addIllegalDialect<mlir::scf::SCFDialect>();

  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  POPToLLVMTypeConverter typeConverter(getOperation()->getLoc(), options);

  // Run HLCF lowerings.
  if (failed(HLCF::lowerControlFlowToLLVM(getOperation(), getAnalysisManager(),
                                          typeConverter)))
    return signalPassFailure();

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateSCFToLLVMPatterns(typeConverter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
