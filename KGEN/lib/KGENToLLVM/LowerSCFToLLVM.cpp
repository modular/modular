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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
namespace scf = mlir::scf;
namespace LLVM = mlir::LLVM;

// These patterns were copied from `ConvertSCFToControlFlow.cpp`. The main
// difference is the use of LLVM operations and the region type conversions,
// mainly to convert `index` and other `pop` dialect types to LLVM.

/// Convert the block argument types.
static LogicalResult convertRegionTypes(mlir::IRRewriter &b, Region *region,
                                        TypeConverter &tc) {
  OpBuilder::InsertionGuard guard(b);
  for (Block &block : *region) {
    b.setInsertionPointToStart(&block);
    for (Value arg : block.getArguments()) {
      Type type = tc.convertType(arg.getType());
      if (!type)
        return mlir::emitError(arg.getLoc(), "failed to convert argument type");
      if (type == arg.getType())
        continue;
      auto cast = b.create<mlir::UnrealizedConversionCastOp>(
          arg.getLoc(), arg.getType(), arg);
      b.replaceAllUsesExcept(arg, cast.getResult(0), cast);
      arg.setType(type);
    }
  }
  return success();
}

/// Materialize a conversion for the given value.
static Value getRemappedValue(mlir::IRRewriter &b, TypeConverter &tc,
                              Value value) {
  OpBuilder::InsertionGuard guard(b);
  Type type = tc.convertType(value.getType());
  if (!type) {
    mlir::emitError(value.getLoc(), "failed to convert operand type");
    return {};
  }
  if (type == value.getType())
    return value;
  return b.create<mlir::UnrealizedConversionCastOp>(value.getLoc(), type, value)
      .getResult(0);
}

/// Materialize conversions for the given values.
static LogicalResult getRemappedValues(mlir::IRRewriter &b, TypeConverter &tc,
                                       ValueRange values,
                                       SmallVectorImpl<Value> &results) {
  for (Value operand : values) {
    if (Value result = getRemappedValue(b, tc, operand))
      results.push_back(result);
    else
      return failure();
  }
  return success();
}

/// Materialize source conversions for the replacement values.
static void replaceOp(mlir::IRRewriter &b, Operation *op, ValueRange values) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(op);
  SmallVector<Value> repls;
  for (auto [type, value] : llvm::zip(op->getResultTypes(), values)) {
    if (type == value.getType()) {
      repls.push_back(value);
      continue;
    }
    repls.push_back(
        b.create<mlir::UnrealizedConversionCastOp>(op->getLoc(), type, value)
            .getResult(0));
  }
  b.replaceOp(op, repls);
}

//===----------------------------------------------------------------------===//
// lowerWhileOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerOpImpl(scf::WhileOp op, scf::WhileOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
  Location loc = op.getLoc();

  // Convert the loop body types.
  if (failed(convertRegionTypes(rewriter, &op.getBefore(), tc)) ||
      failed(convertRegionTypes(rewriter, &op.getAfter(), tc)))
    return op.emitError("failed to convert region argument types");

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
  Value condition = getRemappedValue(rewriter, tc, condOp.getCondition());
  if (!condition)
    return condOp.emitError("failed to convert condition");
  SmallVector<Value> args;
  if (failed(getRemappedValues(rewriter, tc, condOp.getArgs(), args)))
    return condOp.emitError("failed to convert condition arguments");
  rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(condOp, condition, after, args,
                                              continuation, ValueRange());

  rewriter.setInsertionPointToEnd(afterLast);
  auto yieldOp = cast<scf::YieldOp>(afterLast->getTerminator());
  SmallVector<Value> results;
  if (failed(getRemappedValues(rewriter, tc, yieldOp.getResults(), results)))
    return yieldOp.emitError("failed to convert yield results");
  rewriter.replaceOpWithNewOp<LLVM::BrOp>(yieldOp, results, before);

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  replaceOp(rewriter, op, args);

  return success();
}

//===----------------------------------------------------------------------===//
// lowerForOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerOpImpl(scf::ForOp op, scf::ForOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
  Location loc = op.getLoc();

  // Convert the induction variable and iteration variable types.
  if (failed(convertRegionTypes(rewriter, &op.getBodyRegion(), tc)))
    return op.emitError("could not convert region types");

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
  Value stepped = rewriter.create<LLVM::AddOp>(terminator->getLoc(), iv, step);

  SmallVector<Value> loopCarried;
  loopCarried.push_back(stepped);
  if (failed(getRemappedValues(rewriter, tc, terminator->getOperands(),
                               loopCarried)))
    return failure();

  rewriter.replaceOpWithNewOp<LLVM::BrOp>(terminator, loopCarried,
                                          conditionBlock);

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
  replaceOp(rewriter, op, conditionBlock->getArguments().drop_front());
  return success();
}

//===----------------------------------------------------------------------===//
// lowerIfOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerOpImpl(scf::IfOp op, scf::IfOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
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
    if (failed(tc.convertTypes(op.getResultTypes(), resultTypes)))
      return op.emitError("could not convert result types");

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
  if (failed(getRemappedValues(rewriter, tc, thenTerminator->getOperands(),
                               thenTerminatorOperands)))
    return failure();
  rewriter.replaceOpWithNewOp<LLVM::BrOp>(
      thenTerminator, thenTerminatorOperands, continueBlock);
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
    if (failed(getRemappedValues(rewriter, tc, elseTerminator->getOperands(),
                                 elseTerminatorOperands)))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        elseTerminator, elseTerminatorOperands, continueBlock);
    rewriter.inlineRegionBefore(elseRegion, continueBlock);
  }

  rewriter.setInsertionPointToEnd(condBlock);
  rewriter.create<LLVM::CondBrOp>(loc, adaptor.getCondition(), thenBlock,
                                  /*trueArgs=*/ValueRange(), elseBlock,
                                  /*falseArgs=*/ValueRange());

  // Ok, we're done!
  replaceOp(rewriter, op, continueBlock->getArguments());
  return success();
}

//===----------------------------------------------------------------------===//
// lowerIndexSwitchOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerOpImpl(scf::IndexSwitchOp op,
                                 scf::IndexSwitchOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
  // Split the block at the op.
  Block *condBlock = rewriter.getInsertionBlock();
  Block *continueBlock = rewriter.splitBlock(condBlock, Block::iterator(op));

  // Create the arguments on the continue block with which to replace the
  // results of the op.
  SmallVector<Value> results;
  results.reserve(op.getNumResults());
  for (Type resultType : op.getResultTypes()) {
    Type type = tc.convertType(resultType);
    if (!type)
      return op.emitError("could not convert result types");
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
    if (failed(getRemappedValues(rewriter, tc, yield.getOperands(), operands)))
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
  rewriter.create<LLVM::SwitchOp>(op.getLoc(), adaptor.getArg(), *defaultBlock,
                                  ValueRange(), caseValues, caseSuccessors,
                                  caseOperands);
  replaceOp(rewriter, op, continueBlock->getArguments());
  return success();
}

//===----------------------------------------------------------------------===//
// lowerVariantVisitOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerOpImpl(POP::VariantVisitOp op,
                                 POP::VariantVisitOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
  // Store the contents into a block of memory.
  Value content = rewriter.create<LLVM::ExtractValueOp>(
      op.getLoc(), adaptor.getVariant(), 0);
  SmallVector<Value> storageValues;
  auto contentType = cast<LLVM::LLVMArrayType>(content.getType());
  for (unsigned i = 0, e = contentType.getNumElements(); i != e; ++i)
    storageValues.push_back(
        rewriter.create<LLVM::ExtractValueOp>(op.getLoc(), content, i));

  // Split the block at the op.
  Block *condBlock = rewriter.getInsertionBlock();
  Block *continueBlock = rewriter.splitBlock(condBlock, Block::iterator(op));

  // Create the arguments on the continue block with which to replace the
  // results of the op.
  SmallVector<Value> results;
  results.reserve(op.getNumResults());
  for (Type resultType : op.getResultTypes()) {
    Type type = tc.convertType(resultType);
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
    if (failed(getRemappedValues(rewriter, tc, yield.getOperands(), operands)))
      return yield.emitError("failed to get remapped operands");
    rewriter.replaceOpWithNewOp<LLVM::BrOp>(yield, operands, continueBlock);
    return success();
  };

  // Handle the case regions.
  SmallVector<Block *> successors;
  successors.reserve(op.getNumRegions());
  for (auto [caseType, region] : llvm::zip(op.getCases(), op.getRegions())) {
    Block *block = &region.front();

    // Load the content and replace the region argument with it.
    rewriter.setInsertionPointToStart(block);
    Type type = tc.convertType(caseType);
    VariantHelper helper(rewriter, op.getLoc());
    ArrayRef<Value>::iterator valueIt = storageValues.begin();
    unsigned storageOffset = 0;
    unsigned offset = 0;
    Value value =
        helper.walkAndExtractVariant(valueIt, storageOffset, offset, type);
    region.getArgument(0).replaceAllUsesWith(value);
    region.eraseArgument(0);

    // Inline the region.
    if (failed(rewriteYield(block)))
      return failure();
    successors.push_back(block);
    rewriter.inlineRegionBefore(region, continueBlock);
  }

  // Handle the trailing default region if present.
  SmallVector<int32_t> caseValues;
  caseValues.reserve(op.getCases().size());
  for (Type caseType : op.getCases())
    caseValues.push_back(*op.getVariant().getType().getTypeIndex(caseType));
  if (op.getCases().size() != op.getNumRegions()) {
    Block *block = &op.getRegions().back().front();
    // Insert a lifetime end marker on this control path.
    rewriter.setInsertionPointToStart(block);
    if (failed(rewriteYield(block)))
      return failure();
    successors.push_back(block);
    rewriter.inlineRegionBefore(op.getRegions().back(), continueBlock);
  } else {
    caseValues.pop_back();
  }

  // Create the LLVM switch. If all cases were specified, pick the last block
  // as the default.
  rewriter.setInsertionPointToEnd(condBlock);
  Value discr = rewriter.create<LLVM::ExtractValueOp>(op.getLoc(),
                                                      adaptor.getVariant(), 1);
  SmallVector<ValueRange> caseOperands(successors.size(), {});
  rewriter.create<LLVM::SwitchOp>(
      op.getLoc(), discr, successors.back(), ValueRange(), caseValues,
      ArrayRef<Block *>(successors).drop_back(), caseOperands);
  replaceOp(rewriter, op, continueBlock->getArguments());
  return success();
}

//===----------------------------------------------------------------------===//
// lowerSelectOp
//===----------------------------------------------------------------------===//

/// The `scf.if` canonicalizer produces `arith.select` operations.
static LogicalResult lowerOpImpl(mlir::arith::SelectOp op,
                                 mlir::arith::SelectOpAdaptor adaptor,
                                 mlir::IRRewriter &rewriter,
                                 TypeConverter &tc) {
  auto select = rewriter.create<LLVM::SelectOp>(
      op.getLoc(), adaptor.getCondition(), adaptor.getTrueValue(),
      adaptor.getFalseValue());
  replaceOp(rewriter, op, select->getResults());
  return success();
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

template <typename OpT>
static LogicalResult lowerOperation(OpT op, mlir::IRRewriter &b,
                                    TypeConverter &tc) {
  SmallVector<Value> converted;
  if (failed(getRemappedValues(b, tc, op->getOperands(), converted)))
    return failure();
  typename OpT::Adaptor adaptor{converted, op->getAttrDictionary()};
  return lowerOpImpl(op, adaptor, b, tc);
}

void LowerSCFToLLVMPass::runOnOperation() {
  // Set LLVM lowering options.
  FailureOr<mlir::LowerToLLVMOptions> options =
      getTargetLoweringOptions(getOperation());
  if (failed(options))
    return signalPassFailure();
  POPToLLVMTypeConverter typeConverter(getOperation()->getLoc(), *options);

  SmallVector<Operation *> ops;
  getOperation()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<scf::WhileOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
            POP::VariantVisitOp, mlir::arith::SelectOp>(op))
      ops.push_back(op);
  });
  for (Operation *op : ops) {
    mlir::IRRewriter b{OpBuilder(op)};
    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<scf::WhileOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
                  POP::VariantVisitOp, mlir::arith::SelectOp>(
                [&](auto op) { return lowerOperation(op, b, typeConverter); });
    if (failed(result))
      return signalPassFailure();
  }

  // Run HLCF lowerings.
  if (failed(HLCF::lowerControlFlowToLLVM(
          getOperation(), getAnalysis<HLCF::ControlFlowTreeAnalysis>(),
          typeConverter)))
    return signalPassFailure();

  // Erase unreachable blocks that might arise during HLCF lowering.
  mlir::IRRewriter rewriter(&getContext());
  (void)mlir::eraseUnreachableBlocks(rewriter, getOperation()->getRegions());
}
