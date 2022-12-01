//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFToLLVM/HLCFToLLVM.h"
#include "Support/HLCFDialect/Analysis/ControlFlowTree.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace HLCF;
namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// HLCF Lowering
//===----------------------------------------------------------------------===//

namespace {
/// This object contains information needed to lower operations. Lowering will
/// happen top-down, so we need to traverse the tree once to build references
/// between terminators and their branch targets and then start lowering the
/// operations.
struct ControlFlowConverter {
  explicit ControlFlowConverter(MLIRContext *ctx, ControlFlowTree &tree,
                                mlir::LLVMTypeConverter &typeConverter)
      : b(ctx), tree(tree), typeConverter(typeConverter) {}

  /// Lower the control-flow node.
  LogicalResult lowerNode(ControlFlowNode node, unsigned &termId);

  /// Lower the terminator.
  LogicalResult lowerTerminator(ControlFlowTerminator term, unsigned &termId);

  /// The rewriter to use.
  mlir::IRRewriter b;

  /// The control flow tree analysis.
  ControlFlowTree &tree;

  /// The type converter to use to convert result and argument types.
  mlir::LLVMTypeConverter &typeConverter;

  /// A map of operations to their lowered entry and exit blocks. The ID is the
  /// depth-first visit order of the operation.
  SmallVector<std::pair<SmallVector<Block *, 2>, Block *>> blocks;
};
} // namespace

static Block *getTargetBlock(ArrayRef<Block *> entries, Block *after,
                             Optional<unsigned> index) {
  if (!index)
    return after;
  return entries[*index];
}

LogicalResult ControlFlowConverter::lowerNode(ControlFlowNode node,
                                              unsigned &termId) {
  Block *before = node->getBlock();
  Block *after = b.splitBlock(before, Block::iterator(node));
  SmallVector<Block *, 2> entries;
  for (Region &region : node->getRegions())
    entries.push_back(&region.front());
  blocks.emplace_back(entries, after);
  SmallVector<Operation *> nestedNodes;

  // Process each region in the operation.
  for (Region &region : node->getRegions()) {
    // Rewrite the block argument types and inline the body.
    for (Block &block : region) {
      b.setInsertionPointToStart(&block);
      for (BlockArgument arg : block.getArguments()) {
        Type argType = typeConverter.convertType(arg.getType());
        if (!argType)
          return mlir::emitError(arg.getLoc(),
                                 "failed to convert argument type");
        // Materialize the source conversion.
        auto source = b.create<mlir::UnrealizedConversionCastOp>(
            arg.getLoc(), arg.getType(), arg);
        arg.replaceAllUsesExcept(source.getResult(0), source);
        arg.setType(argType);
      }
      // Lower the terminator.
      if (isa<ControlFlowTerminator>(block.getTerminator()))
        if (failed(lowerTerminator(block.getTerminator(), termId)))
          return failure();
      // Defer nested nodes.
      for (Operation &op : block.without_terminator())
        if (isa<ControlFlowNode>(op))
          nestedNodes.push_back(&op);
    }

    // Inline the region.
    b.inlineRegionBefore(region, after);
  }

  // Replace the results of the operation with the arguments of the exit block.
  b.setInsertionPointToStart(after);
  for (OpResult result : node->getOpResults()) {
    Type argType = typeConverter.convertType(result.getType());
    if (!argType)
      return mlir::emitError(result.getLoc(), "failed to convert result #")
             << result.getResultNumber() << " type: " << result.getType();
    BlockArgument arg = after->addArgument(argType, result.getLoc());
    auto source = b.create<mlir::UnrealizedConversionCastOp>(
        arg.getLoc(), result.getType(), arg);
    result.replaceAllUsesWith(source.getResult(0));
  }

  b.setInsertionPointToEnd(before);
  // Replace the operation.
  if (auto cond = dyn_cast<IfOp>(node.getOperation())) {
    b.create<LLVM::CondBrOp>(node->getLoc(), cond.getCond(), entries.front(),
                             ValueRange(), entries.back(), ValueRange());
    b.eraseOp(node);
  } else {
    SmallVector<ControlFlowTarget, 1> targets;
    node.getEntryTargets(
        SmallVector<Attribute>(node->getNumOperands(), Attribute()), targets);
    if (targets.size() != 1)
      return node.emitOpError("cannot lower node without 1 entry target");

    // Materialize conversions for the entry inputs.
    SmallVector<Value> inputs;
    inputs.reserve(targets.front().inputs.size());
    for (Value input : targets.front().inputs) {
      Type type = typeConverter.convertType(input.getType());
      if (!type)
        return mlir::emitError(input.getLoc(), "failed to convert input type");
      auto dest = b.create<mlir::UnrealizedConversionCastOp>(node->getLoc(),
                                                             type, input);
      inputs.push_back(dest.getResult(0));
    }
    b.create<LLVM::BrOp>(node->getLoc(), inputs,
                         getTargetBlock(entries, after, targets.front().index));
    b.eraseOp(node);
  }

  // Process nested nodes.
  for (Operation *node : nestedNodes)
    if (failed(lowerNode(node, termId)))
      return failure();
  return success();
}

LogicalResult ControlFlowConverter::lowerTerminator(ControlFlowTerminator term,
                                                    unsigned &termId) {
  // Convert the operand types.
  b.setInsertionPoint(term);
  SmallVector<Value> results;
  results.reserve(term->getNumOperands());
  for (OpOperand &operand : term->getOpOperands()) {
    Type type = typeConverter.convertType(operand.get().getType());
    if (!type)
      return mlir::emitError(operand.get().getLoc(),
                             "failed to convert operand type");
    auto dest = b.create<mlir::UnrealizedConversionCastOp>(term->getLoc(), type,
                                                           operand.get());
    results.push_back(dest.getResult(0));
  }

  // Rewrite the terminator.
  if (isa<ReturnOp>(term))
    return lowerReturnOperationToLLVM(term, results, b, typeConverter);

  assert(termId < tree.targets.size() && "malformed tree");
  auto &[nodeId, target] = tree.targets[termId];
  assert(nodeId < blocks.size() && "malformed tree");
  if (target.size() != 1)
    return term.emitOpError("cannot lower terminator without 1 target");
  b.replaceOpWithNewOp<LLVM::BrOp>(term, results,
                                   getTargetBlock(blocks[nodeId].first,
                                                  blocks[nodeId].second,
                                                  target.front().index));
  ++termId;
  return success();
}

/// Lower a single control-flow tree.
static LogicalResult
lowerControlFlowTree(Operation *root, ControlFlowTree &tree,
                     mlir::LLVMTypeConverter &typeConverter) {
  assert(!isa<ControlFlowNode>(root->getParentOp()));
  ControlFlowConverter converter(root->getContext(), tree, typeConverter);

  // Build the control-flow tree.
  converter.blocks.reserve(tree.ops.size());

  unsigned termId = 0;
  return converter.lowerNode(root, termId);
}

LogicalResult
HLCF::lowerControlFlowToLLVM(Operation *op, mlir::AnalysisManager mgr,
                             mlir::LLVMTypeConverter &typeConverter) {
  // Collect all the roots first since the lowering will break the walk order.
  SmallVector<Operation *> roots;
  op->walk([&](Operation *op) {
    if (isa<ControlFlowNode>(op) && !isa<ControlFlowNode>(op->getParentOp()))
      roots.push_back(op);
  });

  for (Operation *root : roots) {
    mlir::AnalysisManager nestedMgr = mgr.nest(root);
    auto &tree = nestedMgr.getAnalysis<ControlFlowTree>();
    if (failed(lowerControlFlowTree(root, tree, typeConverter)))
      return failure();
  }
  return success();
}

LogicalResult
HLCF::lowerReturnOperationToLLVM(Operation *op, ValueRange operands,
                                 mlir::RewriterBase &rewriter,
                                 mlir::LLVMTypeConverter &typeConverter) {
  // If the results don't need to be packed, create the LLVM return.
  if (op->getNumOperands() <= 1) {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), operands);
    return success();
  }

  // Pack the function results in a struct.
  Type type = typeConverter.packFunctionResults(op->getOperandTypes());
  if (!type)
    return emitError(op->getLoc(), "failed to convert return types");
  Value result = rewriter.create<LLVM::UndefOp>(op->getLoc(), type);
  for (auto &it : llvm::enumerate(operands)) {
    result = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), result,
                                                  it.value(), it.index());
  }

  // Create the LLVM return.
  rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::HLCF {
#define GEN_PASS_DEF_LOWERHLCFTOLLVMPASS
#include "Support/HLCFToLLVM/HLCFToLLVM.h.inc"
} // namespace M::HLCF

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct LowerHLCFToLLVMPass
    : public impl::LowerHLCFToLLVMPassBase<LowerHLCFToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    // This is a test pass. Use the default index width.
    mlir::LLVMTypeConverter typeConverter(&getContext());
    if (failed(HLCF::lowerControlFlowToLLVM(
            getOperation(), getAnalysisManager(), typeConverter)))
      return signalPassFailure();
  }
};
} // namespace
