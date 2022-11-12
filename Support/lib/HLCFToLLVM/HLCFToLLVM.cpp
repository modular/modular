//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFToLLVM/HLCFToLLVM.h"
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
  explicit ControlFlowConverter(MLIRContext *ctx,
                                mlir::LLVMTypeConverter &typeConverter)
      : b(ctx), typeConverter(typeConverter) {}

  /// Build the control-flow relations.
  void buildTree(Operation *node, unsigned &nodeId,
                 SmallVectorImpl<unsigned> &loopIds);

  /// Lower the control-flow node.
  LogicalResult lowerNode(Operation *node, unsigned &termId);

  /// Lower the terminator.
  LogicalResult lowerTerminator(Operation *term, unsigned &termId);

  /// The rewriter to use.
  mlir::IRRewriter b;

  /// The type converter to use to convert result and argument types.
  mlir::LLVMTypeConverter &typeConverter;

  /// A map of operations to their lowered entry and exit blocks. The ID is the
  /// depth-first visit order of the operation.
  SmallVector<std::pair<Block *, Block *>> blocks;

  /// A map of terminators to their branch target and a flag indicating whether
  /// the target is before or after the operation.
  SmallVector<std::pair<unsigned, bool>> targets;
};
} // namespace

void ControlFlowConverter::buildTree(Operation *node, unsigned &nodeId,
                                     SmallVectorImpl<unsigned> &loopIds) {
  auto loop = dyn_cast<LoopOp>(node);
  if (loop)
    loopIds.push_back(nodeId);

  // Process the immediate terminators and then the nested nodes. This order has
  // to be mirrored in the rewrite walk.
  for (Region &region : node->getRegions()) {
    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      if (!terminator->hasTrait<OpTrait::ControlFlowTerminator>())
        continue;
      if (isa<YieldOp>(terminator))
        targets.emplace_back(nodeId, true);
      else if (isa<BreakOp>(terminator))
        targets.emplace_back(loopIds.back(), true);
      else if (isa<ContinueOp>(terminator))
        targets.emplace_back(loopIds.back(), false);
    }
  }
  for (Region &region : node->getRegions()) {
    for (Operation &op : region.getOps()) {
      if (!op.hasTrait<OpTrait::ControlFlowNode>())
        continue;
      ++nodeId;
      buildTree(&op, nodeId, loopIds);
    }
  }

  if (loop)
    loopIds.pop_back();
}

LogicalResult ControlFlowConverter::lowerNode(Operation *node,
                                              unsigned &termId) {
  Block *before = node->getBlock();
  Block *after = b.splitBlock(before, Block::iterator(node));
  Block *entry = nullptr;
  if (auto loop = dyn_cast<LoopOp>(node))
    entry = &loop.getBody().front();
  blocks.emplace_back(entry, after);
  SmallVector<Block *, 2> entries;
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
      if (block.getTerminator()->hasTrait<OpTrait::ControlFlowTerminator>())
        if (failed(lowerTerminator(block.getTerminator(), termId)))
          return failure();
      // Defer nested nodes.
      for (Operation &op : block.without_terminator())
        if (op.hasTrait<OpTrait::ControlFlowNode>())
          nestedNodes.push_back(&op);
    }

    // Inline the region.
    entries.push_back(&region.front());
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
  if (auto cond = dyn_cast<IfOp>(node)) {
    b.create<LLVM::CondBrOp>(node->getLoc(), cond.getCond(), entries.front(),
                             ValueRange(), entries.back(), ValueRange());
    b.eraseOp(node);
  } else {
    // Materialize conversions for the operands.
    SmallVector<Value> results;
    results.reserve(node->getNumOperands());
    for (Value operand : node->getOperands()) {
      Type type = typeConverter.convertType(operand.getType());
      if (!type)
        return mlir::emitError(operand.getLoc(),
                               "failed to convert operand type");
      auto dest = b.create<mlir::UnrealizedConversionCastOp>(node->getLoc(),
                                                             type, operand);
      results.push_back(dest.getResult(0));
    }
    b.create<LLVM::BrOp>(node->getLoc(), results, entries.front());
    b.eraseOp(node);
  }

  // Process nested nodes.
  for (Operation *node : nestedNodes)
    if (failed(lowerNode(node, termId)))
      return failure();
  return success();
}

LogicalResult ControlFlowConverter::lowerTerminator(Operation *term,
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
  if (isa<ReturnOp>(term)) {
    if (term->getNumOperands() <= 1) {
      b.replaceOpWithNewOp<LLVM::ReturnOp>(term, results);
      return success();
    }
    Type packType = typeConverter.packFunctionResults(term->getOperandTypes());
    Value pack = b.create<LLVM::UndefOp>(term->getLoc(), packType);
    for (auto [idx, value] : llvm::enumerate(results))
      pack = b.create<LLVM::InsertValueOp>(term->getLoc(), pack, value, idx);
    b.replaceOpWithNewOp<LLVM::ReturnOp>(term, pack);
    return success();
  }

  assert(termId < targets.size() && "malformed tree");
  auto [nodeId, after] = targets[termId];
  assert(nodeId < blocks.size() && "malformed tree");
  Block *target = after ? blocks[nodeId].second : blocks[nodeId].first;
  b.replaceOpWithNewOp<LLVM::BrOp>(term, results, target);
  ++termId;
  return success();
}

/// Lower a single control-flow tree.
static LogicalResult
lowerControlFlowTree(Operation *root, mlir::LLVMTypeConverter &typeConverter) {
  assert(!root->getParentOp()->hasTrait<OpTrait::ControlFlowNode>());
  ControlFlowConverter converter(root->getContext(), typeConverter);

  // Build the control-flow tree.
  unsigned nodeId = 0;
  SmallVector<unsigned> loopIds;
  converter.buildTree(root, nodeId, loopIds);
  converter.blocks.reserve(nodeId);

  unsigned termId = 0;
  return converter.lowerNode(root, termId);
}

LogicalResult
HLCF::lowerControlFlowToLLVM(Operation *op,
                             mlir::LLVMTypeConverter &typeConverter) {
  // Collect all the roots first since the lowering will break the walk order.
  SmallVector<Operation *> roots;
  op->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::ControlFlowNode>() &&
        !op->getParentOp()->hasTrait<OpTrait::ControlFlowNode>())
      roots.push_back(op);
  });

  for (Operation *root : roots)
    if (failed(lowerControlFlowTree(root, typeConverter)))
      return failure();
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
    if (failed(HLCF::lowerControlFlowToLLVM(getOperation(), typeConverter)))
      return signalPassFailure();
  }
};
} // namespace
