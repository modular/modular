//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/Analysis/DataFlow.h"
#include "Support/HLCFDialect/Analysis/ControlFlowTree.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace M;
using namespace HLCF;
using namespace mlir::dataflow;

LogicalResult HLCF::DeadCodeAnalysis::visit(mlir::ProgramPoint point) {
  auto *op = point.dyn_cast<Operation *>();
  if (!op || !op->hasTrait<mlir::OpTrait::ControlFlowNode>())
    return mlir::dataflow::DeadCodeAnalysis::visit(point);

  auto markLive = [&](Region &region, ValueRange values = {}) {
    auto *executable = getOrCreate<Executable>(&region.front());
    propagateIfChanged(executable, executable->setToLive());
    auto *pred = getOrCreate<PredecessorState>(&region.front());
    propagateIfChanged(pred, pred->join(op, values));
  };

  // Set the liveness of the entry blocks of each region.
  if (auto loopOp = dyn_cast<LoopOp>(op)) {
    // The loop body is always live.
    markLive(loopOp.getBody(), loopOp.getOperands());
  } else {
    auto ifOp = cast<IfOp>(op);
    // Check the constant value of the condition.
    auto *cv = getOrCreateFor<Lattice<ConstantValue>>(point, ifOp.getCond());
    if (!cv->getValue().isUninitialized()) {
      Attribute value = cv->getValue().getConstantValue();
      if (!value) {
        // The condition value is unknown. Mark both regions as live.
        markLive(ifOp.getThenRegion());
        markLive(ifOp.getElseRegion());
      } else {
        // The condition value is known. Mark the appropriate region as live.
        markLive(cast<BoolAttr>(value).getValue() ? ifOp.getThenRegion()
                                                  : ifOp.getElseRegion());
      }
    }
  }

  // Only perform analysis from the root node.
  if (op->getParentOp()->hasTrait<mlir::OpTrait::ControlFlowNode>())
    return success();

  auto &tree = mgr.nest(op).getAnalysis<ControlFlowTree>();

  // FIXME: ControlFlowTree is optimized for fast lookups when traversing in
  // DFS, but that means we have to redo the traversal whenever analysis
  // conditions cause a re-visit of the root.
  std::function<void(Operation *, unsigned &)> visitNode =
      [&](Operation *op, unsigned &termId) {
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            Operation *term = block.getTerminator();
            if (!term->hasTrait<mlir::OpTrait::ControlFlowTerminator>())
              continue;
            ++termId;
            // If the block is not live, ignore it.
            if (!getOrCreateFor<Executable>(point, &block)->isLive())
              continue;

            // Determine if the terminator is reachable by iterating back to
            // the nearest operation with non-CFG control-flow.
            bool reachable = true;
            for (Operation &op : llvm::reverse(block.without_terminator())) {
              if (!isa<mlir::CallOpInterface, mlir::RegionBranchOpInterface>(
                      op) &&
                  !op.hasTrait<mlir::OpTrait::ControlFlowNode>())
                continue;
              // If the operation is known to have no predecsesors, the
              // terminator is not reachable.
              auto *preds = getOrCreateFor<PredecessorState>(point, &op);
              reachable = !preds->allPredecessorsKnown() ||
                          !preds->getKnownPredecessors().empty();
              break;
            }
            if (!reachable)
              continue;

            if (isa<ReturnOp>(term)) {
              auto func = term->getParentOfType<mlir::CallableOpInterface>();
              auto *callsites = getOrCreateFor<PredecessorState>(point, func);
              for (Operation *predecessor : callsites->getKnownPredecessors()) {
                auto *predecessors = getOrCreate<PredecessorState>(predecessor);
                propagateIfChanged(predecessors, predecessors->join(term));
              }
              continue;
            }

            auto [targetId, after] = tree.targets[termId - 1];
            Operation *target = tree.ops[targetId];
            PredecessorState *pred;
            if (after) {
              // If the terminator branches to after an operation, make this
              // terminator a predecessor of that operation.
              pred = getOrCreate<PredecessorState>(target);
            } else {
              // If the terminator branches to the entry block of an operation,
              // make this terminator a predecessor of that block.
              pred = getOrCreate<PredecessorState>(
                  &cast<LoopOp>(target).getBody().front());
            }
            propagateIfChanged(pred, pred->join(term, term->getOperands()));
          }
        }
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (Operation &op : block.without_terminator())
              if (op.hasTrait<mlir::OpTrait::ControlFlowNode>())
                visitNode(&op, termId);
          }
        }
      };

  unsigned termId = 0;
  visitNode(op, termId);
  return success();
}
