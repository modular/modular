//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/DataFlow.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace M;
using namespace HLCF;
using namespace mlir::dataflow;

LogicalResult HLCF::DeadCodeAnalysis::visit(mlir::ProgramPoint point) {
  auto node = dyn_cast_or_null<ControlFlowNode>(point.dyn_cast<Operation *>());
  if (!node)
    return mlir::dataflow::DeadCodeAnalysis::visit(point);

  SmallVector<Attribute> operands;
  for (Value operand : node->getOperands()) {
    auto *cv = getOrCreate<Lattice<ConstantValue>>(operand);
    cv->useDefSubscribe(this);
    if (cv->getValue().isUninitialized())
      return success();
    operands.push_back(cv->getValue().getConstantValue());
  }
  SmallVector<ControlFlowTarget> targets;
  node.getEntryTargets(operands, targets);
  for (const ControlFlowTarget &target : targets) {
    PredecessorState *pred;
    if (target.index) {
      Block *entry = &node->getRegion(*target.index).front();
      auto *exec = getOrCreate<Executable>(entry);
      propagateIfChanged(exec, exec->setToLive());
      pred = getOrCreate<PredecessorState>(entry);
    } else {
      pred = getOrCreate<PredecessorState>(node);
    }
    propagateIfChanged(pred, pred->join(node, target.inputs));
  }

  // Only perform tree analysis from the root node.
  if (isa<ControlFlowNode>(node->getParentOp()))
    return success();

  const ControlFlowTree &tree = analysis.getOrCreate(node);

  // FIXME: ControlFlowTree is optimized for fast lookups when traversing in
  // DFS, but that means we have to redo the traversal whenever analysis
  // conditions cause a re-visit of the root.
  std::function<void(Operation *, unsigned &)> visitNode =
      [&](Operation *op, unsigned &termId) {
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            Operation *term = block.getTerminator();
            if (!isa<ControlFlowTerminator>(term))
              continue;
            bool isReturn = term->hasTrait<mlir::OpTrait::ReturnLike>();
            termId += !isReturn;

            // If the block is not live, ignore it.
            if (!getOrCreateFor<Executable>(point, &block)->isLive())
              continue;

            // Determine if the terminator is reachable by iterating back to
            // the nearest operation with non-CFG control-flow.
            bool reachable = true;
            for (Operation &op : llvm::reverse(block.without_terminator())) {
              if (!isa<mlir::CallOpInterface, mlir::RegionBranchOpInterface,
                       ControlFlowNode>(op))
                continue;
              // If the operation is known to have no predecessors, the
              // terminator is not reachable.
              auto *preds = getOrCreateFor<PredecessorState>(point, &op);
              reachable = !preds->allPredecessorsKnown() ||
                          !preds->getKnownPredecessors().empty();
              break;
            }
            if (!reachable)
              continue;

            // Process returns.
            if (isReturn) {
              auto func = term->getParentOfType<mlir::CallableOpInterface>();
              auto *callsites = getOrCreateFor<PredecessorState>(point, func);
              for (Operation *predecessor : callsites->getKnownPredecessors()) {
                auto *predecessors = getOrCreate<PredecessorState>(predecessor);
                propagateIfChanged(predecessors, predecessors->join(term));
              }
              continue;
            }

            // Process all other kinds of terminators.
            auto [targetId, targets] = tree.targets[termId - 1];
            ControlFlowNode node = tree.ops[targetId];
            for (const ControlFlowTarget &target : targets) {
              PredecessorState *pred;
              if (target.index) {
                pred = getOrCreate<PredecessorState>(
                    &node->getRegion(*target.index).front());
              } else {
                pred = getOrCreate<PredecessorState>(node);
              }
              propagateIfChanged(pred, pred->join(term, target.inputs));
            }
          }
        }
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (Operation &op : block.without_terminator())
              if (isa<ControlFlowNode>(op))
                visitNode(&op, termId);
          }
        }
      };

  unsigned termId = 0;
  visitNode(node, termId);
  return success();
}
