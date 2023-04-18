//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/CFG.h"

using namespace M;
using namespace HLCF;

/// Get the parent operation of a terminator.
static ControlFlowNode getParentNode(ControlFlowTerminator term) {
  Operation *op = term->getParentOp();
  while (!term.isParentNode(op))
    op = op->getParentOp();
  return cast<ControlFlowNode>(op);
}

CFGAnalysis::CFGAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (auto node = dyn_cast<ControlFlowNode>(op)) {
      SmallVector<Attribute> operands(op->getNumOperands());
      SmallVector<ControlFlowTarget> targets;
      node.getEntryTargets(operands, targets);
      SmallVector<CFGNode> successors;
      for (const ControlFlowTarget &target : targets) {
        successors.emplace_back(node, target.index);
        predecessors[successors.back()].push_back(op);
      }
    } else if (auto term = dyn_cast<ControlFlowTerminator>(op)) {
      SmallVector<Attribute> operands(op->getNumOperands());
      SmallVector<ControlFlowTarget> targets;
      term.getBranchTargets(operands, targets);
      SmallVector<CFGNode> successors;
      ControlFlowNode node = getParentNode(term);
      for (const ControlFlowTarget &target : targets) {
        successors.emplace_back(node, target.index);
        predecessors[successors.back()].push_back(op);
      }
    }
  });
}
