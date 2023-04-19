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
  return dyn_cast<ControlFlowNode>(op);
}

CFGAnalysis::CFGAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (auto node = dyn_cast<ControlFlowNode>(op)) {
      // Ensure each node has a predecessor list, even if empty.
      predecessors.insert({{node, {}}, {}});
      for (unsigned i = 0, e = op->getNumRegions(); i != e; ++i)
        predecessors.insert({{node, i}, {}});

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
      // If the successor is not a control-flow node, then it must be a
      // function, which does not participate in the CFG.
      if (!node)
        return;
      for (const ControlFlowTarget &target : targets) {
        successors.emplace_back(node, target.index);
        predecessors[successors.back()].push_back(op);
      }
    }
  });
}
