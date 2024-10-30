//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/Analysis/CFG.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"

using namespace M;
using namespace HLCF;

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
      SmallVector<CFGNode> curSuccessors;
      for (const ControlFlowTarget &target : targets) {
        curSuccessors.emplace_back(node, target.index);
        predecessors[curSuccessors.back()].push_back(op);
      }
      successors.try_emplace(op, std::move(curSuccessors));
    } else if (auto term = dyn_cast<ControlFlowTerminator>(op)) {
      SmallVector<Attribute> operands(op->getNumOperands());
      SmallVector<ControlFlowTarget> targets;
      term.getBranchTargets(operands, targets);
      SmallVector<CFGNode> curSuccessors;
      auto node = dyn_cast<ControlFlowNode>(HLCF::getParentNode(term));
      // If the successor is not a control-flow node, then it must be a
      // function, which does not participate in the CFG.
      if (!node)
        return;
      for (const ControlFlowTarget &target : targets) {
        curSuccessors.emplace_back(node, target.index);
        predecessors[curSuccessors.back()].push_back(op);
      }
      successors.try_emplace(op, std::move(curSuccessors));
    }
  });
}
