//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/HLCFDialect/Analysis/ControlFlowTree.h"
#include "Support/HLCFDialect/HLCFOps.h"

using namespace M;
using namespace HLCF;

ControlFlowTree::ControlFlowTree(Operation *op) {
  unsigned nodeId = 0;
  SmallVector<unsigned> loopIds;
  buildTree(op, nodeId, loopIds);
}

void ControlFlowTree::buildTree(Operation *node, unsigned &nodeId,
                                SmallVectorImpl<unsigned> &loopIds) {
  ops.push_back(node);
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
