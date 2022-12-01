//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H
#define SUPPORT_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H

#include "Support/HLCFDialect/HLCFInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace M::HLCF {
/// This analysis contains information about the control-flow tree rooted at the
/// given operation.
class ControlFlowTree {
public:
  /// Build the tree at the given operation.
  explicit ControlFlowTree(Operation *op);

  /// A map of operation ID to the operation. The ID is the depth-first visit
  /// order of the operation.
  SmallVector<ControlFlowNode> ops;

  /// A map of terminators to their branch target and a flag indicating whether
  /// the target is before or after the operation.
  SmallVector<std::pair<unsigned, SmallVector<ControlFlowTarget, 1>>> targets;

private:
  /// Build the control-flow relations.
  void buildTree(ControlFlowNode node, unsigned &nodeId,
                 SmallVectorImpl<unsigned> &nodeIds);
};
} // namespace M::HLCF

#endif // SUPPORT_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H
