//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_ANALYSIS_CFG_H
#define KGEN_HLCFDIALECT_ANALYSIS_CFG_H

#include "KGEN/HLCFDialect/HLCFInterfaces.h"

namespace M::HLCF {
/// A node in the CFG is a region in a parent operation or the operation
/// itself.
struct CFGNode {
  CFGNode(ControlFlowNode node, std::optional<unsigned> index)
      : node(node), index(index) {}

  /// The parent operation.
  ControlFlowNode node;
  /// The index of the region in the parent operation, or None if this refers to
  /// the operation itself.
  std::optional<unsigned> index;
};

/// This analysis builds a CFG from a control-flow tree. This analysis is useful
/// for jumping from terminators to their target nodes, or walking backwards
/// from parent operations to their predecessors.
class CFGAnalysis {
public:
  /// Build the analysis starting from the given operation.
  explicit CFGAnalysis(Operation *op);

  /// The predecessors for each node.
  DenseMap<CFGNode, SmallVector<Operation *>> predecessors;

  /// The successors for each node.
  DenseMap<Operation *, SmallVector<CFGNode>> successors;
};
} // namespace M::HLCF

namespace llvm {
template <>
struct DenseMapInfo<M::HLCF::CFGNode> {
  static M::HLCF::CFGNode getEmptyKey() {
    return {nullptr, DenseMapInfo<unsigned>::getEmptyKey()};
  }
  static M::HLCF::CFGNode getTombstoneKey() {
    return {nullptr, DenseMapInfo<unsigned>::getTombstoneKey()};
  }
  static unsigned getHashValue(M::HLCF::CFGNode node) {
    return hash_combine(
        DenseMapInfo<void *>::getHashValue(node.node.getOperation()),
        node.index ? DenseMapInfo<unsigned>::getHashValue(*node.index) : 0);
  }
  static bool isEqual(const M::HLCF::CFGNode &lhs,
                      const M::HLCF::CFGNode &rhs) {
    return std::tie(lhs.node, lhs.index) == std::tie(rhs.node, rhs.index);
  }
};
} // namespace llvm

#endif // KGEN_HLCFDIALECT_ANALYSIS_CFG_H
