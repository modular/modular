//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_TRANSFORMS_CALLGRAPHUTILS_H
#define KGEN_LIB_TRANSFORMS_CALLGRAPHUTILS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/RWMutex.h"

namespace M::KGEN {

template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdge;
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdgeIterator;

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// A node in a call graph contains a function, edges to its callers, and edges
/// to its callees. A node is ready to inline its callees when all of its
/// callees have been processed.
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphNodeBase {
  using FuncOpT = FuncT;
  using CallOpT = CallT;
  using BaseT = CallGraphNodeBase<DerivedT, FuncT, CallT>;
  using EdgeT = CallGraphEdge<DerivedT, FuncT, CallT>;
  using EdgeIteratorT = CallGraphEdgeIterator<DerivedT, FuncT, CallT>;

  /// Create the node for the given function.
  explicit CallGraphNodeBase(FuncT func) : func(func) {}

  /// This class is only move-constructed when the node map in
  /// `InliningGraphBase` is resized. That occurs before any references are
  /// taken to instances of this object, so just default-construct all other
  /// members of this class.
  CallGraphNodeBase(CallGraphNodeBase &&other) : func(other.func) {}

  /// The function represented by the node.
  FuncOpT func;

  /// Nodes of functions that inline call this function. These are the child
  /// edges.
  SmallVector<DerivedT *> callers;
  /// Calls and callees to inline inside this function. These are the parent
  /// edges.
  SmallVector<std::pair<CallOpT, DerivedT *>> callsites;
  /// This mutex guards `callsites` and `callers` during parallel graph
  /// construction.
  llvm::sys::SmartRWMutex<true> mutex;

  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

/// A callgraph is a graph where the nodes represent functions and the edges
/// represent calls between functions. This is a generic callgraph that provides
/// a build method.
template <typename DerivedT, typename NodeT>
struct CallGraphBase {
  using FuncOpT = typename NodeT::FuncOpT;
  using CallOpT = typename NodeT::CallOpT;

  /// Get a reference to the derived class.
  DerivedT &getDerived() { return *static_cast<DerivedT *>(this); }

  /// Build the inlining graph for a module.
  void build(ModuleOp module, const SymbolTable &symtab);

  /// Dump the callgraph. For debugging.
  void dump();

  /// The nodes in the graph. The map does not resize after it is constructed,
  /// so references always remain valid.
  llvm::MapVector<FuncOpT, NodeT> nodes;
};

template <typename DerivedT, typename NodeT>
void CallGraphBase<DerivedT, NodeT>::build(ModuleOp module,
                                           const SymbolTable &symtab) {
  CompilerTimeTraceScope traceScope("CallGraphBase::build");

  // Instantiate the nodes for each generator first.
  for (auto func : llvm::make_early_inc_range(module.getOps<FuncOpT>()))
    nodes.insert(std::make_pair(func, NodeT(func)));

  // Build the graph by walking all the calls in each function and adding edges
  // as appropriate.
  auto workFn = [this, &symtab](std::pair<FuncOpT, NodeT> &value) {
    auto &[func, node] = value;
    NodeT *callerNode = &node;
    func.getBodyRegion().walk([&](CallOpT call) {
      Operation *calleeOp = symtab.lookup(
          cast<FlatSymbolRefAttr>(
              cast<SymbolConstantAttr>(call.getCallee()).getSymbol())
              .getAttr());
      assert(calleeOp && "invalid IR?");
      // Only add the edge if the symbol we found is of the type we expect.
      auto callee = dyn_cast<FuncOpT>(calleeOp);
      if (!callee)
        return;

      NodeT *calleeNode = &nodes.find(callee)->second;
      // Filter calls that do not satisfy the inlining level.
      if (!getDerived().shouldAddToGraph(call, calleeNode))
        return;
      {
        llvm::sys::SmartScopedWriter<true> lock(callerNode->mutex);
        callerNode->callsites.emplace_back(call, calleeNode);
      }
      {
        llvm::sys::SmartScopedWriter<true> lock(calleeNode->mutex);
        calleeNode->callers.push_back(callerNode);
      }
    });
  };
  mlir::parallelForEach(module.getContext(), nodes, workFn);
}

template <typename DerivedT, typename NodeT>
void CallGraphBase<DerivedT, NodeT>::dump() {
  for (auto &[func, node] : nodes) {
    llvm::errs() << "@" << func.getSymName() << ":\n";
    for (auto [call, callee] : node.callsites)
      llvm::errs() << "  -> @" << callee->func.getSymName() << "\n";
    llvm::errs() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// CallGraphEdgeIterator
//===----------------------------------------------------------------------===//

/// Iterator over the edges in a callgraph. The iterator refers to a node in the
/// callgraph and a specific callsite within the function, representing the edge
/// from one node to the callee function's node.
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdgeIterator {
  using NodeT = CallGraphNodeBase<DerivedT, FuncT, CallT>;
  using ItT = CallGraphEdgeIterator<DerivedT, FuncT, CallT>;
  using RefT = CallGraphEdge<DerivedT, FuncT, CallT>;

  NodeT *node;
  size_t childIdx;

  bool operator==(const ItT &rhs) const {
    return node == rhs.node && childIdx == rhs.childIdx;
  }
  bool operator!=(const ItT &rhs) const { return !(*this == rhs); }
  ItT operator++() {
    ++childIdx;
    return *this;
  }
  ItT operator++(int) {
    ItT tmp = *this;
    ++*this;
    return tmp;
  }
  RefT operator*();
};

/// This struct represents a edge in a callgraph. It contains a callee node and
/// the call operation in the caller from edge originates to the callee node.
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdge {
  using NodeT = CallGraphNodeBase<DerivedT, FuncT, CallT>;
  using ItT = CallGraphEdgeIterator<DerivedT, FuncT, CallT>;
  using RefT = CallGraphEdge<DerivedT, FuncT, CallT>;

  NodeT *node;
  CallT call;

  bool operator==(const RefT &rhs) const {
    return node == rhs.node && call == rhs.call;
  }
  bool operator!=(const RefT &rhs) const { return !(*this == rhs); }

  ItT begin() const { return {node, 0}; }
  ItT end() const { return {node, node->callsites.size()}; }
};

template <typename DerivedT, typename FuncT, typename CallT>
auto CallGraphEdgeIterator<DerivedT, FuncT, CallT>::operator*() -> RefT {
  auto [call, child] = node->callsites[childIdx];
  return {child, call};
}

} // namespace M::KGEN

namespace llvm {
template <typename DerivedT, typename FuncT, typename CallT>
struct DenseMapInfo<M::KGEN::CallGraphEdge<DerivedT, FuncT, CallT>> {
  using EltT = M::KGEN::CallGraphEdge<DerivedT, FuncT, CallT>;
  using NodeT = typename EltT::NodeT;

  static EltT getEmptyKey() {
    return {DenseMapInfo<NodeT *>::getEmptyKey(), nullptr};
  }
  static EltT getTombstoneKey() {
    return {DenseMapInfo<NodeT *>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(const EltT &node) {
    return llvm::hash_combine(
        DenseMapInfo<NodeT *>::getHashValue(node.node),
        DenseMapInfo<mlir::Operation *>::getHashValue(node.call));
  }
  static bool isEqual(const EltT &lhs, const EltT &rhs) { return lhs == rhs; }
};

template <typename DerivedT, typename FuncT, typename CallT>
struct GraphTraits<M::KGEN::CallGraphNodeBase<DerivedT, FuncT, CallT> *> {
  using NodeRef = M::KGEN::CallGraphEdge<DerivedT, FuncT, CallT>;
  using ChildIteratorType = typename NodeRef::ItT;

  static NodeRef getEntryNode(typename NodeRef::NodeT *node) {
    return {node, nullptr};
  }
  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

#endif // KGEN_LIB_TRANSFORMS_CALLGRAPHUTILS_H
