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
#include "llvm/Support/RWMutex.h"

namespace M::KGEN {
/// A node in a call graph contains a function, edges to its callers, and edges
/// to its callees. A node is ready to inline its callees when all of its
/// callees have been processed.
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphNodeBase {
  using FuncOpT = FuncT;
  using CallOpT = CallT;

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
  std::vector<DerivedT *> callers;
  /// Calls and callees to inline inside this function. These are the parent
  /// edges.
  std::vector<std::pair<CallOpT, DerivedT *>> callsites;
  /// This mutex guards `callsites` and `callers` during parallel graph
  /// construction.
  llvm::sys::SmartRWMutex<true> mutex;

  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;
};

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
} // namespace M::KGEN

#endif // KGEN_LIB_TRANSFORMS_CALLGRAPHUTILS_H
