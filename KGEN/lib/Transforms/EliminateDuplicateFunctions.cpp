//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "AsyncRT/Support/ForkJoin.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "llvm/ADT/SCCIterator.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_ELIMINATEDUPLICATEFUNCTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct DuplicateFuncOpEquivalenceInfo : public llvm::DenseMapInfo<FuncOp> {

  static unsigned getHashValue(const FuncOp cFunc) {
    if (!cFunc)
      return DenseMapInfo<FuncOp>::getHashValue(cFunc);

    llvm::hash_code hash = {};
    FuncOp func = const_cast<FuncOp &>(cFunc);
    StringAttr symNameAttrName = func.getSymNameAttrName();
    for (NamedAttribute namedAttr : cFunc->getAttrs()) {
      StringAttr attrName = namedAttr.getName();
      // Ignoring the symbol name.
      if (attrName == symNameAttrName)
        continue;
      hash = llvm::hash_combine(hash, namedAttr);
    }

    func.getBody()->walk([&](Operation *op) {
      hash = llvm::hash_combine(
          hash,
          mlir::OperationEquivalence::computeHash(
              op, /*hashOperands=*/mlir::OperationEquivalence::ignoreHashValue,
              /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
              /*flags=*/mlir::OperationEquivalence::Flags::None));
    });

    return hash;
  }

  static bool isEqual(FuncOp lhs, FuncOp rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    DictionaryAttr lhsDictAttr = lhs->getAttrDictionary();
    DictionaryAttr rhsDictAttr = rhs->getAttrDictionary();
    if (lhsDictAttr.size() != rhsDictAttr.size())
      return false;

    // Other than the symbol name, all other attributes need to be exact same.
    StringAttr symNameAttrName = lhs.getSymNameAttrName();
    for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsDictAttr, rhsDictAttr)) {
      if (lhsAttr.getName() == symNameAttrName && rhsAttr.getName())
        continue;
      if (lhsAttr != rhsAttr)
        return false;
    }

    // Compare function body, we can be more aggressive by using
    // `mlir::OperationEquivalence::Flags::IgnoreLocations`.
    // NOTE: Different constant operation orders will break the equivalence.
    return mlir::OperationEquivalence::isRegionEquivalentTo(
        &lhs.getBodyRegion(), &rhs.getBodyRegion(),
        /*flags=*/mlir::OperationEquivalence::Flags::None);
  }
};

struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, FuncOp, KGENCallOpInterface> {
  using CallGraphNodeBase::CallGraphNodeBase;
  using CallGraphNodeBase::EdgeT;
  CallGraphNode(FuncOp func, AsyncRT::Runtime &runtime)
      : CallGraphNodeBase(func),
        doneDedup(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)) {}
  CallGraphNode(CallGraphNode &&other)
      : CallGraphNodeBase(std::move(other)),
        doneDedup(std::move(other.doneDedup)) {}

  /// Chain value to mark if the deduplication is done or not for synchronizing
  /// CallGraph dependencies.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> doneDedup;

  /// The set of dependencies that has to be deduplicated before we can start
  /// working on this node.
  llvm::SmallPtrSet<CallGraphNode *, 6> dependencies;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  // Construct and build the callgraph
  CallGraph(mlir::ModuleOp module, const SymbolTable &symtable,
            AsyncRT::Runtime &runtime);

  bool shouldAddToGraph(CallOpT call, CallGraphNode *node) {
    // Skip external functions.
    return !node->func.isExternal();
  }

  void deduplicateNode(CallGraphNode *node);

  void startDeduplication();

  void finishHandleNode(CallGraphNode *node) {
    node->doneDedup.copy().emplace();
    if (numWorkItems.fetch_sub(1) == 1)
      done.copy().emplace();
  }

  /// External node that has all the functions that do not have a caller in the
  /// Module as callees. This node is the entry node of the CallGraph for
  /// computing SCCs.
  CallGraphNode externalNode;

  /// Number of remaining work items.
  std::atomic<size_t> numWorkItems;
  /// All work items (i.e. functions might need to be deduplicated).
  SmallVector<CallGraphNode *> workItems;

  /// Reference to the AsyncRT runtime for launch jobs in parallel.
  AsyncRT::Runtime &runtime;

  /// Chain value to mark if all work items are done to mark main thread
  /// (this deduplication pass) to be done.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> done;

  // Map from the target FuncOp to callsites that need to be redirected.
  DenseSet<FuncOp, DuplicateFuncOpEquivalenceInfo> uniqueFuncSet;
  // The mutex to protect uniqueFuncSet accesses.
  std::mutex setAccessMutex;
};

struct EliminateDuplicateFunctionsPass
    : M::KGEN::impl::EliminateDuplicateFunctionsBase<
          EliminateDuplicateFunctionsPass> {
  void runOnOperation() override;
};
} // namespace

namespace llvm {
template <>
struct GraphTraits<CallGraph *> : public GraphTraits<CallGraph::BaseT *> {};
} // namespace llvm

CallGraph::CallGraph(mlir::ModuleOp module, const SymbolTable &symtable,
                     AsyncRT::Runtime &runtime)
    : externalNode(nullptr, runtime), numWorkItems(0), runtime(runtime),
      done(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)),
      uniqueFuncSet() {
  CallGraphBase::build(module, symtable, runtime);

  for (auto &[func, node] : nodes) {
    // We add an edge from externalNode to every other node. Note that it would
    // *NOT* change the topology of the callgraph nor does it introduce extra
    // cycles into the callgraph, as there is no incoming edge to the external
    // node.
    //
    // This is needed because EDF pass runs before DCE, thus there might
    // be unreachable cycles of dead symbols (which can not be easily identified
    // by `callers.empty()`). When that happens, the AsyncValue in those nodes
    // will remain to be uninitialized after the pass finishes. And AsyncRT,
    // unfortunately, does not know how to destruct uninitialized AsyncValue.
    externalNode.callsites.emplace_back(nullptr, &node);
  }

  // Capture the dependencies.
  for (auto scc = llvm::scc_begin(this); scc != llvm::scc_end(this); ++scc) {
    if (scc.hasCycle()) {
      // Cycles need extra handling. Mark them as ready (They will NOT be
      // deduplicated).
      for (CallGraphNode *node : *scc)
        node->doneDedup.copy().emplace();
      continue;
    }

    CallGraphNode *cn = scc->front();
    FuncOp toExam = cn->func;
    if (!toExam) // Skip entry node.
      continue;

    for (CallGraphNode *caller : cn->callers)
      caller->dependencies.insert(cn);

    workItems.push_back(cn);
  }
  // plus the entry node
  numWorkItems = workItems.size() + 1;
}

void CallGraph::deduplicateNode(CallGraphNode *node) {

  SmallVector<AnyAsyncValueRef> waitUtils;
  for (CallGraphNode *dep : node->dependencies)
    waitUtils.emplace_back(dep->doneDedup.copy());

  auto dedupFunc = [this, node](ArrayRef<AnyAsyncValueRef>) mutable {
    for (auto [call, callee] : node->callsites) {
      FuncOp replaceTo(nullptr);
      // Looks up potential replacement.
      {
        std::lock_guard<std::mutex> lock(setAccessMutex);
        auto it = uniqueFuncSet.find(callee->func);
        if (it != uniqueFuncSet.end())
          replaceTo = *it;
      }
      if (replaceTo) {
        call.setCalleeAttr(SymbolConstantAttr::get(replaceTo.getSymNameAttr(),
                                                   replaceTo.getSignature()));
      }
    }

    // Done rewriting the FuncOp, now updates the map.
    {
      std::lock_guard<std::mutex> lock(setAccessMutex);
      uniqueFuncSet.insert(node->func);
    }

    // Mark the node as ready.
    finishHandleNode(node);
  };

  AsyncRT::andThenAsyncMoving(waitUtils, std::move(dedupFunc));
}

void CallGraph::startDeduplication() {
  for (CallGraphNode *node : workItems)
    deduplicateNode(node);

  finishHandleNode(&externalNode);
  AsyncRT::await(done);
}

void EliminateDuplicateFunctionsPass::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope("eliminateDuplicateFunctions");

  mlir::ModuleOp module = getOperation();
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &runtime = *loadContext(&getContext())->get<AsyncRT::Runtime>();

  CallGraph cg(module, symtab, runtime);
  cg.startDeduplication();
  // NOTE: We do not erase the duplicated function in the pass but rely later
  // passes to cleanup. A duplicate function is not always a dead symbol as it
  // might be referenced by operations other than KGENCallOpInterface.
}
