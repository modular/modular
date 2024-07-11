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
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "KGEN/TransformUtils/InliningUtils.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/RWMutex.h"

using namespace M;
using namespace KGEN;

static bool isAlwaysInlineFunction(FuncOp func) {
  return func.getInlineLevel() == InlineLevel::AlwaysNoDebug ||
         func.getInlineLevel() == InlineLevel::Always;
}

namespace {
//===----------------------------------------------------------------------===//
// AlwaysInlineGraphNode
//===----------------------------------------------------------------------===//

struct AlwaysInlineGraphNode
    : CallGraphNodeBase<AlwaysInlineGraphNode, FuncOp, KGENCallOpInterface> {
  explicit AlwaysInlineGraphNode(FuncOp func) : CallGraphNodeBase(func) {}
};

//===----------------------------------------------------------------------===//
// AlwaysInlineGraphNodeGraph
//===----------------------------------------------------------------------===//

struct AlwaysInlineGraph
    : CallGraphBase<AlwaysInlineGraph, AlwaysInlineGraphNode> {
  explicit AlwaysInlineGraph() : externalNode(nullptr) {}

  static bool shouldAddToGraph(KGENCallOpInterface call,
                               AlwaysInlineGraphNode *node) {
    return isAlwaysInlineFunction(node->func);
  }

  LogicalResult diagnoseCycle(ModuleOp module, const SymbolTable &symtab);

  /// External node that has all the functions that do not have a caller in the
  /// Module as callees. This node is the entry node of the CallGraph for
  /// computing SCCs.
  AlwaysInlineGraphNode externalNode;
};

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

struct CallGraphNode
    : CallGraphNodeBase<CallGraphNode, FuncOp, KGENCallOpInterface> {
  explicit CallGraphNode(FuncOp func, AsyncRT::Runtime &runtime)
      : CallGraphNodeBase(func),
        doneInlining(
            AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)) {}
  CallGraphNode(CallGraphNode &&other)
      : CallGraphNodeBase(std::move(other)),
        doneInlining(std::move(other.doneInlining)) {}

  /// If an error occurred during inlining, nodes can end up owning the function
  /// upon destruction. Erase the function.
  ~CallGraphNode() {

    // `func` is null for the root node.
    if (func && isFunctionDead()) {
      func->remove();
      func->erase();
    }
  }

  /// Return true if at the end of processing, the function is dead and will be
  /// erased.
  bool isFunctionDead() {
    return (isAllInlined() || !reachable) && !func.isExported();
  }

  /// Should callee be inlined or not given a threshold.
  bool shouldInlineCallee(CallGraphNode *callee, uint64_t threshold);

  /// Can the caller try to inline the callee so that the caller's
  /// worker for inlining should wait for the callee's to finish
  /// before starting. This is a first step check to establish
  /// parallel inline work dependencies. More inlining heuristics
  /// are defined in the shouldInlineCallee() function.
  bool canInlineCallee(CallGraphNode *callee);

  /// Lazily compute and return the number of operations in the function.
  uint64_t getNumOps() {
    if (numOps != -1)
      return numOps;
    numOps = KGEN::getNumOperations(func);
    return numOps;
  }

  /// Is all callsite of this function inlined,
  /// if so the operation can be erased.
  bool isAllInlined();

  /// The number of callers this callgraph node has.
  std::atomic<size_t> numCallers = 0;

  /// Track the number of times the function has been inlined. Once the counter
  /// reaches the number of callers, the function can be erased.
  std::atomic<size_t> numTimesInlined = 0;

  /// Chain value to mark if inlining is done or not for synchronizing CallGraph
  /// dependencies.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> doneInlining;

  /// Nodes in the same SCC as the current one.
  llvm::SmallPtrSet<CallGraphNode *, 6> sccNodes;

  /// If node is reachable in the CallGraph
  /// (e.g. not in any scc that has no callers outside of the scc.)
  bool reachable = false;

  /// Cached number of operations on the function. If -1, then it has not been
  /// computed yet.
  std::atomic<int64_t> numOps = -1;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

struct CallGraph : CallGraphBase<CallGraph, CallGraphNode> {
  CallGraph(AsyncRT::Runtime &runtime, PerThreadPassManagers &pms,
            std::optional<StringAttr> updateAttrName, bool debugCallsite)
      : externalNode(nullptr, runtime), runtime(runtime), pms(pms),
        updateAttrName(updateAttrName), debugCallsite(debugCallsite),
        numWorkItems(0),
        done(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)) {}

  /// Build the CallGraph.
  void build(ModuleOp module, const SymbolTable &symtab);

  static bool shouldAddToGraph(KGENCallOpInterface call, CallGraphNode *node) {
    ++node->numCallers;
    return true;
  }

  /// Inline callees in caller.
  void inlineNode(CallGraphNode *caller, uint64_t threshold);

  /// Performing inlining on the graph.
  void performInlining(uint64_t threshold);

  /// If inner pipeline for function optimization failed or not.
  std::atomic<bool> innerPipelineFailed = false;

  /// External node that has all the functions that do not have a caller in the
  /// Module as callees. This node is the entry node of the CallGraph for
  /// computing SCCs.
  CallGraphNode externalNode;

  LogicalResult diagnoseAlwaysInliningCycle(ModuleOp module,
                                            const SymbolTable &symtab);

private:
  /// Reference to the LLCL runtime for launch jobs in parallel.
  AsyncRT::Runtime &runtime;

  /// The pass managers to use.
  PerThreadPassManagers &pms;

  /// Complete processing of a function that has inlined all eligible callees.
  void completeFunctionProcessing(CallGraphNode *caller);

  /// When a work item ends, call this function for post-processing.
  void endWork();

  /// How to update debuginfo while inlining.
  /// - Optional has null StringAttr: Update debuginfo immediately.
  /// - Optional has non-null StringAttr: Defer debuginfo update. Tag scopes
  ///   with the StringAttr.
  /// - Optional does not have value: Do not update debuginfo.
  std::optional<StringAttr> updateAttrName;

  /// Whether to insert debug information for inlined callsites.
  bool debugCallsite;

  /// Number of work items (i.e. functions to inline)
  std::atomic<size_t> numWorkItems;

  /// Chain value to mark if all work items are done to mark main thread
  /// (this inlining pass)to be done.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> done;
};

} // namespace

namespace llvm {
template <>
struct GraphTraits<CallGraph *> : public GraphTraits<CallGraph::BaseT *> {};

template <>
struct GraphTraits<AlwaysInlineGraph *>
    : public GraphTraits<AlwaysInlineGraph::BaseT *> {};
} // namespace llvm

LogicalResult AlwaysInlineGraph::diagnoseCycle(ModuleOp module,
                                               const SymbolTable &symtab) {
  build(module, symtab);

  for (auto &[func, node] : nodes) {
    if (node.callers.empty() || func.isExported() ||
        isAlwaysInlineFunction(node.func))
      externalNode.callsites.emplace_back(nullptr, &node);
  }

  llvm::scc_iterator<AlwaysInlineGraph *> sccIt = llvm::scc_begin(this);

  while (!sccIt.isAtEnd() && !sccIt.hasCycle())
    ++sccIt;

  if (sccIt.isAtEnd())
    return success();

  // Build a set of nodes in the SCC for efficient queries.
  DenseSet<AlwaysInlineGraphNode *> sccNodes;
  for (AlwaysInlineGraphNode *node : (*sccIt))
    sccNodes.insert(node);

  // Determine the first cycle we can see in the SCC.
  SmallVector<AlwaysInlineGraphNode::EdgeIteratorT> path;
  DenseSet<AlwaysInlineGraphNode *> nodesInPath;
  AlwaysInlineGraphNode *nextNode = sccIt->back();

  while (nodesInPath.insert(nextNode).second) {
    auto it = nextNode->begin();
    while (!sccNodes.contains(it->node))
      ++it;
    path.push_back(it);
    nextNode = it->node;
  }

  // Okay, emit the errors.
  InFlightDiagnostic diag =
      mlir::emitError(nextNode->func.getLoc())
      << "function has recursive call to 'always_inline' function";
  for (auto &it : path) {
    AlwaysInlineGraphNode::EdgeT node = *it;
    diag.attachNote(node.call.getLoc())
        << (&it == &path.back() ? "call here recurses" : "through call here");
    diag.attachNote(node.node->func.getLoc())
        << (&it == &path.back() ? "back to function here"
                                : "to function marked 'always_inline' here");
  }

  return failure();
}

bool CallGraphNode::canInlineCallee(CallGraphNode *callee) {
  // Always inlines.
  if (callee->func.getInlineLevel() == InlineLevel::Always ||
      callee->func.getInlineLevel() == InlineLevel::AlwaysNoDebug)
    return true;

  // Try to inline callee if it is not in the same SCC as the current node
  // (which is the caller).
  return !sccNodes.contains(callee) && (callee != this) && callee->reachable;
}

bool CallGraphNode::shouldInlineCallee(CallGraphNode *callee,
                                       uint64_t threshold) {
  // Should always inline `always_inline` ones.
  if (callee->func.getInlineLevel() == InlineLevel::Always ||
      callee->func.getInlineLevel() == InlineLevel::AlwaysNoDebug)
    return true;

  // Don't handle functions that are not annotated as automatic.
  if (callee->func.getInlineLevel() != InlineLevel::Automatic)
    return false;

  // Don't inline callee who is in the same scc as current node (caller).
  if (sccNodes.contains(callee))
    return false;

  // TODO: Add more sophisticated heuristics for cost model based inlining
  // strategy.
  return callee->getNumOps() < threshold;
}

bool CallGraphNode::isAllInlined() {
  return numCallers == numTimesInlined && numTimesInlined > 0;
}

void CallGraph::completeFunctionProcessing(CallGraphNode *caller) {
  if (failed(pms.getPassManager().run(caller->func)))
    innerPipelineFailed = true;
  caller->doneInlining.copy().emplace();
  endWork();
}

void CallGraph::endWork() {
  if (numWorkItems.fetch_sub(1) == 1)
    done.copy().emplace();
}

void CallGraph::build(ModuleOp module, const SymbolTable &symtab) {
  CallGraphBase::build(module, symtab, runtime);

  for (auto &[func, node] : nodes) {
    if (node.callers.empty() || func.isExported())
      externalNode.callsites.emplace_back(nullptr, &node);
  }

  for (auto scc = llvm::scc_begin(this); scc != llvm::scc_end(this); ++scc) {
    llvm::SmallPtrSet<CallGraphNode *, 6> sccNodes;
    for (CallGraphNode *node : (*scc)) {
      sccNodes.insert(node);
      node->reachable = true;
    }

    for (CallGraphNode *node : (*scc))
      node->sccNodes = sccNodes;
  }
  // Function nodes plus externalNode
  numWorkItems = nodes.size() + 1;
}

void CallGraph::inlineNode(CallGraphNode *caller, uint64_t threshold) {
  SmallVector<AnyAsyncValueRef> calleeAsynchValues;
  DenseSet<CallGraphNode *> seenNodes;
  // Collect callee dependencies to wait.
  for (auto [_, callee] : caller->callsites) {
    // Don't attempt to inline functions in the SCCs of their callers or those
    // in unreachable SCCs.
    if (!caller->canInlineCallee(callee))
      continue;
    if (seenNodes.insert(callee).second)
      calleeAsynchValues.emplace_back(callee->doneInlining.copy());
  }

  if (calleeAsynchValues.empty()) {
    // If the function has no callees to wait on, run the inner pipeline
    // immediately. Add a task to avoid blocking the main thread.
    runtime.getWorkQueue()->addTask(
        [this, caller] { completeFunctionProcessing(caller); });
    return;
  }

  auto inlineFunc = [caller, threshold,
                     this](ArrayRef<AnyAsyncValueRef>) mutable {
    for (auto [call, callee] : caller->callsites) {
      // Make sure we don't call shouldInlineCallee on a callee that we are not
      // waiting on.
      if (!caller->canInlineCallee(callee))
        continue;

      // Now that the callee has finished processing along with its inner
      // function pipeline, we can run our heuristic on it to determine if we
      // should inline it.
      if (!caller->shouldInlineCallee(callee, threshold))
        continue;

      // Mark callsite location explicitly.
      if (debugCallsite && callee->func.getLocScope())
        OpBuilder(call).create<DebugInfo::LineTableLocOp>(call->getLoc());
      // Inline the callee.
      IRMapping map;
      auto [scope, singleExit] =
          inlineRegion(map, call, callee->func.getBodyRegion());

      maybeUpdateDebugInfo(scope, updateAttrName, singleExit);
      callee->numTimesInlined++;
    }

    completeFunctionProcessing(caller);
  };
  AsyncRT::andThenAsyncMoving(calleeAsynchValues, std::move(inlineFunc));
}

void CallGraph::performInlining(uint64_t threshold) {
  for (auto &[func, node] : nodes)
    inlineNode(&node, threshold);

  // Mark externalNode's work done (since it's not doing anything.)
  externalNode.doneInlining.copy().emplace();
  endWork();
  AsyncRT::await(done);
}

LogicalResult
CallGraph::diagnoseAlwaysInliningCycle(ModuleOp module,
                                       const SymbolTable &symtab) {
  CallGraphBase::build(module, symtab, runtime);

  for (auto &[func, node] : nodes) {
    if (node.callers.empty() || func.isExported())
      externalNode.callsites.emplace_back(nullptr, &node);
  }

  llvm::scc_iterator<CallGraph *> sccIt = llvm::scc_begin(this);

  while (!sccIt.isAtEnd() && !sccIt.hasCycle())
    ++sccIt;

  if (sccIt.isAtEnd())
    return success();

  // Build a set of nodes in the SCC for efficient queries.
  DenseSet<CallGraphNode *> sccNodes;
  for (CallGraphNode *node : (*sccIt))
    sccNodes.insert(node);

  // Determine the first cycle we can see in the SCC.
  SmallVector<CallGraphNode::EdgeIteratorT> path;
  DenseSet<CallGraphNode *> nodesInPath;
  CallGraphNode *nextNode = sccIt->front();

  while (nodesInPath.insert(nextNode).second) {
    auto it = nextNode->begin();
    while (!sccNodes.contains(it->node))
      ++it;
    path.push_back(it);
    nextNode = it->node;
  }

  // Okay, emit the errors.
  InFlightDiagnostic diag =
      mlir::emitError(nextNode->func.getLoc())
      << "function has recursive call to 'always_inline' function";
  for (auto &it : path) {
    CallGraphNode::EdgeT node = *it;
    diag.attachNote(node.call.getLoc())
        << (&it == &path.back() ? "call here recurses" : "through call here");
    diag.attachNote(node.node->func.getLoc())
        << (&it == &path.back() ? "back to function here"
                                : "to function marked 'always_inline' here");
  }

  done.copy().emplace();
  return failure();
}

//===----------------------------------------------------------------------===//
// AutomaticInlinePass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_AUTOMATICINLINE
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct AutomaticInline : impl::AutomaticInlineBase<AutomaticInline> {
  explicit AutomaticInline(
      const AutomaticInlineOptions &options = {},
      const std::function<void(mlir::OpPassManager &)> &buildFuncPasses = {})
      : AutomaticInlineBase(options), buildFuncPasses(buildFuncPasses) {}

  LogicalResult initialize(MLIRContext *ctx) override {
    // Parse the pass pipeline if provided.
    if (!buildFuncPasses) {
      buildFuncPasses = [this](mlir::OpPassManager &pm) {
        (void)mlir::parsePassPipeline(funcPipelineStr, pm);
      };
      return success();
    }
    // Otherwise, convert the pipeline functor to a string so that reproducer
    // generation has the nested passes.
    mlir::OpPassManager pipeline;
    buildFuncPasses(pipeline);
    std::string str;
    llvm::raw_string_ostream os(str);
    pipeline.printAsTextualPipeline(os);
    // Strip `any(...)` from the textual pipeline.
    funcPipelineStr = StringRef(str).drop_front(4).drop_back().str();
    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::OpPassManager pipeline;
    if (buildFuncPasses)
      buildFuncPasses(pipeline);
    else
      (void)mlir::parsePassPipeline(funcPipelineStr, pipeline);
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override;

  /// The function pass pipeline builder.
  std::function<void(mlir::OpPassManager &)> buildFuncPasses;

  /// Get inlining threshold based optimization level.
  uint64_t getInlineThreshold();
};
} // namespace

uint64_t AutomaticInline::getInlineThreshold() {
  // TODO: add better heuristics
  switch (optimizationLevel) {
  case 0:
    return 0;
  case 1:
    return 10;
  case 2:
    return 20;
  case 3:
    return 50;
  default:
    return 50;
  }
}

void AutomaticInline::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  std::optional<StringAttr> updateAttrName;
  switch (updateDebugInfo) {
  case InlinerDebugInfoUpdateTime::kImmediate:
    updateAttrName = StringAttr();
    break;
  case InlinerDebugInfoUpdateTime::kDeferred:
    updateAttrName = StringAttr::get(&getContext(), "inliner_debuginfo_update");
    break;
  case InlinerDebugInfoUpdateTime::kNever:
    break;
  }

  AsyncRT::Runtime &runtime =
      *loadContext(&getContext())->get<AsyncRT::Runtime>();
  PerThreadPassManagers pms(&getContext(), buildFuncPasses);

  AlwaysInlineGraph verifyGraph;
  // Check if there is cyclic always_inline function call chains, and errors out
  // if found.
  if (failed(verifyGraph.diagnoseCycle(
          mlir::OperationPass<ModuleOp>::getOperation(), symtab))) {
    return signalPassFailure();
  }

  CallGraph graph(runtime, pms, updateAttrName, !optimizationLevel.getValue());
  // The CallGraph should be legal for always_inline functions (no cycles).
  // Perform inlining here. For always_inline functions in a non-trivial SCC,
  // there must be a function that is not always_inline which can be used to
  // break the cycle. Don't inline any such functions that is in a non-trivial
  // SCC and is not always_inline.
  // Further optimization can be done by automatically inlining
  // non-always_inline functions in an SCC as long as we can break the
  // recursive cycle of inlining.
  graph.build(getOperation(), symtab);
  graph.performInlining(getInlineThreshold());

  // If any inner function pipeline failed, then fail the overall pass.
  if (graph.innerPipelineFailed)
    return signalPassFailure();

  // If we deferred debuginfo update, do that now.
  if (updateDebugInfo == InlinerDebugInfoUpdateTime::kDeferred) {
    VerboseCompilerTimeTraceScope traceScope("updateDebugInfo");
    AsyncRT::ForkJoin state(runtime);
    for (auto &[func, node] : graph.nodes) {
      if (node.isFunctionDead() || node.callsites.empty())
        continue;
      state.fork(
          [&, func = func] { updateScopeDebugInfo(func, *updateAttrName); });
    }
    state.join();
  }
}

std::unique_ptr<mlir::Pass> KGEN::createAutomaticInline(
    const AutomaticInlineOptions &options,
    std::function<void(mlir::OpPassManager &)> buildFuncPasses) {
  return std::make_unique<AutomaticInline>(options, buildFuncPasses);
}
