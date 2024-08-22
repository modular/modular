//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_ELABORATOR_H
#define KGEN_ELABORATOR_ELABORATOR_H

#include "AsyncRT/CompilerSupport/AsyncSideEffectMap.h"
#include "Cache/CachedTransform.h"
#include "IREvaluator.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Support/ThreadLocalCache.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

/// This struct represents the expansion of a callgraph during elaboration.
struct ExpansionGraph {
  ExpansionGraph(AsyncRT::Runtime &runtime)
      : worklistCh(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)),
        quiesceChain(
            AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)) {}

  virtual ~ExpansionGraph();

  /// Increment the task count.
  void didAddTask();
  /// Decrement the task count.
  void didCompleteTask();

  /// Map from generator instantiation to expansion tree node.
  Shared<
      llvm::MapVector<std::pair<ParameterExprArrayAttr, GeneratorOpInterface>,
                      std::unique_ptr<ParamNode>>>
      nodes;

  /// Map from concrete function to implementation node.
  Shared<DenseMap<InstantiatedOpInterface, ImplNode *>> concreteNodes;

  /// The current number of tasks scheduled anywhere in the elaborator on the
  /// worklist.
  std::atomic<size_t> numWorkItems = 1;
  /// This chain is signalled when all active work items have completed. This is
  /// used to starve the workqueue before running evaluators, because evaluation
  /// cannot be reliably performed while the compiler is doing work on other
  /// threads.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> worklistCh;

  /// Concrete functions added directly to the expansion graph.
  std::vector<std::unique_ptr<ImplNode>> elaboratedNodes;

  /// Number of outstanding resources created from this runtime.
  std::atomic<size_t> numOutstandingResources = 1;
  /// If quiesce() has been called, the chain it returned. Otherwise null.
  AsyncValueRef<Chain> quiesceChain;

  /// Get or create the node for a generator instantiation.
  ParamNode *getOrCreate(AsyncRT::Runtime &runtime,
                         ParameterExprArrayAttr values,
                         GeneratorOpInterface gen, size_t depth);
};

//===----------------------------------------------------------------------===//
// ElaborationState
//===----------------------------------------------------------------------===//

class ElaborationState {
  enum State { NEW_IMPL_SCOPE, NEW_PARAM_SCOPE, ADVANCE, ERROR };

public:
  /// Return the elaboration state that indicates an operation was successfully
  /// processed.
  static ElaborationState advance() { return ADVANCE; }
  /// Return the elaboration state that indicates elaboration should be
  /// pre-empted by a new frame within a function.
  static ElaborationState skipFrame() { return NEW_IMPL_SCOPE; }
  /// Return the elaboration state that indicates elaboration should be
  /// pre-empted by a new parameter node.
  static ElaborationState skipNode() { return NEW_PARAM_SCOPE; }
  /// Return the elaboration state that indicates a fatal error has occurred
  /// during elaboration.
  static ElaborationState error() { return ERROR; }

  /// Return true if the frame should be skipped.
  bool shouldSkipFrame() const { return state == NEW_IMPL_SCOPE; }
  /// Return true if the node should be skipped.
  bool shouldSkipNode() const { return state == NEW_PARAM_SCOPE; }
  /// Return true if an error occurred.
  bool isError() const { return state == ERROR; }

  /// Allow implicit conversion from `LogicalResult`.
  ElaborationState(LogicalResult result)
      : state(succeeded(result) ? ADVANCE : ERROR) {}

private:
  ElaborationState(State state) : state(state) {}

  State state;
};

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

using ElaboratorCompileAsmFnRef = function_ref<ErrorOr<CrossDeviceFunction>(
    GeneratorOp, SymbolConstantAttr, StringAttr, const SymbolTable &,
    TargetInfoAttr, EmissionKind, mlir::DiagnosticEngine::HandlerID)>;

struct ElaboratorCallbacks {
  /// The functor used to compile a module to assembly.
  ElaboratorCompileAsmFnRef compileAsmFn;
};

/// This class provides the elaborator, which constructs the expansion tree as
/// it walks the IR and specializes operations. This outputs IR that has been
/// fully specialized/concretized, with the appropriate functions
/// multi-versioned.
class Elaborator {
public:
  /// Initialize the elaborator and its symbol table.
  Elaborator(SymbolTable &symtab, ParameterCollector::Analysis &paramCache,
             TargetInfoAttr target, ElaboratorCallbacks callbacks,
             const ElaborateGeneratorsOptions &config,
             mlir::DiagnosticEngine::HandlerID diagHandleID);

  //===--------------------------------------------------------------------===//
  // IREvaluator Interface
  //===--------------------------------------------------------------------===//

  /// Concretize all non-parametric symbol references within the provided
  /// parameter expression.
  ErrorTreeOr<Attribute>
  concretizeSymbolsWithin(Attribute value, ImplNode *parent, Location loc);

  /// Lookup an existing concrete function.
  FuncOp lookupConcreteFunction(SymbolRefAttr symbol) {
    StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
    InstantiatedOpInterface &localResult = (*tlInsts)[name];
    if (!localResult) {
      localResult =
          concreteInsts.read([name](auto &map) { return map.at(name); });
    }
    return cast<FuncOp>(*localResult);
  }

  /// Look up the callee symbol. If it's a FuncOp, return it. Otherwise,
  /// elaborate the generator or interface and return the first concrete
  /// implementation. Return none if the specialization is not ready yet.
  ErrorTreeOr<FuncOp> getConcreteFunction(ImplNode *parent, Location loc,
                                          SymbolConstantAttr symbol);

  /// Look up the callee symbol (for a struct generator). If elaboration is
  /// already in-progress or done, return the concrete symbol reference.
  /// Otherwise, dispatch an elaboration of the struct generator and immediately
  /// return the concrete symbol reference (the symbol may not exist yet, but
  /// its elaboration will have been scheduled).
  ErrorTreeOr<TypeConstantRefAttr>
  getConcreteStructTypeReference(ImplNode *parent, Location loc,
                                 TypeConstantRefAttr typeref);

  /// Get the environment defines.
  EnvAttr getCompilationEnvAttr() const { return env; }

  /// Get the pre-elaboration symbol table we can slice from.
  const SymbolTable &getSliceSymTab() const { return oldSymTab; }

  /// Get the functor for compiling a generator to assembly.
  ElaboratorCompileAsmFnRef getCompileAsmFn() const {
    return callbacks.compileAsmFn;
  }

  /// Add an owned function operation that should be appended to the moydule at
  /// the end of elaboration. This is where generated functions during
  /// elaboration should go.
  void addDeferredFunction(OwningOpRef<FuncOp> func);

  /// Get the target associated with this instance of the elaborator.
  TargetInfoAttr getTarget() const { return target; }
  /// Get the elaborator config.
  const ElaborateGeneratorsOptions &getOptions() const { return config; }

  /// Lookup and return a reference to a cached interpreter result in the
  /// current thread. Populates the value from the global cache if necessary.
  TypedAttr &lookupCachedInterpretation(FuncOp func,
                                        ParameterExprArrayAttr operands) {
    TypedAttr &tlValue = (*tlInterp)[{func, operands}];
    if (!tlValue) {
      tlValue = interpCache.read(
          [func, operands](auto &map) { return map.lookup({func, operands}); });
    }
    return tlValue;
  }
  /// Wriet a cached interpreter result to the global cache.
  void writeGlobalCachedInterpretation(FuncOp func,
                                       ParameterExprArrayAttr operands,
                                       TypedAttr value) {
    interpCache.modify([func, operands, value](auto &map) {
      map.insert({{func, operands}, value});
    });
  }

  //===--------------------------------------------------------------------===//
  // Entry Point
  //===--------------------------------------------------------------------===//

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ModuleOp theModule,
                    ArrayRef<GeneratorOp> primaryGenerators);

private:
  //===--------------------------------------------------------------------===//
  // Utility Functions
  //===--------------------------------------------------------------------===//

  /// Insert an ImplNode for a function that is already concrete.
  void addConcreteFunc(FuncOp func) {
    auto inst = cast<InstantiatedOpInterface>(*func);
    auto node = std::make_unique<ImplNode>(inst, /*parent=*/nullptr,
                                           func.getBodyRegion(),
                                           func.getSymName().str());
    g.concreteNodes.modify(
        [this, inst, node = std::move(node)](auto &map) mutable {
          map.try_emplace(inst, node.get());
          g.elaboratedNodes.push_back(std::move(node));
        });
  }

  /// Once a concrete instance has finished specializing, finish processing the
  /// instance and call the verifier.
  void finalizeInstance(ImplNode *node);

  //===--------------------------------------------------------------------===//
  // Operation Processing
  //===--------------------------------------------------------------------===//

  /// Parameter constants are the only operation that can bridge the parameter
  /// domain into the runtime domain. Use it to root concretization of nested
  /// symbol references.
  template <typename OpT>
  ElaborationState processParamConstantOp(ImplNode *parent, OpT op);

  /// Process a `kgen.param.apply` operation by concretizing its operands and
  /// callee and then invoking the interpreter.
  ElaborationState processParamApplyOp(ImplNode *inode, ParamApplyOp op,
                                       FuncOp func);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  ElaborationState processCallOp(ImplNode *parent,
                                 GeneratorUserOpInterface call);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  ElaborationState processGeneratorUser(GeneratorUserOpInterface user,
                                        SymbolConstantAttr calleeSymbol,
                                        ImplNode *parent);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  ElaborationState processParamIfOp(ImplNode *parent, ParamIfOp op);

  /// Process a param.for op by evaluating the sequence of induction variables
  /// and then instantiating the body for each value of the sequence.
  ElaborationState processParamForOp(ImplNode *parent, ParamForOp op);

  //===--------------------------------------------------------------------===//
  // Worklist
  //===--------------------------------------------------------------------===//

  /// Signal the worklist to tell it a job has completed or has been taken off
  /// the workqueue.
  void signalWorklist() {
    if (g.numWorkItems.fetch_sub(1) == 1)
      g.worklistCh.copy().emplace();
  }
  /// Schedule an implementation node on the AsyncRT work queue.
  void scheduleImplNode(ImplNode *inode);
  /// Process the scopes within an implementation node. This function returns
  /// `success` if all scopes completed processing and the node is completely
  /// elaborated. If the function returns `failure`, that means elaboration of
  /// the node hit a suspension point and must halt before completion.
  LogicalResult processImplNode(ImplNode *inode);
  /// Complete processing of an implementation node. If all dependencies have
  /// been completed, this will process them, performing any required
  /// multi-versioning, and finalize and verify the function, unless an error
  /// has occurred.
  void completeImplNodeProcessing(ImplNode *inode);
  /// Process a worklist of ops. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// scope should be pre-empted with a new scope, returns `skipNode` if
  /// processing the current implementation node should suspended.
  ElaborationState processScope(ImplNode *node, ImplNode::WorkItem &item);
  /// Process a single operation. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// should be pre-empted with a new scope, returns `skipNode` if processing
  /// the current implementation node should be suspended.
  ElaborationState processOp(ImplNode *node, Operation *op);

  //===--------------------------------------------------------------------===//
  // Specialization
  //===--------------------------------------------------------------------===//

  /// Request specialization of the generator at `genNode`. If the node is ready
  /// complete, then the function returns `advance` and the concrete functions
  /// can be retrieved from the node. Otherwise, the function returns
  /// `skipNode`, indicating that elaboration of the current function should be
  /// suspended.
  ElaborationState specializeGenerator(ImplNode *inode, ParamNode *genNode,
                                       Location from, bool addWaiter);

  /// Instantiate a generator reference by retrieving the concrete
  /// implementations of a reference. If this function returns `advance` but
  /// `inputParamKey` is not populated, then the callee is a direct function
  /// reference.
  std::pair<ElaborationState, ImplNode *> instantiateGeneratorReference(
      ImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
      ParameterExprArrayAttr &inputParamKey, GeneratorOpInterface &gen,
      function_ref<bool(ParamNode *)> shouldWait = [](ParamNode *) {
        return true;
      });
  /// Given a user of a completed parameter node, collect concrete
  /// implementations whose bindings are consistent with the current node.
  FailureOr<ImplNode *> collectConcreteImplementations(Location loc,
                                                       ImplNode *parent,
                                                       ParamNode *calleeNode);

  /// Attempt to diagnose concrete recursion and break recursion where possible.
  /// Return true if recursion was broken at least once. The "generation" is
  /// used to know whether a visited flag is valid. It must be at least 1,
  /// because the initial generation is always 0.
  bool diagnoseAndBreakRecursion(unsigned generation,
                                 ArrayRef<ParamNode *> roots);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The target we are compiling code for.
  TargetInfoAttr target;

  /// The elaborator config.
  ElaborateGeneratorsOptions config;

  /// This is the symbol table of the new module.
  Shared<DenseMap<StringAttr, InstantiatedOpInterface>> concreteInsts;
  mlir::ThreadLocalCache<DenseMap<StringAttr, InstantiatedOpInterface>> tlInsts;

  /// This is used to cache interpreter invocations across the elaborator.
  using InterpreterMapTy =
      DenseMap<std::pair<FuncOp, ParameterExprArrayAttr>, TypedAttr>;
  Shared<InterpreterMapTy> interpCache;
  mlir::ThreadLocalCache<InterpreterMapTy> tlInterp;

  /// This symbol table allows efficient lookups across the module.
  SymbolTable &oldSymTab;

  /// The environment defines the module is being elaborated with.
  EnvAttr env;

  /// Hash table of known ParameterUseDefGraphs. This ensures we only compute a
  /// graph once for each generator. This is extra state generated by
  /// specializeGenerator that is *required for correctness* - this will cause
  /// issues with caching (though it would be easy to simply recompute) unless
  /// we create a ParametricNode or something we can use to store these in a
  /// proper data structure.
  Shared<DenseMap<GeneratorOpInterface, std::unique_ptr<ParameterUseDefGraph>>>
      knownGraphs;

  /// The AsyncRT runtime instance to use.
  AsyncRT::Runtime &runtime;

  /// The callgraph being expanded.
  ExpansionGraph g;

  /// This is the cached parameter collector analysis.
  ThreadLocalCache<ParameterCollector::Analysis> paramCache;

  /// Callbacks to use for JIT functionalities.
  ElaboratorCallbacks callbacks;

  /// Deferred generated symbols to append to the module.
  SmallVector<FuncOp> deferredSymbols;

  /// A unique ID for Diagnostic Handle if filter is needed.
  mlir::DiagnosticEngine::HandlerID diagHandlerID;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_ELABORATOR_H
