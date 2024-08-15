//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_IREVALUATOR_H
#define KGEN_ELABORATOR_IREVALUATOR_H

#include "Cache/CachedTransform.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "mlir/Support/IndentedOstream.h"

namespace M::KGEN {
class Elaborator;
class FuncOp;
struct ImplNode;
struct ParamNode;
struct ExpansionGraph;

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

/// This IR evaluator is a parameter evaluator that can work during elaboration
/// to concretize parameter expressions and compute symbolic parameter
/// expressions, such as `apply` on a symbol constant or `get_sizeof` and
/// `get_alignof` a decl type.
class IREvaluator : public ParameterEvaluator, public InterpreterState {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(Elaborator &elaborator, ImplNode *parent);
  IREvaluator(const IREvaluator &other)
      : ParameterEvaluator(other), InterpreterState(other.getTarget()),
        elaborator(other.elaborator), parent(other.parent) {}

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available).
  ErrorTreeOr<Attribute> concretizeParameterExpr(ImplNode *parent, Location loc,
                                                 Attribute expr);
  ErrorTreeOr<Type> concretizeParameterExpr(ImplNode *parent, Location loc,
                                            Type expr);

  /// Lookup the body of the referenced function. Ensure the function is
  /// inflated as well.
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

  /// Evaluate the function with the provided constant inputs.
  ErrorTreeOr<TypedAttr> evaluateFunction(FuncOp func,
                                          ArrayRef<TypedAttr> inputs);

  /// Evaluate the result slot function with the provided constant inputs.
  ErrorTreeOr<TypedAttr>
  evaluateFunctionWithResultSlot(FuncOp func, ArrayRef<TypedAttr> inputs);

private:
  /// Evaluate an apply-like operator.
  FailureOr<TypedAttr> evaluateApplyLike(ParamOperatorAttr op,
                                         bool withResultSlot);
  /// Evaluate a `inst_struct` operator.
  FailureOr<TypeConstantRefAttr>
  evaluateInstantiateStruct(ParamOperatorAttr op);
  /// Evaluate a `get_env` operator.
  FailureOr<TypedAttr> evaluateGetEnv(ParamOperatorAttr op);
  /// Evaluate a `compile_assembly` operator.
  FailureOr<TypedAttr> evaluateCompileAssembly(ParamOperatorAttr op);
  /// Evaluate a `get_linkage_name` operator.
  FailureOr<TypedAttr> evaluateGetLinkageName(ParamOperatorAttr op);

  Attribute getReboundAttribute(Attribute attr) {
    return ParameterEvaluator::getReboundAttribute(attr);
  }
  Type getReboundType(Type type) {
    return ParameterEvaluator::getReboundType(type);
  }

  /// A reference to the elaborator instance. The elaborator is invoked to
  /// concretize symbol constants prior to interpreting them.
  Elaborator *elaborator;

  /// The contextual node being elaborated.
  ImplNode *parent = nullptr;
  /// The contextual location of an error.
  std::optional<Location> errorLoc;
  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;
};

//===----------------------------------------------------------------------===//
// ImplNode
//===----------------------------------------------------------------------===//

/// This struct represents a concrete instantiation of a generator -- generators
/// may have multiple concrete instantiations -- and contains the current state
/// of elaboration for that concrete instance.
struct ImplNode {
  /// Create a new generator implementation node.
  ImplNode(InstantiatedOpInterface inst, ParamNode *parent,
           ParameterUseDefGraph &&graph, std::string &&baseName)
      : inst(inst), parent(parent), paramGraph(std::move(graph)),
        baseName(std::move(baseName)) {}

  /// Create a special root node. Root nodes can be identified with a null
  /// symbol.
  ImplNode(ParamNode *parent);

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err) {
    assert(!error && "impl node already has an error");
    hasError.store(true);
    error = std::move(err);
  }

  /// Get the current active evaluator instance.
  IREvaluator &getEvaluator() { return stack.back().evaluator; }

  /// This op represents a concrete instantiation of a generator.
  InstantiatedOpInterface inst;
  /// The parent expansion tree node.
  ParamNode *parent;
  /// Keep track of the nested parameter scopes within this symbol.
  ParameterUseDefGraph paramGraph;
  /// The base name of the node to use to create derived names. This may differ
  /// from the actual name of the symbol.
  std::string baseName;

  /// An error contained by this node. This allows us to delay error handling in
  /// cases where an error is recoverable.
  std::optional<ErrorTree> error;
  /// An atomic indicating whether an error is present. This can be used to
  /// check for an error when the ImplNode is shared.
  std::atomic<bool> hasError = false;

  struct WorkItem {
    /// The operations to process.
    std::vector<Operation *> ops;
    /// The completion callback. This function is invoked when the processing of
    /// a scope completes. The callback should perform any necessary cleanup and
    /// additional work scheduling if necessary. The callback is passed the
    /// current node that owns the work item, and it is allowed to set errors,
    /// access operations, modify bindings and worklists, etc. It is imperative
    /// that the callback closure does not capture any operation handles but
    /// that it accessing them through the node. This is because nodes can be
    /// cloned and the operations get remapped.
    std::function<LogicalResult(ImplNode *)> onComplete;

    /// The evaluator to use. We need one per work item because each represents
    /// a distinct parameter scope.
    IREvaluator evaluator;
  };

  /// The current stack of worklists and scopes.
  std::vector<WorkItem> stack;

  /// The elaborator will asynchronously dispatch elaboration of generator
  /// instantiations with no result parameters in separate tasks, deferring
  /// handling of the calls until they are complete. This atomic tracks the
  /// number of in-flight so-called "dependencies". Upon hitting zero, the
  /// elaborator will complete processing of this node by handling the calls in
  /// `dependencies`.
  std::atomic<size_t> numDependencies = 1;
  /// This is the list of deferred generator instantiations via calls that need
  /// to be handled when the implementation node is complete and all its
  /// dependencies are ready.
  std::vector<std::pair<GeneratorUserOpInterface, ParamNode *>> dependencies;
  /// Other non-direct-call dependencies, such as through parameter calls. These
  /// dependencies cannot be processed in parallel because they indicate a hard
  /// dependency edge: we need the result to be available to proceed with
  /// elaboration of the current generator.
  std::vector<std::pair<Location, ParamNode *>> otherDeps;
  /// Dependencies that need to be instantiated eventually, but are not needed
  /// for the elaboration of this node. This is used for struct instantiations,
  /// which will be immediately resolved to the concrete symbol reference after
  /// its elaboration is scheduled. Errors from these dependencies are not
  /// propagate back up, so this list is used to keep track.
  /// TODO(MOCO-1055): Merge this with non-type dependencies.
  std::vector<std::pair<Location, ParamNode *>> eventualDeps;
  /// This flag is set when the implementation node is done processing. A
  /// separate flag is needed because an error state can cause the node to
  /// complete early. This flag prevents double-completion.
  std::atomic<bool> done = false;

  /// A chain representing SCC completion.
  AsyncValueRef<Chain> sccCh;
};

//===----------------------------------------------------------------------===//
// ParamNode
//===----------------------------------------------------------------------===//

/// The state of a parameter node, including the number of tasks waiting on it.
/// These are stored together in an atomic so that the waiter count and the
/// state can be transitioned atomically together, preventing erroenous waiters
/// from being counted due to a race.
class ParamNodeState {
  static constexpr uint64_t IN_PROGRESS_BIT = 62;
  static constexpr uint64_t DONE_BIT = 63;

public:
  /// All parameter nodes are created with `FRESH` status. When a worker has
  /// scheduled the first child of the node, it is moved to `IN_PROGRESS`. When
  /// all children complete processing, the state is moved to `DONE`. Use the
  /// upper 2 bits to represent the status.
  enum State { FRESH = 0, IN_PROGRESS = 1, DONE = 3 };

  /// Attempt to mark the status as `IN_PROGRESS`. Return the previous status.
  State markInProgress() {
    return static_cast<State>(
        value.fetch_or(static_cast<uint64_t>(1) << IN_PROGRESS_BIT) >>
        IN_PROGRESS_BIT);
  }

  /// Add a waiter. Return true if the task is not `DONE` and the waiter was
  /// successfully added.
  bool addWaiter() { return (value.fetch_add(1) >> DONE_BIT) == 0; }

  /// Mark the status as done and return the number of waiters at that time.
  size_t markDone() {
    return value.fetch_or(static_cast<uint64_t>(1) << DONE_BIT) << 2 >> 2;
  }

  /// Reset the state of the node to `FRESH`.
  void refresh() {
    value.fetch_xor(static_cast<uint64_t>(1) << IN_PROGRESS_BIT);
  }

  /// Get the current state.
  State getValue() {
    return static_cast<State>(value.load() >> IN_PROGRESS_BIT);
  }

private:
  /// The upper 2 bits of the value represent the status and the lower 62 bits
  /// the number of waiting tasks.
  std::atomic<uint64_t> value = 0;
};

/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct ParamNode {
  /// Create an expansion tree node to represent a generator instantiation.
  ParamNode(AsyncRT::Runtime &runtime, GeneratorOpInterface gen,
            ParameterExprArrayAttr vals, size_t depth,
            ExpansionGraph *expansionGraph)
      : gen(gen), inputParams(vals), depth(depth),
        paramCh(AsyncRT::AsyncValueRef<AsyncRT::Chain>::allocate(runtime)),
        expansionGraph(expansionGraph) {
    assert(expansionGraph && "Expansion graph cannot be null");
  }
  ParamNode() {}

  /// Return the first concrete node in the subtree rooted on `this`. This is
  /// often called from a node that is either concrete, or only has one
  /// concretization. For generality in cases where the full list of concrete
  /// nodes is required, use getAllConcrete below. Returns an error if there are
  /// no concretizations of this node.
  ErrorTreeOr<ImplNode *> getFirstConcreteNode();

  /// Get the first concrete FuncOp. This finds the first concrete node in the
  /// subtree, and returns its op cast to a FuncOp. This is always safe because
  /// if the node has been concretized, then the op is a FuncOp.
  ErrorTreeOr<FuncOp> getFirstConcreteFunc();

  /// Return an error if expansion of this parameter node failed. If any
  /// implementation succeeded, return success instead.
  /// `visited` contains the set of previously visited nodes to prevent cycles.
  ErrorTreeOrSuccess collectErrorsOrSuccess(DenseSet<ParamNode *> &visited);

  /// Return the mangled name of this ParamNode. Calculates it on first
  /// invocation.
  StringAttr getMangledName();

  /// The generator represented by this node.
  GeneratorOpInterface gen;
  /// The input parameters with which the generator is being instantiated.
  ParameterExprArrayAttr inputParams;
  /// The current depth of the node. The depth varies based on the traversal
  /// order of the callgraph.
  size_t depth;

  /// The instantiation of the parametric function.
  std::unique_ptr<ImplNode> impl;

  /// The current state of the node. This flag is used to break recursion.
  ParamNodeState state;

  /// Add a waiter to the runtime and report task completion to
  /// ParamNodeRuntime.
  void andThenAsync(AsyncValue::Waiter &&waiter);

  /// Add a waiter to the runtime.
  void andThenSync(AsyncValue::Waiter &&waiter);

  /// Construct the async value. This will notify waiters.
  void emplace();

  /// Construct the async value to error state. This is used when we want to
  /// abandon uncompleted tasks.
  void setToError();

  /// An async value is ready if the underlying async value is in Active or
  /// Error state.
  bool getIsError() const { return done == DoneState::ERROR; }

  /// Make explicit copy of this AsyncValueRef, increasing the AsyncValue's
  /// refcount by one.
  AsyncValueRef<Chain> copy() const;

private:
  /// The name of the parameterized node.
  StringAttr mangledName;

  /// The chain to signal when this parameter node is done processing.
  AsyncRT::AsyncValueRef<AsyncRT::Chain> paramCh;

  /// Atomic to prevent race on emplace.
  std::atomic<uint8_t> done = false;
  enum DoneState : uint8_t { NOT_DONE, DONE, ERROR };

  /// The runtime manages the set of tasks kicked off in a given process. The
  /// ParamNode alerts the runtime upon creation and completion of tasks so that
  /// the runtime can sync tasks.
  ExpansionGraph *expansionGraph;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATOR_H
