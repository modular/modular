//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_IREVALUATOR_H
#define KGEN_ELABORATOR_IREVALUATOR_H

#include "Cache/CachedTransform.h"
#include "IREvaluatorContext.h"
#include "KGEN/Interpreter/BytecodeInterpreter.h"
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
class IREvaluator : public ParameterEvaluationContext,
                    public ParameterEvaluator,
                    public IREvaluatorContext,
                    public BytecodeInterpreter {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(Elaborator &elaborator, ImplNode *parent);
  IREvaluator(const IREvaluator &other);

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) override;

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

  /// Set the location to associate errors with.
  void setErrorLoc(Location loc) { errorLoc = loc; }

  /// Get compilation error limit from the elaborator.
  int getErrorLimit();

  /// Get error with prelude setting from the elaborator.
  bool getElabErrorIncludePrelude();

private:
  /// Evaluate an apply-like operator.
  FailureOr<TypedAttr> evaluateApplyLike(ParamOperatorAttr op,
                                         bool withResultSlot);
  FailureOr<TypedAttr> evaluateStringAddress(ParamOperatorAttr op);
  FailureOr<TypedAttr> evaluateGetWitnessAttr(GetWitnessAttr getWitnessEntry);

  Attribute getReboundAttribute(Attribute attr) {
    return ParameterEvaluator::getReboundAttribute(attr);
  }
  Type getReboundType(Type type) {
    return ParameterEvaluator::getReboundType(type);
  }

  ParamNodeBase *lookupParamNodeBase(SymbolRefAttr symbol) override;

  /// Compute the expected mangled name of a generator from a parameter.
  /// Returns both the mangled name and the generator referenced by the
  /// parameter. The parameter will be legalized to ensure a SymbolConstantAttr.
  /// If `allowParametric`, any not fully bound symbol reference will just have
  /// its symbol name returned. Otherwise, not fully bound symbols are errors.
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> getExpectedMangledName(
      Location errorLoc, StringRef errorContext, TypedAttr symCst,
      bool allowParametric, bool sanitize,
      function_ref<std::string(StringRef)> getPrefix) override;

  GeneratorOp getGenerator(SymbolRefAttr symbol) override;

  ErrorOr<CrossDeviceFunction>
  compileAsm(MLIRContext *ctx, GeneratorOp, SymbolConstantAttr, StringAttr,
             TargetInfoAttr, EmitAs, EmissionOptions emissionOptions) override;

  void addDeferredFunction(OwningOpRef<FuncOp> func) override;

  ImplNodeBase *getParentNode() override;

  /// A reference to the elaborator instance. The elaborator is invoked to
  /// concretize symbol constants prior to interpreting them.
  Elaborator *elaborator;

  /// The contextual node being elaborated.
  ImplNode *parent = nullptr;
};

//===----------------------------------------------------------------------===//
// ImplNode
//===----------------------------------------------------------------------===//

/// This struct represents a concrete instantiation of a generator -- generators
/// may have multiple concrete instantiations -- and contains the current state
/// of elaboration for that concrete instance.
struct ImplNode : public ImplNodeBase {
  /// Create a new generator implementation node.
  ImplNode(InstantiatedOpInterface inst, ParamNode *parent,
           ParameterUseDefGraph &&graph)
      : ImplNodeBase(inst, std::move(graph)), parent(parent) {}

  ImplNode(ParamNode *parent);
  virtual ~ImplNode() = default;

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err) override;

  /// Get the current active evaluator instance.
  IREvaluator &getEvaluator() { return stack.back().evaluator; }

  /// The parent expansion tree node.
  ParamNode *parent;
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

  /// This is the list of deferred generator instantiations via calls that need
  /// to be handled when the implementation node is complete and all its
  /// dependencies are ready.
  std::vector<std::pair<Location, ParamNode *>> dependencies;

  /// The current downstream node blocking elaboration of this node. E.g. when
  /// elaboration of this node requires elaboration of another node. The blocker
  /// node has to be completed before elaboration of this node can continue.
  std::optional<std::pair<Location, ParamNode *>> blocker;
};

//===----------------------------------------------------------------------===//
// ParamNode
//===----------------------------------------------------------------------===//

/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct ParamNode : public ParamNodeBase {
  /// Create an expansion tree node to represent a generator instantiation.
  ParamNode(AsyncRT::Runtime &runtime, GeneratorOpInterface gen,
            ParameterExprArrayAttr vals, size_t depth,
            ExpansionGraph *expansionGraph)
      : ParamNodeBase(runtime, gen, vals, depth), impl(this),
        expansionGraph(expansionGraph) {
    assert(expansionGraph && "Expansion graph cannot be null");
  }

  /// Create a special root node. Root nodes can be identified with a null
  /// symbol.
  ParamNode() : impl(nullptr, this, ParameterUseDefGraph(nullptr)) {}

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
  ErrorTreeOrSuccess collectErrorsOrSuccess();

  /// The instantiation of the parametric function.
  ImplNode impl;

  /// Add a waiter to the runtime and report task completion to
  /// ParamNodeRuntime.
  void andThenAsync(AsyncValue::Waiter &&waiter);

  /// Construct the async value. This will notify waiters.
  void emplace();

  /// Construct the async value to error state. This is used when we want to
  /// abandon tasks that are not complete.
  void setToError();

private:
  /// The runtime manages the set of tasks kicked off in a given process. The
  /// ParamNode alerts the runtime upon creation and completion of tasks so that
  /// the runtime can sync tasks.
  ExpansionGraph *expansionGraph;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_IREVALUATOR_H
