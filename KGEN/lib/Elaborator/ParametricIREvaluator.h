//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ELABORATOR_PARAMETRICIREVALUATOR_H
#define KGEN_ELABORATOR_PARAMETRICIREVALUATOR_H

#include "IREvaluator.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/lib/Elaborator/IREvaluatorContext.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/Threading/Shared.h"
#include "mlir/Support/IndentedOstream.h"

namespace M::KGEN {
class ParametricElaborator;
class FuncOp;
struct PImplNode;
struct PParamNode;
struct ParametricExpansionGraph;

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

/// This IR evaluator is a parameter evaluator that can work during elaboration
/// to concretize parameter expressions and compute symbolic parameter
/// expressions, such as `apply` on a symbol constant or `get_sizeof` and
/// `get_alignof` a decl type.
class ParametricIREvaluator : public ParameterEvaluationContext,
                              public IREvaluatorContext,
                              public ParametricParameterEvaluator,
                              public ParametricIRInterpreter {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  ParametricIREvaluator(ParametricElaborator &elaborator, PImplNode *parent);
  ParametricIREvaluator(const ParametricIREvaluator &other);

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr>
  evaluateExpression(ContextuallyEvaluatedAttrInterface attr) override;

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available).
  ErrorTreeOr<Attribute> concretizeParameterExpr(PImplNode *parent,
                                                 Location loc, Attribute expr);
  ErrorTreeOr<Type> concretizeParameterExpr(PImplNode *parent, Location loc,
                                            Type expr);

  /// Lookup the body of the referenced function. Ensure the function is
  /// inflated as well.
  ErrorOr<std::pair<Region *, Operation *>>
  lookupParametricFunctionBody(SymbolRefAttr symbol) override;

  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;

  ErrorOr<Type> lookupFuncTypeGenerator(SymbolRefAttr symbol) override;

  /// Evaluate the function with the provided constant inputs.
  ErrorTreeOr<TypedAttr> evaluateFunction(FuncOp func,
                                          ArrayRef<TypedAttr> inputs);

  ErrorTreeOr<TypedAttr> evaluateGenerator(GeneratorOp func,
                                           ArrayRef<TypedAttr> inputs);

  /// Evaluate the result slot function with the provided constant inputs.
  ErrorTreeOr<TypedAttr>
  evaluateFunctionWithResultSlot(FuncOp func, ArrayRef<TypedAttr> inputs);

  ErrorTreeOr<TypedAttr>
  evaluateGeneratorWithResultSlot(GeneratorOp func, ArrayRef<TypedAttr> inputs);

  /// Set the location to associate errors with.
  void setErrorLoc(Location loc) { errorLoc = loc; }

  Attribute getReboundAttribute(Attribute attr) override {
    if (!isCurrOpParam)
      return attr;
    return getCurrentParamEval().getReboundAttribute(attr);
  }

  Type getReboundType(Type type) override {
    if (!isCurrOpParam)
      return type;
    return getCurrentParamEval().getReboundType(type);
  }

  Type getReboundTypeAlways(Type type) override {
    return getCurrentParamEval().getReboundType(type);
  }

  TypedAttr getReboundAttribute(TypedAttr attr) override {
    if (!isCurrOpParam)
      return attr;
    return getCurrentParamEval().getReboundAttribute(attr);
  }

  void setDeclBinding(Attribute decl, Attribute value,
                      bool overwrite = false) override {
    getCurrentParamEval().setDeclBinding(cast<ParamDeclAttr>(decl), value,
                                         overwrite);
  }

  bool overwriteDeclBinding(Attribute decl, Attribute value) override {
    return overwriteDeclBinding(cast<ParamDeclAttr>(decl), value);
  }

  bool overwriteDeclBinding(ParamDeclAttr decl, Attribute value) {
    return getCurrentParamEval().overwriteDeclBinding(cast<ParamDeclAttr>(decl),
                                                      value);
  }

  TypedAttr getFailableReboundAttribute(TypedAttr attr) override {
    if (!isCurrOpParam)
      return attr;
    return getCurrentParamEval().getFailableReboundAttribute(attr);
  }

  ErrorTreeOr<TypedAttr>
  interpretGenerator(Attribute calleeAttr,
                     llvm::ArrayRef<TypedAttr> paramValues,
                     ArrayRef<Attribute> arguments, Location loc) override;

  ErrorTreeOr<TypedAttr> interpretGeneratorWithResultSlot(
      Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
      ArrayRef<Attribute> arguments, Location loc) override;

  void setDeclBindings(const DenseMap<StringAttr, Attribute> &values) override {
    getCurrentParamEval().setDeclBindings(values);
  }

  void setDeclBindings(Operation *gen, ArrayRef<TypedAttr> values) override;

  void clearParameterCache() override;
  void pushEvalFrame(Operation *op, Region *region,
                     llvm::ArrayRef<TypedAttr> paramValues, int id) override;
  void popEvalFrame() override;
  void popEvalFrame(size_t size) override;
  void dumpParams() override { dump(); }

  void *currentEvaluator() override { return &paramEvaluators.back(); }
  size_t numParamEvals() override { return paramEvaluators.size(); }
  void *currentFrame() override { return &stack.back(); }

  void pushParamValues(llvm::ArrayRef<TypedAttr> values, bool pushFrame,
                       Operation *op = nullptr) override;
  void popParamValues(bool popFrame, Operation *op,
                      Operation *tillOp = nullptr) override;
  void appendParamValues(llvm::ArrayRef<TypedAttr> values, int id,
                         Operation *op) override;

  DenseMap<Operation *, OpSideEffectState> &currOpSideEffectState() override {
    return stack.back().opSideEffectState;
  }

  void setRewritten(const DenseMap<std::pair<size_t, const void *>,
                                   const void *> &value) override {
    getCurrentParamEval().setRewritten(value);
  }

  DenseSet<Operation *> *getParamOps(Operation *op, std::string &name) override;
  void setIsCurrOpParam(Operation *op) override;

  struct FrameParamInfo {
    SmallVector<TypedAttr> paramValues;
    SmallVector<std::pair<Operation *, size_t>> numParamsPerScope;
  };

  SmallVector<FrameParamInfo> frameParamInfos;

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
  FailureOr<TypedAttr>
  evaluateGetLinkageNameAttr(GetLinkageNameAttr getLinkageNameAttr);
  FailureOr<TypedAttr>
  evaluateGetSourceNameAttr(GetSourceNameAttr getSourceNameAttr);
  FailureOr<TypedAttr> evaluateGetTypeNameAttr(GetTypeNameAttr getTypeNameAttr);
  FailureOr<TypedAttr> evaluateTypeConformToTraitAttr(
      TypeConformsToTraitAttr typeConformToTraitAttr);
  FailureOr<TypedAttr> evaluateCompileOffloadClosureAttr(
      CompileOffloadClosureAttr compileOffloadClosureAttr);
  FailureOr<TypedAttr> evaluateCompileAssemblyAttr(CompileAssemblyAttr attr);

  std::string stringifyTypeInstanceRef(TypeInstanceRefAttr instanceRef,
                                       bool qualifiedBuiltins);
  void printParamValue(raw_ostream &os, ParamDeclAttr decl, TypedAttr value,
                       bool qualifiedBuiltins);

  void dump() {
    for (auto pair : getCurrentParamEval().getDeclBindings()) {
      llvm::dbgs() << "[param name]: " << pair.first
                   << " value: " << pair.second << "\n";
    }
  }

  ParametricParameterEvaluator &getCurrentParamEval() {
    return paramEvaluators.back();
  }

  /// A reference to the elaborator instance. The elaborator is invoked to
  /// concretize symbol constants prior to interpreting them.
  ParametricElaborator *elaborator;

  /// The contextual node being elaborated.
  PImplNode *parent = nullptr;
  /// The contextual location of an error.
  std::optional<Location> errorLoc;
  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;

  std::vector<ParametricParameterEvaluator> paramEvaluators;
};

//===----------------------------------------------------------------------===//
// PImplNode
//===----------------------------------------------------------------------===//

/// This struct represents a concrete instantiation of a generator -- generators
/// may have multiple concrete instantiations -- and contains the current state
/// of elaboration for that concrete instance.
struct PImplNode : ImplNodeBase {
  /// Create a new generator implementation node.
  PImplNode(InstantiatedOpInterface inst, PParamNode *parent,
            ParameterUseDefGraph &&graph)
      : ImplNodeBase(inst, std::move(graph)), parent(parent) {}

  PImplNode(PParamNode *parent);

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err);

  /// Get the current active evaluator instance.
  ParametricIREvaluator &getEvaluator() { return stack.back().evaluator; }

  /// The parent expansion tree node.
  PParamNode *parent;

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
    std::function<LogicalResult(PImplNode *)> onComplete;

    /// The evaluator to use. We need one per work item because each represents
    /// a distinct parameter scope.
    ParametricIREvaluator evaluator;
  };

  /// The current stack of worklists and scopes.
  std::vector<WorkItem> stack;

  /// This is the list of deferred generator instantiations via calls that need
  /// to be handled when the implementation node is complete and all its
  /// dependencies are ready.
  std::vector<std::pair<Location, PParamNode *>> dependencies;
  /// The current downstream node blocking elaboration of this node. E.g. when
  /// elaboration of this node requires elaboration of another node. The blocker
  /// node has to be completed before elaboration of this node can continue.
  std::optional<std::pair<Location, PParamNode *>> blocker;

  std::optional<Location> fromLoc;
};

//===----------------------------------------------------------------------===//
// PParamNode
//===----------------------------------------------------------------------===//

/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct PParamNode : public ParamNodeBase {
  /// Create an expansion tree node to represent a generator instantiation.
  PParamNode(AsyncRT::Runtime &runtime, GeneratorOpInterface gen,
             ParameterExprArrayAttr vals, size_t depth,
             ParametricExpansionGraph *expansionGraph)
      : ParamNodeBase(runtime, gen, vals, depth), impl(this),
        expansionGraph(expansionGraph) {
    assert(expansionGraph && "Expansion graph cannot be null");
  }

  /// Create a special root node. Root nodes can be identified with a null
  /// symbol.
  PParamNode() : impl(nullptr, this, ParameterUseDefGraph(nullptr)) {}

  /// Return the first concrete node in the subtree rooted on `this`. This is
  /// often called from a node that is either concrete, or only has one
  /// concretization. For generality in cases where the full list of concrete
  /// nodes is required, use getAllConcrete below. Returns an error if there are
  /// no concretizations of this node.
  ErrorTreeOr<PImplNode *> getFirstConcreteNode();

  /// Get the first concrete FuncOp. This finds the first concrete node in the
  /// subtree, and returns its op cast to a FuncOp. This is always safe because
  /// if the node has been concretized, then the op is a FuncOp.
  ErrorTreeOr<FuncOp> getFirstConcreteFunc();

  /// Return an error if expansion of this parameter node failed. If any
  /// implementation succeeded, return success instead.
  ErrorTreeOrSuccess collectErrorsOrSuccess();

  /// The instantiation of the parametric function.
  PImplNode impl;

  /// Add a waiter to the runtime and report task completion to
  /// ParamNodeRuntime.
  void andThenAsync(AsyncValue::Waiter &&waiter);

private:
  /// The runtime manages the set of tasks kicked off in a given process. The
  /// ParamNode alerts the runtime upon creation and completion of tasks so that
  /// the runtime can sync tasks.
  ParametricExpansionGraph *expansionGraph;

public:
  /// inst init mutex
  llvm::sys::SmartRWMutex<true> mutex;
};

} // namespace M::KGEN

#endif // KGEN_ELABORATOR_PARAMETRICIREVALUATOR_H
