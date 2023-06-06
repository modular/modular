//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains core logic to parameterized generators into concrete
// function implementations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Elaborator.h"
#include "Cache/CacheDialect/CachedTransform.h"
#include "Elaborator.h"
#include "IREvaluator.h"
#include "KGEN/CLOptions.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Support/AwaitingMutex.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/STLExtras.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "kgen-elaborator"

static constexpr bool EnableTracing = false;

using namespace M;
using namespace KGEN;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// printParameterValue
//===----------------------------------------------------------------------===//

/// Pretty-print the value of a parameter.
static void printParameterValue(Attribute value, raw_ostream &os) {
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    os << intAttr.getValue();
  } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
    SmallString<32> str;
    floatAttr.getValue().toString(str);
    os << str;
  } else if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(value)) {
    os << dtypeAttr.getDType();
  } else if (auto typeConstant = dyn_cast<ConcreteTypeConstantAttr>(value)) {
    // NOTE: Could use pretty mangling for common cases, e.g. "simd2xf32" or
    // something if these get too verbose.
    os << typeConstant.getValue();
  } else if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(value)) {
    if (auto flat = dyn_cast<FlatSymbolRefAttr>(symbolConstant.getSymbol()))
      os << flat.getValue();
    else
      os << symbolConstant.getSymbol();
  } else if (auto stringConstant = dyn_cast<StringAttr>(value)) {
    os << stringConstant.strref();
  } else {
    os << getParamAsString(value);
  }
}

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
static std::string mangleParameterValues(GeneratorOp generator,
                                         ArrayRef<Attribute> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return generator.getName().str();

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getInputParamsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterValue(value, os);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Logger
//===----------------------------------------------------------------------===//

namespace {
/// This class provides structured support for logging steps within the
/// elaborator.
class Logger {
public:
  /// A wrapper delimited scope that also can form a raw_ostream. This makes it
  /// a tad bit easier to chain logging on a scope.
  // TODO: Upstream this.
  struct DelimitedScope : public mlir::raw_indented_ostream::DelimitedScope {
    using mlir::raw_indented_ostream::DelimitedScope::DelimitedScope;
    operator mlir::raw_indented_ostream &() { return os; }

    template <typename Arg>
    mlir::raw_indented_ostream &operator<<(Arg &&arg) {
      LLVM_DEBUG(os << std::forward<Arg>(arg));
      return os;
    }
  };

  Logger() : os(llvm::dbgs()) {}

  /// Start a new logging scope, using the provided arguments to form a message
  /// on the title line of the scope.
  template <typename... TitleLineArgs>
  DelimitedScope scope(TitleLineArgs... titleLineArgs) {
    LLVM_DEBUG({
      ((os << titleLineArgs), ...);
      return DelimitedScope(os, " {\n", "}\n");
    });
    return DelimitedScope(os, /*open=*/"", /*close=*/"", /*indent=*/false);
  }

  /// Log the given operation.
  void logOp(StringRef title, Operation *op) {
    LLVM_DEBUG({
      auto _ = scope(title);
      op->print(os);
      os << "\n";
    });
  }

  template <typename Arg>
  Logger &operator<<(Arg &&arg) {
    LLVM_DEBUG(os << std::forward<Arg>(arg));
    return *this;
  }

  operator mlir::raw_indented_ostream &() { return os; }

private:
  mlir::raw_indented_ostream os;
};
} // namespace

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

namespace {
struct ParamNode;

/// This struct represents a concrete instantiation of a generator -- generators
/// may have multiple concrete instantiations -- and contains the current state
/// of elaboration for that concrete instance.
struct ImplNode {
  /// Create a new generator implementation node.
  ImplNode(FuncOp func, ParamNode *parent, ParameterUseDefGraph &&graph,
           std::string &&baseName)
      : func(func), parent(parent), paramGraph(std::move(graph)),
        baseName(std::move(baseName)) {}

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err) {
    assert(!error && "impl node already has an error");
    error = std::move(err);
  }

  /// Print this tree to the provided indented stream. This preserves any
  /// indentation provided by the caller to make it possible to nest things
  /// nicely.
  void print(mlir::raw_indented_ostream &os, bool printBindings = true);

  /// Get the current active evaluator instance.
  IREvaluator &getEvaluator() {
    assert(!stack.empty() && "empty work stack");
    return stack.back().evaluator;
  }

  /// This function represents a concrete instantiation of a generator.
  FuncOp func;
  /// The parent expansion tree node.
  ParamNode *parent;
  /// Keep track of the nested parameter scopes within this function.
  ParameterUseDefGraph paramGraph;
  /// The base name of the node to use to create derived names. This may differ
  /// from the actual name of the function.
  std::string baseName;

  /// Calls to the same interface/generator should resolve to the same thing in
  /// each func.
  /// FIXME(#14998): Propagating bindings up the expansion graph is superlinear
  /// with respect to the depth of the callgraph, because each node retains a
  /// copy of the map, and each caller node takes the union of its callees'
  /// maps.
  DenseMap<std::pair<ArrayAttr, GeneratorOp>, ImplNode *> bindings;
  /// When you have result parameters, we need to store them to access them from
  /// outer scopes.
  ArrayAttr resultParams;
  /// An error contained by this node. This allows us to delay error handling in
  /// cases where an error is recoverable.
  std::optional<ErrorTree> error;

  struct WorkItem {
    /// The operations to process.
    std::vector<Operation *> ops;
    /// The evaluator to use.
    IREvaluator evaluator;
    /// The completion callback. This function is invoked when the processing of
    /// a scope completes. The callback should perform any necessary cleanup and
    /// additional work scheduling if necessary. The callback is passed the
    /// current node that owns the work item, and it is allowed to set errors,
    /// access operations, modify bindings and worklists, etc. It is imperative
    /// that the callback closure does not capture any operation handles but
    /// that it accessing them through the node. This is because nodes can be
    /// cloned and the operations get remapped.
    std::function<LogicalResult(ImplNode *)> onComplete;
  };

  /// The current stack of worklists and scopes.
  std::vector<WorkItem> stack;
};

void ImplNode::print(mlir::raw_indented_ostream &os, bool printBindings) {
  os << "ImplNode <" << func.getSymName() << ">";
  auto _ = os.scope(" {\n", "}\n");
  if (func) {
    auto opScope = os.scope("Op: {\n", "}\n");
    func.print(os);
    os << "\n";
  }
  if (resultParams && !resultParams.empty())
    os << "ResultParams: " << resultParams << "\n";
  {
    auto _ = os.scope("Bindings: {\n", "}\n");
    for (const auto &[_, bind] : bindings) {
      if (bind != this)
        bind->print(os, false);
      else
        os << "Self\n";
    }
  }
  // Errors are leaves.
  if (error) {
    // Only print the top level error.
    os << "Error: " << error->getError() << "\n";
    return;
  }
}

/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct ParamNode {
  /// Create an expansion tree node to represent a generator instantiation.
  ParamNode(LLCL::Runtime &runtime, GeneratorOp gen, ArrayAttr vals,
            size_t depth)
      : gen(gen), inputParams(vals), depth(depth),
        paramCh(LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime)) {}

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

  /// Get all the concrete nodes in the tree rooted on `this`. This is useful
  /// when you have something like a GeneratorInterface that can concretize to
  /// multiple valid generators, and from that multiple functions.
  void getAllConcreteNodes(std::vector<ImplNode *> &nodes);

  /// Get all the concrete functions in the tree rooted on `this`. Exactly the
  /// same as `getAllConcreteNodes` above, but only returns the FuncOp. Useful
  /// when you don't need the full ParamNode.
  void getAllConcreteFuncs(std::vector<FuncOp> &funcs);

  /// Print this tree to the provided indented stream. This preserves any
  /// indentation provided by the caller to make it possible to nest things
  /// nicely.
  void print(mlir::raw_indented_ostream &os, bool printBindings = true);

  /// The generator represented by this node.
  GeneratorOp gen;
  /// The input parameters with which the generator is being instantiated.
  ArrayAttr inputParams;
  /// The current depth of the node. The depth varies based on the traversal
  /// order of the callgraph.
  size_t depth;

  /// Generators fail immediately if their constrants are not satisfied.
  /// Constraints are only functions of the input parameters. Save the error
  /// here if that happens.
  std::optional<ErrorTree> constraintError;

  /// The children of a node are specializations. They may not be fully concrete
  /// in the case of e.g. an interface - where the children are generators that
  /// themselves have children.
  std::vector<std::unique_ptr<ImplNode>> impls;
  /// The mutex for accessing the implementation list.
  llvm::sys::SmartRWMutex<true> implsMutex;

  /// This set contains all parameter nodes that have children which could call
  /// into this node while the node has not been fully processed. This set is
  /// used to perform cycle detection, because recursive calls need to be
  /// specially handled.
  /// FIXME: Like the bindings map, this set grows with the depth of the
  /// callgraph, and copying the set into children nodes is superlinear.
  DenseSet<ParamNode *> incoming;

  /// The number of in-progress implementations.
  std::atomic<size_t> numActive = 0;

  /// The chain to signal when this parameter node is done processing.
  LLCL::AsyncValueRef<LLCL::Chain> paramCh;

  /// All nodes are created with `FRESH` status. When a worker has scheduled
  /// the first child of the node, it is moved to `IN_PROGRESS`. When all
  /// children complete processing, the state is moved to `DONE`.
  enum Status { FRESH, IN_PROGRESS, DONE };

  /// The current state of the node. This flag is used to break recursion.
  std::atomic<Status> status = FRESH;
};

void ParamNode::print(mlir::raw_indented_ostream &os, bool printBindings) {
  os << "ImplNode <" << gen.getSymName() << ">";
  auto _ = os.scope(" {\n", "}\n");
  {
    auto opScope = os.scope("Op: {\n", "}\n");
    gen.print(os);
    os << "\n";
  }
  if (inputParams && !inputParams.empty())
    os << "InputParams: " << inputParams << "\n";

  // Print the children.
  if (!impls.empty()) {
    auto childrenScope = os.scope("Children: {\n", "}\n");
    for (ImplNode &child : llvm::make_pointee_range(impls))
      child.print(os);
  }
}

/// This struct represents the expansion of a callgraph during elaboration.
struct ExpansionGraph {
  ExpansionGraph(LLCL::Runtime &runtime)
      : worklistCh(LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime)) {}

  /// Map from generator instantiation to expansion tree node.
  Shared<
      DenseMap<std::pair<ArrayAttr, GeneratorOp>, std::unique_ptr<ParamNode>>>
      nodes;

  /// Map from concrete function to implementation node.
  Shared<DenseMap<FuncOp, ImplNode *>> concreteNodes;

  /// The current number of tasks scheduled anywhere in the elaborator on the
  /// worklist.
  std::atomic<size_t> numWorkItems = 1;
  /// This chain is signalled when all active work items have completed. This is
  /// used to starve the workqueue before running evaluators, because evaluation
  /// cannot be reliably performed while the compiler is doing work on other
  /// threads.
  LLCL::AsyncValueRef<LLCL::Chain> worklistCh;
  /// Gate mutex for the evaluation critical section.
  llvm::sys::SmartRWMutex<true> evaluatorMutex;

  /// Fork the expansion of a concrete node.
  ImplNode *fork(ImplNode *cur, IRMapping &map, Shared<SymbolTable &> &symtab,
                 StringRef forkParam, Attribute value) {
    // Clone the function and generate a unique name for it.
    auto clone = cast<FuncOp>(cur->func->clone(map));
    std::string name = cur->baseName;
    llvm::raw_string_ostream os(name);
    os << ',';
    if (!forkParam.empty())
      os << forkParam << '=';
    printParameterValue(value, os);
    clone.setSymName(name);
    // Insert the new function at a location relative to the current one. This
    // ensures all forks are inserted in a deterministic order, regardless of
    // which occur first.
    symtab.modify([clone, it = cur->func->getIterator()](SymbolTable &symtab) {
      symtab.insert(clone, std::next(it));
    });

    // Fork the node and its bindings.
    auto n = std::make_unique<ImplNode>(
        clone, cur->parent, cur->paramGraph.copy(map), std::move(name));
    n->bindings = cur->bindings;

    // Copy over the current work stack.
    for (const ImplNode::WorkItem &item : cur->stack) {
      std::vector<Operation *> clonedOps;
      for (Operation *op : item.ops)
        clonedOps.push_back(map.lookup(op));
      n->stack.push_back(ImplNode::WorkItem{std::move(clonedOps),
                                            item.evaluator, item.onComplete});
    }

    // Track the new node as a new child and concrete node.
    ImplNode *result = n.get();
    ParamNode *p = cur->parent;
    {
      // Multiple forks can happen at the same time.
      llvm::sys::SmartScopedWriter<true> guard(p->implsMutex);
      p->impls.push_back(std::move(n));
    }
    concreteNodes.modify([clone, result](DenseMap<FuncOp, ImplNode *> &map) {
      map.try_emplace(clone, result);
    });
    return result;
  }

  /// Get or create the node for a generator instantiation.
  ParamNode *getOrCreate(LLCL::Runtime &runtime, ArrayAttr values,
                         GeneratorOp gen, size_t depth) {
    // TODO: Split this into `get` and `create` methods, so that some can be
    // read-only accesses.
    return nodes.modify([&](DenseMap<std::pair<ArrayAttr, GeneratorOp>,
                                     std::unique_ptr<ParamNode>> &map) {
      std::unique_ptr<ParamNode> &n = map[{values, gen}];
      if (!n)
        n = std::make_unique<ParamNode>(runtime, gen, values, depth);
      return n.get();
    });
  }
};

} // namespace

ErrorTreeOr<ImplNode *> ParamNode::getFirstConcreteNode() {
  if (impls.empty())
    return ErrorTree(gen.getLoc(), "no viable expansions found");

  ErrorTree err(gen.getLoc(), "no successful concrete nodes");
  for (ImplNode &impl : llvm::make_pointee_range(impls)) {
    if (!impl.error)
      return &impl;
    err.addCause(impl.error->copy());
  }
  return std::move(err);
}

ErrorTreeOr<FuncOp> ParamNode::getFirstConcreteFunc() {
  ErrorTreeOr<ImplNode *> impl = getFirstConcreteNode();
  if (impl.isError())
    return impl.takeError();
  return impl.takeValue()->func;
}

void ParamNode::getAllConcreteNodes(std::vector<ImplNode *> &nodes) {
  for (ImplNode &impl : llvm::make_pointee_range(impls))
    if (!impl.error)
      nodes.push_back(&impl);
}

void ParamNode::getAllConcreteFuncs(std::vector<FuncOp> &funcs) {
  for (ImplNode &impl : llvm::make_pointee_range(impls))
    if (!impl.error)
      funcs.push_back(impl.func);
}

//===----------------------------------------------------------------------===//
// processParamDeclareOp
//===----------------------------------------------------------------------===//

/// Process a param.declare op by setting its parameter value in the provided
/// evaluator.
static LogicalResult processParamDeclareOp(ImplNode *inode, ParamDeclareOp op) {
  // Simplify the input expression.
  auto errorOrValue =
      inode->getEvaluator().concretizeParameterExpr(op.getLoc(), op.getValue());
  if (errorOrValue.isError()) {
    inode->setToError(errorOrValue.takeError());
    return failure();
  }

  // Bind it to the parameter declaration it is setting.
  inode->getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(),
                                                     errorOrValue.takeValue());

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// processParamResultBindOp
//===----------------------------------------------------------------------===//

/// Process a `kgen.param.result_bind` operation by setting the result parameter
/// values of the parent operation.
static LogicalResult processParamResultBindOp(ImplNode *node,
                                              ParamResultBindOp op) {
  // Concretize the result parameter values.
  IREvaluator &evaluator = node->getEvaluator();
  SmallVector<Attribute> resultParams;

  // Retrieve the required parameter decls from the nearest declaration.
  // However, if it refers to the function being elaborated, the declarations
  // are in the generator.
  ArrayRef<ParamDeclAttr> resultParamDecls;
  auto parentDecl = op->getParentOfType<DeclInterface>();
  bool isFunc = isa<FuncOp>(parentDecl.getOperation());
  if (isFunc)
    resultParamDecls = node->parent->gen.getResultParams();
  else
    resultParamDecls = parentDecl.getResultParams();

  for (auto [decl, value] : llvm::zip(resultParamDecls, op.getParameters())) {
    ErrorTreeOr<Attribute> concValue =
        evaluator.concretizeParameterExpr(op.getLoc(), value);
    if (concValue.isError()) {
      node->setToError(concValue.takeError());
      return failure();
    }
    resultParams.push_back(concValue.takeValue());
    evaluator.setOrOverwriteParameterValue(decl, resultParams.back());
  }

  // If this operation binds values for the result parameters of the generator,
  // set them in the node.
  if (isFunc)
    node->resultParams = ArrayAttr::get(op.getContext(), resultParams);

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// processRebindOp
//===----------------------------------------------------------------------===//

static LogicalResult processRebindOp(ImplNode *inode, RebindOp op) {
  ErrorTreeOr<Type> outType =
      inode->getEvaluator().concretizeParameterExpr(op.getLoc(), op.getType());
  if (outType.isError()) {
    inode->setToError(outType.takeError());
    return failure();
  }
  if (outType.getValue() != op.getInput().getType()) {
    inode->setToError(ErrorTree(op.getLoc(),
                                "operand and result type of rebind operation "
                                "did not concretize to the same type"));
    return failure();
  }
  op.replaceAllUsesWith(op.getOperand());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// processParamAssertOp
//===----------------------------------------------------------------------===//

/// Process a param.assert op by folding its parameter expression and checking
/// its constraint. Returns the appropriate error if the constraint failed.
static LogicalResult processParamAssertOp(ImplNode *inode, ParamAssertOp op) {
  // Check the condition expression.
  auto errorOrValue =
      inode->getEvaluator().concretizeParameterExpr(op.getLoc(), op.getCond());
  if (errorOrValue.isError()) {
    inode->setToError(errorOrValue.takeError());
    return failure();
  }

  // If the constraint evaluated to zero then the assert fails.
  auto resultInt = cast<IntegerAttr>(errorOrValue.takeValue());
  if (resultInt.getValue().isZero()) {
    // Evaluate the string to report it.
    errorOrValue = inode->getEvaluator().concretizeParameterExpr(
        op.getLoc(), op.getMessage());

    StringAttr message;
    if (!errorOrValue.isError())
      message = dyn_cast<StringAttr>(errorOrValue.takeValue());

    inode->setToError(ErrorTree(
        op.getLoc(),
        "constraint failed: " + (message ? message.getValue() : "<unknown>")));
    return failure();
  }

  // The kgen.param.assert op serves no further purpose, so we can remove it.
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// processLocation
//===----------------------------------------------------------------------===//

/// Handle location concretization.
static ErrorTreeOrSuccess processLocation(IREvaluator &evaluator,
                                          Operation *op) {
  ErrorTreeOr<Attribute> value = evaluator.concretizeParameterExpr(
      op->getLoc(), op->getLoc(), /*allowUnknown=*/true);
  if (value.isError())
    return value.takeError();
  op->setLoc(cast<Location>(value.takeValue()));
  return success();
}

//===----------------------------------------------------------------------===//
// processGenericOp
//===----------------------------------------------------------------------===//

/// Unknown operations are allowed to use types and attributes with parameter
/// references.  Substitute in concrete values for their references.
static LogicalResult processGenericOp(ImplNode *inode, Operation *op) {
  IREvaluator &evaluator = inode->getEvaluator();

  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    ErrorTreeOr<Attribute> value = evaluator.concretizeParameterExpr(
        op->getLoc(), namedAttr.getValue(), /*allowUnknown=*/true);
    if (value.isError()) {
      inode->setToError(value.takeError());
      return failure();
    }

    newAttrs.emplace_back(namedAttr.getName(), value.takeValue());
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  if (ErrorTreeOrSuccess err = processLocation(evaluator, op); err.isError()) {
    inode->setToError(err.takeError());
    return failure();
  }

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults()) {
    ErrorTreeOr<Type> type =
        evaluator.concretizeParameterExpr(op->getLoc(), result.getType());
    if (type.isError()) {
      inode->setToError(type.takeError());
      return failure();
    }
    result.setType(type.takeValue());
  }

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Value arg : block.getArguments()) {
        ErrorTreeOr<Type> type =
            evaluator.concretizeParameterExpr(op->getLoc(), arg.getType());
        if (type.isError()) {
          inode->setToError(type.takeError());
          return failure();
        }
        arg.setType(type.takeValue());
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// inlineCallToConcreteRegion
//===----------------------------------------------------------------------===//

/// Inline a call to a concretized region. This will clone the ops from the
/// callee into the caller, it replaces the call's uses with the inlined values,
/// and it erases the call. This also handles async calls correctly by creating
/// an AsyncExecuteOp and inlining the body into that.
static void inlineCallToConcreteRegion(KGENCallOpInterface call, Region *callee,
                                       IRMapping &map,
                                       AlwaysInlineLevel alwaysInlineLevel) {
  assert(callee->hasOneBlock() &&
         "callee region must resolve to a single block");

  OpBuilder b(call);
  Operation *scope = nullptr;
  if (isa<CallOp, CallParamOp>(&*call)) {
    // No scope.
  } else if (auto asyncCall = dyn_cast<LIT::AsyncCallOp>(&*call)) {
    scope = b.create<LIT::AsyncExecuteOp>(call.getLoc(), asyncCall.getType());
    b.createBlock(&scope->getRegions().front());
  } else if (auto createClosure = dyn_cast<CreateClosureOp>(&*call)) {
    scope = b.create<StageClosureOp>(call.getLoc(), createClosure.getType());
    b.createBlock(&scope->getRegions().front());
    for (BlockArgument arg :
         callee->getArguments().drop_front(createClosure.getCaptures().size()))
      b.getInsertionBlock()->addArgument(arg.getType(), arg.getLoc());
  } else {
    llvm::report_fatal_error("unhandled call operation in elaborator '" +
                             call->getName().getStringRef() + "'");
  }

  for (auto [callArg, genArg] :
       llvm::zip(call->getOperands(), callee->getArguments()))
    map.map(map.lookupOrDefault(genArg), callArg);

  Operation *terminator = callee->front().getTerminator();
  // Handle debug info as we clone.
  for (Operation &op : callee->front()) {
    // Don't copy DebugInfo::ValueOp ops when we have no debug info.
    if (auto value = dyn_cast<DebugInfo::ValueOp>(&op);
        value && alwaysInlineLevel == AlwaysInlineLevel::EnabledNoDebug)
      continue;

    Operation *cloned = b.clone(op, map);
    // Walk the cloned op because there might be many ops within it.
    cloned->walk([&](Operation *clonedOp) {
      // Erase nested DebugInfo::ValueOp.
      if (isa<DebugInfo::ValueOp>(clonedOp))
        return clonedOp->erase();

      // Update locations to be CallSiteLoc.
      if (alwaysInlineLevel == AlwaysInlineLevel::EnabledNoDebug)
        clonedOp->setLoc(call.getLoc());
      else
        clonedOp->setLoc(
            mlir::CallSiteLoc::get(clonedOp->getLoc(), call.getLoc()));
    });
  }

  Operation *returnOp = map.lookup(terminator);
  // If the remapped return isn't parented under the call's region, then we know
  // it's inside another scope - so use the results of that scope.
  if (scope) {
    if (auto asyncExec = dyn_cast<LIT::AsyncExecuteOp>(scope)) {
      // Replace the returnOp with a LIT::AsyncReturnOp.
      returnOp->replaceAllUsesWith(b.create<LIT::AsyncReturnOp>(
          returnOp->getLoc(), returnOp->getOperands()));
    } else if (isa<StageClosureOp>(scope)) {
      returnOp->replaceAllUsesWith(
          b.create<ReturnOp>(returnOp->getLoc(), returnOp->getOperands()));
    } else {
      llvm::report_fatal_error("unhandled call operation in elaborator '" +
                               call->getName().getStringRef() + "'");
    }
    // And replace the call uses with the results of the AsyncExecuteOp
    // itself.
    call->replaceAllUsesWith(scope->getResults());
  } else {
    call->replaceAllUsesWith(returnOp->getOperands());
  }
  returnOp->erase();
  call->erase();
}

//===----------------------------------------------------------------------===//
// getMangledRegionParamName
//===----------------------------------------------------------------------===//

/// Mangles a region parameter's name with its func parent in order to get a
/// unique name. This is necessary because we need to ensure we inline the
/// region from the correct parent when we're doing inlining.
static StringAttr getMangledRegionParamName(ParamDeclareRegionOp decl) {
  auto parentFunc = decl->getParentOfType<FuncOp>();
  assert(parentFunc && "The parent must be a FuncOp");
  // Construct a name from the region's parameter decl and the parent func. This
  // is required to ensure we get the right region when we resolve it.
  std::string paramName = decl.getParamDecl().getName().getValue().str() + "_" +
                          parentFunc.getNameAttr().getValue().str();
  return StringAttr::get(decl.getContext(), paramName);
}

//===----------------------------------------------------------------------===//
// collectOpsToProcess
//===----------------------------------------------------------------------===//

/// This simply walks the ParameterUseDefGraph and collects the list of ops that
/// need to be rewritten.
static void collectOpsToProcess(Region *scope, const ParameterUseDefGraph &uses,
                                std::vector<Operation *> &opsToRewrite) {
  // FIXME: The elaborator does not correctly handle the new parameter use-def
  // graph. Process the parameters in reverse: the same operation can define
  // multiple parameters, so punt those according to their most dominated
  // definition.
  opsToRewrite.reserve(opsToRewrite.size() + uses.params.size() +
                       uses.paramOps.size());
  llvm::SetVector<Operation *, SmallVector<Operation *, 8>,
                  SmallPtrSet<Operation *, 8>>
      defOps;
  for (StringAttr param : llvm::reverse(uses.params)) {
    auto it = uses.defs.find(param);
    assert(it != uses.defs.end());
    // Ignore the scope parent operation. Input parameters are set contextually.
    if (it->second.defOp == scope->getParentOp())
      continue;
    defOps.insert(it->second.defOp);
  }
  llvm::append_range(opsToRewrite, defOps.getArrayRef());
}

//===----------------------------------------------------------------------===//
// ElaborationState
//===----------------------------------------------------------------------===//

namespace {
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
// ElaboratorImpl Declaration
//===----------------------------------------------------------------------===//

/// This class provides the elaborator, which constructs the expansion tree as
/// it walks the IR and specializes operations. This outputs IR that has been
/// fully specialized/concretized, with the appropriate functions
/// multi-versioned.
class ElaboratorImpl : public Elaborator {
public:
  ElaboratorImpl(SymbolTable &symtab, ParameterCollector::Analysis &paramCache,
                 TargetInfoAttr target,
                 EvaluatorExecutorFnRef evaluatorExecutorFn,
                 LLCL::Runtime &runtime, const ElaboratorConfig &config)
      : Elaborator(symtab, target), config(config), g(runtime),
        paramCache(paramCache, runtime.getWorkQueue()->getParallelismLevel()),
        evaluatorExecutorFn(evaluatorExecutorFn), runtime(runtime) {}

  ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, FlatSymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) override;

  ErrorTreeOrSuccess
  getAllConcreteFunctions(Location loc, FlatSymbolRefAttr symbolRef,
                          ArrayRef<TypedAttr> paramValues,
                          std::vector<FuncOp> &funcs) override;

  /// Implement the evaluator hook. This function ensures that all active work
  /// items on the workqueue are completed or suspended before running the
  /// evaluator, to ensure that, at least with respect to this compiler
  /// instance, the machine is quiet.
  ErrorOr<size_t> evaluateFunctions(FuncOp evaluator,
                                    ArrayRef<FuncOp> options) override;

  /// Once a concrete function has finished specializing, finish processing the
  /// function and call the verifier.
  void finalizeAndVerifyFunction(ImplNode *node);

  /// Process a kgen.param.fork op. This will create a clone for each value of
  /// the parameter search, and will mark the parent as an error. This results
  /// in a very clean model where the parent of the current parent (a generator)
  /// will have its children be the successfully concretized parameter search
  /// nodes.
  LogicalResult processParamForkOp(ImplNode *parent, ParamForkOp op);

  /// Spawn a clone for kgen.param.fork. This creates a new FuncOp that is a
  /// sibling to the parent of the kgen.param.fork op. It replaces the
  /// kgen.param.fork with a param.declare to allow specialization to succeed.
  void spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                           ImplNode *forkParentNode);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  ElaborationState processCallOp(ImplNode *parent, KGENCallOpInterface call);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  ElaborationState processGeneratorUser(KGENCallOpInterface user,
                                        SymbolConstantAttr calleeSymbol,
                                        ImplNode *parent);

  /// Complete processing of a generator user by resolving any bound result
  /// types or parameters in the parent scope. This is the step that propagates
  /// result parameters from the inner scope to the outer scope.
  LogicalResult completeCallProcessing(KGENCallOpInterface user,
                                       ArrayRef<ParamDeclAttr> decls,
                                       ImplNode *thisNode, ImplNode *node,
                                       Logger &logger);

  /// Resolve call input parameters - this is a complex function because calls
  /// can have regions. We take the body of those regions and put it into a
  /// generator with a specially prepared ParameterEvaluator scope and elaborate
  /// the region that way.
  ///
  /// Elaborating region parameters is the most non-local part of the elaborator
  /// - we have to interact with the module symbol table to put these regions
  /// into top-level generators.
  ErrorTreeOr<ArrayAttr>
  resolveCallInputParams(Operation *call, IREvaluator &evaluator,
                         ArrayRef<TypedAttr> inputValues);

  /// Process a param.declare.region op by creating a generator with the correct
  /// captures. We don't specialize the generator until the call-site because we
  /// don't know what the actual input parameters are supposed to be until then.
  LogicalResult processParamDeclareRegionOp(ImplNode *parent,
                                            ParamDeclareRegionOp regionDecl);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  ElaborationState processParamIfOp(ImplNode *parent, ParamIfOp op);

  /// Schedule an implementation node on the LLCL work queue and increment the
  /// initial counters.
  void initialScheduleImplNode(ImplNode *inode) {
    ++inode->parent->numActive;
    scheduleImplNode(inode);
  }
  /// Signal the worklist to tell it a job has completed or has been taken off
  /// the workqueue.
  void signalWorklist() {
    if (--g.numWorkItems == 0)
      g.worklistCh.copy().emplace();
  }
  /// Schedule an implementation node on the LLCL work queue.
  void scheduleImplNode(ImplNode *inode);
  /// Process the scopes within an implementation node.
  LogicalResult processImplNode(ImplNode *inode);
  /// Process a worklist of ops. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// scope should be pre-empted with a new scope, returns `skipNode` if
  /// processing the current implementation node should suspended.
  ElaborationState processScope(ImplNode *node, ImplNode::WorkItem &item);
  /// Process a single operation. Returns error if processing the scope resulted
  /// in an error, returns `skipFrame` if the processing of the current scope
  /// should be pre-empted with a new scope, returns `skipNode` if processing
  /// the current implementation node should suspended.
  ElaborationState processOp(ImplNode *node, Operation *op);

  /// Request specialization of the generator at `genNode`. If the node is ready
  /// complete, then the function returns `advance` and the concrete functions
  /// can be retrieved from the node. Otherwise, the function returns
  /// `skipNode`, indicating that elaboration of the current function should be
  /// suspended. It returns `error` if the generator constraints were not
  /// satisfied.
  ElaborationState specializeGenerator(ParamNode *genNode, ParamNode *from);
  /// Request specialization of the generator at `genNode` and block until the
  /// generator is ready or specialization resulted in an error.
  ErrorTreeOrSuccess specializeGeneratorAndWait(ParamNode *genNode);

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ModuleOp theModule,
                    ArrayRef<GeneratorOp> primaryGenerators);

private:
  /// A logger used to emit information during the elaboration process.
  Logger logger;

  /// Hash table to speed up lookups of generators in the expansion tree.
  /// Hash table of known ParameterUseDefGraphs. This ensures we only compute a
  /// graph once for each generator. This is extra state generated by
  /// specializeGenerator that is *required for correctness* - this will cause
  /// issues with caching (though it would be easy to simply recompute) unless
  /// we create a ParametricNode or something we can use to store these in a
  /// proper data structure.
  Shared<DenseMap<GeneratorOp, std::unique_ptr<ParameterUseDefGraph>>>
      knownGraphs;

  /// The elaborator config.
  ElaboratorConfig config;

  /// The callgraph being expanded.
  ExpansionGraph g;

  /// This is the cached parameter collector analysis.
  ThreadLocalCache<ParameterCollector::Analysis> paramCache;

  /// The functor used for evaluating generator specializations.
  EvaluatorExecutorFnRef evaluatorExecutorFn;

  /// The LLCL runtime instance to use.
  LLCL::Runtime &runtime;

  /// Remove parameter declare regions after generator elaboration.
  DenseMap<StringAttr, ParameterUseDefGraph *> knownRegions;
  SmallVector<OwningOpRef<ParamDeclareRegionOp>> paramDeclareRegionOps;
};
} // namespace

//===----------------------------------------------------------------------===//
// finalizeAndVerifyFunction
//===----------------------------------------------------------------------===//

void ElaboratorImpl::finalizeAndVerifyFunction(ImplNode *node) {
  TimeTraceScope<> traceScope("finalizeAndVerifyFunction");
  // Erase any unreachable blocks that might have arisen.
  FuncOp func = node->func;
  mlir::IRRewriter b(func.getContext());
  (void)mlir::eraseUnreachableBlocks(b, func.getBodyRegion());

  // Check that the thing we just built is correct IR!  We want to catch any
  // errors produced by the verify pass, we don't want them to actually get
  // emitted.
  std::string verificationErrorStr;
  llvm::raw_string_ostream verificationError(verificationErrorStr);
  std::optional<Location> verificationLoc;
  mlir::ScopedDiagnosticHandler diagHandler(
      func.getContext(), [&](Diagnostic &diag) -> LogicalResult {
        // Combine multiple verification errors.
        if (verificationLoc) {
          verificationError << "; " << diag.str();
          verificationLoc =
              FusedLoc::get(verificationLoc->getContext(),
                            {*verificationLoc, diag.getLocation()});
        } else {
          verificationError << diag.str();
          verificationLoc = diag.getLocation();
        }
        return success();
      });
  // Verify the function.
  if (failed(verify(func))) {
    node->setToError(ErrorTree(*verificationLoc, Twine("verification error: ") +
                                                     verificationError.str()));
    LLVM_DEBUG(logger.scope("Result: Failure")
               << verificationError.str() << "\n");
  }
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getConcreteFunction
//===----------------------------------------------------------------------===//

ErrorTreeOr<FuncOp>
ElaboratorImpl::getConcreteFunction(Location loc, FlatSymbolRefAttr symbolRef,
                                    ArrayRef<TypedAttr> paramValues) {
  auto gen =
      symtab.read([name = symbolRef.getAttr()](const SymbolTable &symtab) {
        return symtab.lookup<GeneratorOp>(name);
      });

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(runtime, vals, gen, /*depth=*/0);
  // If the node has already been elaborated, just use that result.
  if (node->status != ParamNode::DONE) {
    if (ErrorTreeOrSuccess err = specializeGeneratorAndWait(node);
        err.isError())
      return err.takeError();
  }

  return node->getFirstConcreteFunc();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getAllConcreteFunctions
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ElaboratorImpl::getAllConcreteFunctions(
    Location loc, FlatSymbolRefAttr symbolRef, ArrayRef<TypedAttr> paramValues,
    std::vector<FuncOp> &funcs) {
  auto gen =
      symtab.read([name = symbolRef.getAttr()](const SymbolTable &symtab) {
        return symtab.lookup<GeneratorOp>(name);
      });

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(runtime, vals, gen, /*depth=*/0);
  if (node->status != ParamNode::DONE) {
    if (ErrorTreeOrSuccess err = specializeGeneratorAndWait(node);
        err.isError())
      return err.takeError();
  }
  node->getAllConcreteFuncs(funcs);
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::evaluateFunctions
//===----------------------------------------------------------------------===//

ErrorOr<size_t> ElaboratorImpl::evaluateFunctions(FuncOp evaluator,
                                                  ArrayRef<FuncOp> options) {
  // This implements a turnstile for the evaluator. Decrement once for the
  // current worker.
  signalWorklist();
  ErrorOr<size_t> result = 0;
  {
    // Let in only one thread at a time.
    llvm::sys::SmartScopedWriter<true> guard(g.evaluatorMutex);
    // Decrement once again to allow the chain to complete.
    signalWorklist();
    // Starve the workqueue. This also waits for any threads on their way into
    // this function to hit the turnstile.
    // FIXME: This should acquire a semaphore shared across all compiler
    // processes to ensure search is performed in isolation.
    LLCL::await(g.worklistCh);
    // Run evaluator.
    result = symtab.read([&](const SymbolTable &symtab) {
      return evaluatorExecutorFn(evaluator, symtab, getTarget(), options);
    });
    // Re-initialize the worklist chain and counter, plus 1 for this current
    // thread. This is safe to do because there are no other active work items.
    assert(g.numWorkItems == 0 && "work count underflow");
    g.numWorkItems = 2;
    g.worklistCh = LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamForkOp
//===----------------------------------------------------------------------===//

/// Process a kgen.param.fork op.
LogicalResult ElaboratorImpl::processParamForkOp(ImplNode *parent,
                                                 ParamForkOp op) {
  auto _ = logger.scope("Processing ParamForkOp");
  LLVM_DEBUG(logger.scope("Options") << op.getValuesAttr() << "\n");

  IREvaluator &evaluator = parent->getEvaluator();
  ErrorTreeOr<Attribute> errorOrValue =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getValuesAttr());
  if (errorOrValue.isError()) {
    parent->setToError(errorOrValue.takeError());
    return failure();
  }

  auto forkValuesAttr = cast<VariadicAttr>(errorOrValue.takeValue());

  if (forkValuesAttr.getValues().empty()) {
    parent->setToError(ErrorTree(op.getLoc(), "no candidates found"));
    return failure();
  }

  // Loop over all the possible candidates that we will search over, spawning
  // N possibilities to explore.
  SmallVector<ErrorTree> errors;
  DenseSet<Attribute> seenValues;
  for (Attribute candidate : forkValuesAttr.getValues().drop_front()) {
    // Simplify the input expressions.
    ErrorTreeOr<Attribute> errorOrValue =
        evaluator.concretizeParameterExpr(op.getLoc(), candidate);
    if (errorOrValue.isError()) {
      errors.push_back(errorOrValue.takeError());
      continue;
    }

    Attribute value = errorOrValue.takeValue();

    // If we've already seen this concrete value before,
    // ignore the duplicate.
    if (!seenValues.insert(value).second)
      continue;

    // Otherwise, spawn a clone for this value.
    spawnParamForkClone(op, value, parent);
  }

  // Take the first value for the current function.
  parent->getEvaluator().setOrOverwriteParameterValue(
      op.getParamDecl(), forkValuesAttr.getValues().front());
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::spawnParamForkClone
//===----------------------------------------------------------------------===//

/// Spawn a clone from a kgen.param.fork op.
void ElaboratorImpl::spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                                         ImplNode *forkParentNode) {
  auto _ = logger.scope("Spawning ParamForkClone for '", value, "'");

  // Start by cloning the current WIP func to a new copy of it.
  IRMapping map;

  // Hook this new clone up correctly.
  ImplNode *newFuncNode = g.fork(forkParentNode, map, symtab,
                                 forkOp.getParamDecl().getName(), value);

  // Change the future of this func by resolving the forkOp in the new func
  // to the specified value.
  auto newFork = cast<ParamForkOp>(map.lookup(forkOp.getOperation()));

  LLVM_DEBUG(logger << "Setting '" << newFork.getParamDecl() << "' = '" << value
                    << "'\n");

  // Update the evaluator.
  newFuncNode->getEvaluator().setOrOverwriteParameterValue(
      newFork.getParamDecl(), value);

  // Immediately process the fork operation in the clone by erasing it here.
  // Take it off the clone's worklist before doing so.
  assert(newFuncNode->stack.back().ops.back() == newFork);
  newFuncNode->stack.back().ops.pop_back();
  newFork->erase();

  // And finally, push the forked node onto its parent's worklist, so that it
  // will get processed after this function returns.
  initialScheduleImplNode(newFuncNode);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processGeneratorUser
//===----------------------------------------------------------------------===//

ElaborationState
ElaboratorImpl::processGeneratorUser(KGENCallOpInterface user,
                                     SymbolConstantAttr calleeSymbol,
                                     ImplNode *parent) {
  auto _ = logger.scope("Processing Generator User");
  LLVM_DEBUG(logger.logOp("User", user));

  // Add in the mapping for parameters in the calls.
  auto resolvedCallParamsOr = resolveCallInputParams(
      user, parent->getEvaluator(), calleeSymbol.getParamValues());
  if (resolvedCallParamsOr.isError()) {
    parent->setToError(resolvedCallParamsOr.takeError());
    return failure();
  }
  ArrayAttr inputParamKey = *resolvedCallParamsOr;

  // Lookup the callee.
  auto calleeOp = symtab.read(
      [name = cast<FlatSymbolRefAttr>(calleeSymbol.getSymbol()).getAttr()](
          const SymbolTable &symtab) { return symtab.lookup(name); });
  if (!calleeOp) {
    parent->setToError(
        ErrorTree(user.getLoc(), "could not find callee '" +
                                     mlir::debugString(calleeSymbol) +
                                     "' (compiler bug, please report!)"));
    return failure();
  }

  ArrayRef<ParamDeclAttr> decls = user.getParamDecls();
  if (auto func = dyn_cast<FuncOp>(calleeOp)) {
    ImplNode *node =
        g.concreteNodes.read([func](const DenseMap<FuncOp, ImplNode *> &map) {
          return map.lookup(func);
        });
    if (!node) {
      parent->setToError(ErrorTree(user.getLoc(),
                                   "concrete callee doesn't have a node "
                                   "(compiler bug, please report!)"));
      return failure();
    }
    return completeCallProcessing(user, decls, node, parent, logger);
  }

  LLVM_DEBUG({
    logger.logOp("Callee", calleeOp);
    logger << "Input Params: " << inputParamKey << "\n";
  });

  // If we already have a binding for this, we're done.
  auto gen = cast<GeneratorOp>(calleeOp);
  auto found = parent->bindings.find({inputParamKey, gen});
  if (found != parent->bindings.end()) {
    LLVM_DEBUG(
        found->getSecond()->print(logger.scope("Result: Existing Binding")));
    return completeCallProcessing(user, decls, found->getSecond(), parent,
                                  logger);
  }

  // Check for excessive instantiation depth.
  if (parent->parent->depth > config.maxDepth) {
    parent->setToError(ErrorTree(parent->parent->gen.getLoc(),
                                 "elaborator expansion is " +
                                     Twine(config.maxDepth + 1) +
                                     " levels deep - infinite recursion?"));
    return ElaborationState::error();
  }

  // Find the tree node that corresponds to the thing we're calling.
  ParamNode *calleeNode =
      g.getOrCreate(runtime, inputParamKey, gen, parent->parent->depth + 1);
  ElaborationState result = specializeGenerator(calleeNode, parent->parent);
  if (result.isError()) {
    assert(calleeNode->constraintError && "expected a constraint failure");
    parent->setToError(calleeNode->constraintError->copy());
    return ElaborationState::error();
  }
  if (result.shouldSkipNode()) {
    calleeNode->paramCh.andThenAsync(
        [parent, this] { scheduleImplNode(parent); });
    return ElaborationState::skipNode();
  }

  // Complete processing for all the leaves of this subtree.
  std::vector<ImplNode *> concrete;
  calleeNode->getAllConcreteNodes(concrete);

  // If the concrete thing has bindings, they must be consistent with the
  // parent's bindings for us to consider it. Remove nodes from the vector that
  // have bindings that are inconsistent with the parent.
  auto newEnd = llvm::remove_if(concrete, [&](ImplNode *node) {
    bool hasConsistentBindings = llvm::all_of(node->bindings, [&](auto pair) {
      // The binding is only inconsistent if it (a) does exist and (b) is
      // different.
      auto found = parent->bindings.find(pair.first);
      if (found != parent->bindings.end())
        return found->second == pair.second;
      // Otherwise, we're good.
      return true;
    });
    // If it has inconsistent bindings
    if (!hasConsistentBindings && !node->bindings.empty() &&
        !parent->bindings.empty()) {
      LLVM_DEBUG(logger << "Removing node for inconsistent bindings: ";
                 node->print(logger));
      return true;
    }
    return false;
  });
  concrete.erase(newEnd, concrete.end());

  // If there are no implementations, return the callee's errors.
  if (concrete.empty()) {
    ErrorTree out(user.getLoc(),
                  "call expansion failed - no concrete specializations");
    ErrorTree err(calleeNode->gen.getLoc(), "no viable expansions found");
    for (ImplNode &impl : llvm::make_pointee_range(calleeNode->impls))
      if (impl.error)
        err.addCause(impl.error->copy());
    out.addCause(std::move(err));
    parent->setToError(std::move(out));
    return failure();
  }

  LLVM_DEBUG({
    auto _ = logger.scope("Concrete Implementations, n=", concrete.size());
    for (auto &impl : concrete)
      impl->print(logger);
  });

  // There are more concrete things, we have to multi-version the parent!
  for (auto *c : llvm::drop_begin(concrete)) {
    // Clone the parent.
    IRMapping map;

    auto _ = logger.scope("New Multi-Versioning Op");
    logger.logOp("Op", c->func);
    LLVM_DEBUG(c->print(logger << "Concrete Implementation "));

    // This is a sibling to the parent, and it clones the parent's evaluator.
    ImplNode *newNode = g.fork(parent, map, symtab, "", c->func.getNameAttr());
    // Bind this concrete impl to this callee for this node.
    newNode->bindings[{inputParamKey, gen}] = c;

    // The call operation in the cloned function wil be handled by
    // `completeCallProcessing` below, so take it off the clone's worklist
    // beforehand, since new ops may be added.
    assert(map.lookup(&*user) == newNode->stack.back().ops.back());
    newNode->stack.back().ops.pop_back();

    if (failed(completeCallProcessing(
            cast<KGENCallOpInterface>(map.lookup(user.getOperation())), decls,
            c, newNode, logger))) {
      // If call processing completion failed, then don't enqueue this node.
      assert(newNode->error && "expected an error on new node");
      continue;
    }

    LLVM_DEBUG(newNode->print(logger << "New Op "));

    // Process the rest of the worklist in this new scope. If the scope
    // processing failed, do nothing.
    initialScheduleImplNode(newNode);
  }

  // Bind this concrete impl to this callee for this node.
  parent->bindings[{inputParamKey, gen}] = concrete.front();

  // Call completeGeneratorUserProcessing on the first concrete thing. This will
  // flow nested bindings upward correctly.
  return completeCallProcessing(user, decls, concrete.front(), parent, logger);
}

//===----------------------------------------------------------------------===//
// completeCallProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
static LogicalResult processParamApplyOp(ImplNode *inode, ParamApplyOp op,
                                         FuncOp func) {

  SmallVector<TypedAttr> operands;
  for (TypedAttr operand : op.getOperands()) {
    ErrorTreeOr<Attribute> value =
        inode->getEvaluator().concretizeParameterExpr(op.getLoc(), operand);
    if (value.isError()) {
      inode->setToError(value.takeError());
      return failure();
    }
    operands.push_back(cast<TypedAttr>(value.takeValue()));
  }
  ErrorTreeOr<TypedAttr> result =
      inode->getEvaluator().evaluateFunction(func, operands);
  if (result.isError()) {
    inode->setToError(result.takeError());
    return failure();
  }

  // Bind the result and erase the operation.
  inode->getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(),
                                                     result.takeValue());
  op.erase();
  return success();
}

LogicalResult ElaboratorImpl::completeCallProcessing(
    KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls, ImplNode *thisNode,
    ImplNode *node, Logger &logger) {
  IREvaluator &evaluator = node->getEvaluator();

  // Add the callee's bindings to the parent of the call. This ensures that we
  // don't re-bind something we've already bound.
  for (const auto &[k, v] : thisNode->bindings) {
    auto &oldV = node->bindings[k];
    assert(!oldV || oldV == v);
    oldV = v;
  }

  if (thisNode->error) {
    node->setToError(ErrorTree(user.getLoc(), "call expansion failed",
                               thisNode->error->copy()));
    return failure();
  }

  FuncOp newCalleeFunc = thisNode->func;

  // If this is a `kgen.param.apply`, bind its result here.
  if (auto apply = dyn_cast<ParamApplyOp>(*user))
    return processParamApplyOp(node, apply, newCalleeFunc);

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (auto result : user->getResultTypes()) {
    ErrorTreeOr<Type> typeOr =
        evaluator.concretizeParameterExpr(user.getLoc(), result);
    if (typeOr.isError()) {
      node->setToError(typeOr.takeError());
      return failure();
    }

    resultTypes.push_back(typeOr.takeValue());
  }

  // Save the user's location before we delete the op.
  Location userLoc = user.getLoc();

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  mlir::IRRewriter b{OpBuilder(user)};
  auto newCallee = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
      newCalleeFunc.getSignature());
  user.concretizeCallee(b, newCallee, resultTypes);

  if (decls.empty())
    return success();

  // If we don't have the result parameters yet, then either no result
  // parameters are necessary, or we have another problem entirely wherein we
  // could not complete the callee's result parameter resolution at all - likely
  // meaning we're in an infinite recursive loop. Essentially, we came back to
  // the same combination of generator + input parameters without resolving the
  // result parameters yet.
  ArrayAttr resultParams = thisNode->resultParams;
  if (!resultParams && !decls.empty()) {
    node->setToError(ErrorTree(userLoc,
                               "could not resolve callee's necessary result "
                               "parameters, infinite recursive loop?"));
    return failure();
  }

  // Bind the result parameters to the output parameter decls.
  assert(decls.size() == resultParams.size());
  for (auto [decl, bindValue] : llvm::zip(decls, resultParams)) {
    LLVM_DEBUG(logger << "Binding " << decl << " to " << bindValue << "\n");
    evaluator.setOrOverwriteParameterValue(decl, bindValue);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::resolveCallInputParams
//===----------------------------------------------------------------------===//

/// Resolve input params on a call_param op.
ErrorTreeOr<ArrayAttr>
ElaboratorImpl::resolveCallInputParams(Operation *call, IREvaluator &evaluator,
                                       ArrayRef<TypedAttr> inputValues) {
  LLVM_DEBUG(logger.logOp("Resolving Call Input Param", call);
             logger << " with input bindings: ";
             llvm::interleaveComma(inputValues, logger); logger << "\n");

  SmallVector<Attribute> boundInputParams;
  for (TypedAttr param : inputValues) {
    // Fold the parameter expression in this context to a simple constant.
    ErrorTreeOr<Attribute> valueOr =
        evaluator.concretizeParameterExpr(call->getLoc(), param);
    if (valueOr.isError())
      return valueOr.takeError();

    Attribute value = valueOr.takeValue();
    boundInputParams.push_back(value);
  }

  return ArrayAttr::get(call->getContext(), boundInputParams);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processCallOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
ElaborationState ElaboratorImpl::processCallOp(ImplNode *parent,
                                               KGENCallOpInterface call) {
  ErrorTreeOr<Attribute> symbol =
      parent->getEvaluator().concretizeParameterExpr(call.getLoc(),
                                                     call.getCallee());
  if (symbol.isError()) {
    parent->setToError(symbol.takeError());
    return ElaborationState::error();
  }

  if (auto sym = dyn_cast<SymbolConstantAttr>(*symbol))
    return processGeneratorUser(call, sym, parent);

  auto decl = dyn_cast<RegionAttr>(*symbol);
  if (LLVM_UNLIKELY(!decl)) {
    parent->setToError(ErrorTree(call.getLoc(),
                                 "concrete parameter must be a symbol or a "
                                 "region (compiler bug, please report!)"));
    return ElaborationState::error();
  }

  // OK we found a region, put it into the machinery.
  ParameterUseDefGraph *regionGraph = knownRegions.lookup(decl.getRegionName());
  Region *region = regionGraph->scope;
  LLVM_DEBUG(logger.logOp("Inlining call to parameter region:",
                          region->getParentOp()));

  // Inline the call now. We clone them now so that we don't modify the original
  // region in case it's re-used.
  IRMapping map;
  assert(parent->stack.back().ops.back() == call);
  inlineCallToConcreteRegion(call, region, map, AlwaysInlineLevel::Enabled);
  // Delete the call operation now, so when the new frame consisting of the
  // inlined operations has been processed, the elaborator can immediately
  // continue on to the next operation.
  parent->stack.back().ops.pop_back();

  // Collect all the ops to process *in the region*.
  std::vector<Operation *> opsToRewriteInRegion;
  llvm::append_range(opsToRewriteInRegion,
                     llvm::reverse(regionGraph->paramOps));
  collectOpsToProcess(region, *regionGraph, opsToRewriteInRegion);
  std::vector<Operation *> opsToRewrite;
  for (Operation *op : opsToRewriteInRegion)
    opsToRewrite.push_back(map.lookup(op));

  // Push a new work item.
  ImplNode::WorkItem item{std::move(opsToRewrite), parent->getEvaluator(),
                          [](ImplNode *) { return success(); }};

  // Set any parameter bindings on the region in the evaluator context.
  item.evaluator.clearCache();
  for (auto [decl, value] : llvm::zip(
           cast<ParamDeclareRegionOp>(region->getParentOp()).getInputParams(),
           decl.getParamValues()))
    item.evaluator.setOrOverwriteParameterValue(decl, value);

  parent->stack.push_back(std::move(item));
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamDeclareRegionOp
//===----------------------------------------------------------------------===//

/// Process a param.declare.region by creating a generator for the contained
/// region.
LogicalResult
ElaboratorImpl::processParamDeclareRegionOp(ImplNode *parent,
                                            ParamDeclareRegionOp regionDecl) {
  StringAttr regionName = getMangledRegionParamName(regionDecl);
  // Set the region's parameter decl as the value for this name. That will
  // signal to the call_param handler that it needs to inline the region it
  // finds for that decl. The region attr itself holds onto a bit that knows if
  // the region is isolated from above (in SSA-land) or not.
  parent->getEvaluator().setOrOverwriteParameterValue(
      regionDecl.getParamDecl().getName(),
      RegionAttr::get(
          regionName, {},
          BoolAttr::get(regionDecl.getContext(),
                        operationIsIsolatedFromAbove(regionDecl)),
          cast<SignatureType>(regionDecl.getParamDecl().getType())));

  // Save the known region parameter use-def graph.
  auto found =
      parent->paramGraph.nestedScopes.find(&regionDecl.getBodyRegion());
  assert(found != parent->paramGraph.nestedScopes.end() &&
         "must have a nested region");
  LLVM_DEBUG(logger << "Storing known region: " << regionName << "\n");
  knownRegions[regionName] = &found->getSecond();

  // Transfer ownership to the elaborator.
  regionDecl->remove();
  paramDeclareRegionOps.push_back(
      OwningOpRef<ParamDeclareRegionOp>(regionDecl));

  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamIfOp
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::processParamIfOp(ImplNode *parent,
                                                  ParamIfOp op) {
  // Check the condition expression.
  auto errorOrValue =
      parent->getEvaluator().concretizeParameterExpr(op.getLoc(), op.getCond());
  if (errorOrValue.isError()) {
    parent->setToError(errorOrValue.takeError());
    return ElaborationState::error();
  }

  // Take whichever branch the condition indicated, and simply inline those ops
  // then elaborate them. We can do this by splicing the op list into the parent
  // block. We splice it this way to avoid remapping the ops when we process
  // them later.
  bool resultBool = cast<BoolAttr>(errorOrValue.takeValue()).getValue();
  // Get the appropriate region.
  Region &toProcess = op->getRegion(!resultBool);

  auto foundNestedScope = parent->paramGraph.nestedScopes.find(&toProcess);
  if (foundNestedScope == parent->paramGraph.nestedScopes.end()) {
    parent->setToError(ErrorTree(
        op.getLoc(),
        "expected a nested parameter scope (compiler bug, please report!)"));
    return ElaborationState::error();
  }

  ParameterUseDefGraph &uses = foundNestedScope->getSecond();

  LLVM_DEBUG(logger << "Elaborating block:\n"; toProcess.front().print(logger));

  // Only process the ops in the branch that we ended up taking.
  std::vector<Operation *> opsToRewrite;
  for (Operation *paramOp : llvm::reverse(uses.paramOps)) {
    // Check if this op is in a region that is a child of the region we care
    // about. If not, don't process it.
    if (!toProcess.isAncestor(paramOp->getParentRegion()))
      continue;

    opsToRewrite.push_back(paramOp);
  }
  collectOpsToProcess(&toProcess, uses, opsToRewrite);

  // Push a new node and skip over the current frame until it completes.
  ImplNode::WorkItem item{std::move(opsToRewrite), parent->getEvaluator(),
                          nullptr};
  item.evaluator.clearCache();

  // When the nested scope completes processing, finish processing the current
  // parameter if.
  item.onComplete = [this, resultBool](ImplNode *node) -> LogicalResult {
    assert(node->stack.size() >= 2 && "expected at least two work items");
    // Retrieve the current state.
    ImplNode::WorkItem &curFrame = node->stack.back();
    ImplNode::WorkItem &parentFrame = *std::next(node->stack.rbegin());
    auto op = cast<ParamIfOp>(parentFrame.ops.back());
    LLVM_DEBUG(logger << "Parameter if completion callback: " << op);

    // Bind the result parameters from the nested scope.
    for (ParamDeclAttr decl : op.getResultParams()) {
      parentFrame.evaluator.setOrOverwriteParameterValue(
          decl, curFrame.evaluator.getParameterValues().at(decl.getName()));
    }

    // Splice the ops into the parent. Grab the terminator before the iterators
    // invalidate.
    Block::iterator iter = op->getIterator();
    Block &block = op->getRegion(!resultBool).front();
    Operation *terminator = block.getTerminator();
    op->getBlock()->getOperations().splice(iter, block.getOperations());

    // Update the values for the result parameters and do other processing
    // necessary for param.yield.
    if (auto yieldOp = dyn_cast<ParamYieldOp>(terminator)) {
      // RAUW the op's results with the terminator's inputs.
      op->getResults().replaceAllUsesWith(yieldOp.getOperands());

      // Erase the terminator.
      terminator->erase();
    } else if (auto hlcfTerm =
                   dyn_cast<HLCF::ControlFlowTerminator>(terminator)) {
      // If it's an kgen.return op, we have to split the block after the return.
      hlcfTerm->getBlock()->splitBlock(++hlcfTerm->getIterator());
      // Drop all uses of the if op because any of its uses will be null and
      // void at this point.
      op->dropAllDefinedValueUses();
    } else {
      node->setToError(ErrorTree(terminator->getLoc(),
                                 "unknown terminator kind for parameter if "
                                 "(compiler bug, please report!)"));
      return failure();
    }

    // We always erase this op and its nested scopes from the parameter graph -
    // it's been handled, and we don't want anyone else touching it later
    // considering we're about to delete the op itself.
    ParameterUseDefGraph &paramGraph = node->paramGraph;
    auto eraseIfScopes = [op](ParameterUseDefGraph &graph) mutable {
      // Erase any regions from the nested scopes that belong either to this op
      // or under this op.
      for (auto &[r, _] : graph.nestedScopes)
        if (op->isAncestor(r->getParentOp()))
          graph.nestedScopes.erase(r);

      // Do the same for nested decls. These two are somehow not always in sync,
      // so we have to check both separately.
      auto newEnd = llvm::remove_if(graph.nestedDecls, [&](Region *r) {
        return op->isAncestor(r->getParentOp());
      });
      graph.nestedDecls.erase(newEnd, graph.nestedDecls.end());
    };
    // Delete references to this nested declaration from all nested graphs.
    eraseIfScopes(paramGraph);
    for (auto &[scope, graph] : paramGraph.nestedScopes)
      eraseIfScopes(graph);

    // The callback to the current frame finishes processing the current
    // operation, so take it off the parent frame's worklist.
    op->erase();
    parentFrame.ops.pop_back();
    LLVM_DEBUG(
        logger.logOp("param.if parent scope (after processing)", node->func));
    return success();
  };

  parent->stack.push_back(std::move(item));
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processScope
//===----------------------------------------------------------------------===//

void ElaboratorImpl::scheduleImplNode(ImplNode *inode) {
  // Increment the number of scheduled work items.
  ++g.numWorkItems;
  runtime.getWorkQueue()->addTask([inode, this] {
    // Process the node. If processing the node got pre-empted, then return. It
    // will get scheduled again later.
    if (succeeded(processImplNode(inode))) {
      // If this is the last implementation node for its parent parameter node
      // to complete, then the parameter node is done.
      ParamNode *p = inode->parent;
      if (--p->numActive == 0) {
        p->status = ParamNode::DONE;
        p->paramCh.copy().emplace();
      }
    }
    // Signal the worklist that the work is complete.
    signalWorklist();
  });
}

LogicalResult ElaboratorImpl::processImplNode(ImplNode *inode) {
  LLVM_DEBUG(inode->print(logger << "Processing implementation node: ",
                          /*printBindings=*/false));
  assert(!inode->stack.empty() && "expected at least one work item");

  TimeTraceScope<EnableTracing> traceScope(
      "processParamNode", [inode] { return inode->func.getSymName().str(); });

  while (!inode->stack.empty()) {
    ImplNode::WorkItem &item = inode->stack.back();
    [[maybe_unused]] size_t size = inode->stack.size();
    ElaborationState result = processScope(inode, item);
    if (result.isError()) {
      // Interrupt indicates a fatal error.
      assert(inode->error && "node processing interrupted but no error set");
      return success();
    }
    if (result.shouldSkipFrame()) {
      // Skip indicates we need to move to another frame first.
      assert(inode->stack.size() == size + 1 && "skip with no new frame");
      continue;
    }
    if (result.shouldSkipNode()) {
      // Node skip indicates to suspend elaboration of the current function and
      // come back later.
      return failure();
    }
    // Advance indicates the current work item's operation list was exhausted.
    assert(inode->stack.size() == size && "new frame with no skip");
    assert(item.ops.empty() && "advance did not exhaust worklist");
    if (failed(item.onComplete(inode))) {
      assert(inode->error && "callback failed but no error set");
      return success();
    }
    inode->stack.pop_back();
  }
  assert(!inode->error && "unexpected error");
  finalizeAndVerifyFunction(inode);
  return success();
}

ElaborationState ElaboratorImpl::processScope(ImplNode *node,
                                              ImplNode::WorkItem &item) {
  LLVM_DEBUG({
    auto _ = logger.scope("Operations to Rewrite");
    for (Operation *op : item.ops)
      logger << *op << "\n";
  });
  TimeTraceScope<EnableTracing> traceScope(
      "processScope", std::to_string(item.ops.size()) + " ops");

  // Processing an op may generate more stuff, or even delete the op being
  // processed.
  while (!item.ops.empty()) {
    Operation *op = item.ops.back();
    ElaborationState result = processOp(node, op);
    if (result.isError() || result.shouldSkipFrame() || result.shouldSkipNode())
      return result;
    item.ops.pop_back();
  }
  LLVM_DEBUG(node->print(logger << "Completed processing "));
  return ElaborationState::advance();
}

ElaborationState ElaboratorImpl::processOp(ImplNode *node, Operation *op) {
  if (!op->getBlock()->isEntryBlock() && op->getBlock()->hasNoPredecessors())
    return ElaborationState::advance();

  auto _ = logger.scope("Processing: '", op->getName(), "'");
  logger.logOp("Op", op);

  if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamDeclareOp");
    return processParamDeclareOp(node, declare);
  } else if (auto region = dyn_cast<ParamDeclareRegionOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamDeclareRegionOp");
    return processParamDeclareRegionOp(node, region);
  } else if (auto bind = dyn_cast<ParamResultBindOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamResultBindOp");
    return processParamResultBindOp(node, bind);
  } else if (auto fork = dyn_cast<ParamForkOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamForkOp");
    return processParamForkOp(node, fork);
  } else if (auto rebindOp = dyn_cast<RebindOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processRebindOp");
    return processRebindOp(node, rebindOp);
  } else if (auto assertOp = dyn_cast<ParamAssertOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamAssertOp");
    return processParamAssertOp(node, assertOp);
  } else if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processParamIfOp");
    return processParamIfOp(node, ifOp);
  } else if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
    TimeTraceScope<EnableTracing> traceScope("processCallOp");
    return processCallOp(node, call);
  } else {
    TimeTraceScope<EnableTracing> traceScope("processGenericOp");
    return processGenericOp(node, op);
  }
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeGenerator
//===----------------------------------------------------------------------===//

/// Try extracting a short name from a mangled name.
/// E.g. for the mangled name "$Math::log($SIMD::SIMD[type, simd_width])"
/// we want to extract "log".
/// This is the part before the opening brace and after the last ':' before it.
static StringRef tryGettingShortName(StringRef s) {
  return s.split('(').first.rsplit(':').second;
}

ErrorTreeOrSuccess
ElaboratorImpl::specializeGeneratorAndWait(ParamNode *genNode) {
  ElaborationState result = specializeGenerator(genNode, /*from=*/nullptr);
  if (result.isError()) {
    assert(genNode->constraintError && "expected a constraint error");
    return genNode->constraintError->copy();
  }
  if (result.shouldSkipNode()) {
    LLCL::await(genNode->paramCh);
    assert(genNode->status == ParamNode::DONE);
  }
  ErrorTree err(genNode->gen.getLoc(), "no viable expansions found");
  for (ImplNode &impl : llvm::make_pointee_range(genNode->impls)) {
    if (!impl.error)
      return success();
    err.addCause(impl.error->copy());
  }
  return std::move(err);
}

ElaborationState ElaboratorImpl::specializeGenerator(ParamNode *genNode,
                                                     ParamNode *from) {
  ParamNode::Status existing = ParamNode::FRESH;
  if (!genNode->status.compare_exchange_strong(existing,
                                               ParamNode::IN_PROGRESS)) {
    if (existing == ParamNode::DONE) {
      // If this generator instantiation is already known to always be invalid,
      // indicate an error.
      if (genNode->constraintError)
        return ElaborationState::error();
      // If this generator node is already concrete and has no error, don't
      // re-concretize.
      if (config.testDiagnostics)
        genNode->gen.emitRemark("Generator has already been specialized");
      return ElaborationState::advance();
    }
    // If the worker hit a parameter node that is already in progress, this
    // could mean two things:
    //
    // 1. The parameter node is being handled by another worker.
    // 2. A generator recursively calls into the same instantiation of itself.
    //
    // The first case is impossible in single-threaded, DFS traversal of the
    // expansion graph, because the elaborator will process generator
    // instantiations as soon as they are encountered.
    //
    // In that situation, the elaborator assumes the recursive generator
    // instantiation will have at most one successful candidate. This is valid
    // because:
    //
    // 1. If there is more than one, the total number of candidates is infinity
    //    due to recursion.
    // 2. If there are zero successful candidates, then elaboration of the rest
    //    of the function will fail anyways, and the error will be propagated
    //    up.
    //
    // However, the elaborator does not know will candidate will succeed, so it
    // must defer the processing of the recursive call to the end of the
    // worklist. The elaborator also places the restriction that recursive calls
    // cannot have result parameters. Although the following is well-formed:
    //
    // ```mlir
    // kgen.generator @foo<() -> x>() {
    //   kgen.call @foo<() -> y>()
    //   %0 = kgen.param.constant = <y>
    //   kgen.param.result_bind<2>
    //   kgen.return
    // }
    // ```
    //
    // It will be rejected as forbidden, because analyzing which operations to
    // defer would be too complex, and it could result in recursively deferring
    // operations if, for example, another recursive call would depend on `y`.
    //
    // In multi-threaded execution, call resolution is also deferred as late as
    // possible. This maximizes parallelism on the expansion graph (without
    // intra-node parallelism) while correctly handling recursion.
    if (from && from->incoming.contains(genNode))
      return ElaborationState::advance();
    return ElaborationState::skipNode();
  }

  // Flatten the callgraph up to this point.
  genNode->incoming.insert(genNode);
  if (from)
    genNode->incoming.insert(from->incoming.begin(), from->incoming.end());

  GeneratorOp generator = genNode->gen;

  // Bind all parameter values in this scope.
  ArrayRef<Attribute> inputParamValues = genNode->inputParams.getValue();
  ArrayRef<ParamDeclAttr> inputParamDecls = generator.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  IREvaluator evaluator(*this);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setOrOverwriteParameterValue(decl, val);

  // If the generator's constraints don't satisfy, set an error and move on.
  if (auto err =
          KGEN::evaluateConstraints(generator.getConstraints(), evaluator)) {
    // This node is complete. It can never be valid.
    genNode->constraintError = std::move(err.value());
    genNode->status = ParamNode::DONE;
    genNode->paramCh.copy().emplace();
    return ElaborationState::error();
  }

  TimeTraceScope<> traceScope(
      "specializeGenerator:" + tryGettingShortName(generator.getName()).str(),
      generator.getName().str() /* + " Input params: " +
          mlir::debugString(generator.getInputParamsAttr()) + " = " +
          mlir::debugString(genNode->inputParams) */);
  auto genScope =
      logger.scope("Specializing Generator: @", generator.getName());
  logger.logOp("Generator", generator);

  // Get a partial ordering of parameter definitions and uses that are listed
  // "top down" in our evaluation order, if we don't have one already. This
  // should happen exactly once for each generator node. This will be tricky to
  // parallelize as-is - we should change the approach a bit to have a
  // ParametricNode (or similar) that doesn't store the input parameters, in
  // which we could store the ParameterUseDefGraph.
  ParameterUseDefGraph *genNodeGraph =
      knownGraphs.read([generator](const auto &map) -> ParameterUseDefGraph * {
        if (auto it = map.find(generator); it != map.end())
          return it->second.get();
        return nullptr;
      });
  if (!genNodeGraph) {
    // Compute a new graph. The computed graph could end up getting discarded if
    // two threads end up here at the same time for the same generator.
    auto newGraph =
        std::make_unique<ParameterUseDefGraph>(generator.getBodyRegion());
    newGraph->calculate(paramCache.getThreadLocalCache());
    // Make sure to use whichever graph ended up in the map.
    genNodeGraph = knownGraphs.modify(
        [generator, newGraph = std::move(newGraph)](auto &map) mutable {
          return map.try_emplace(generator, std::move(newGraph))
              .first->second.get();
        });
  }

  std::string baseName = mangleParameterValues(generator, inputParamValues);

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(generator.getContext());
  auto newFunc = b.create<FuncOp>(
      generator.getLoc(),
      b.getStringAttr(baseName +
                      Twine(inputParamValues.empty() ? "_concrete" : "")),
      SignatureType::get(TypeArrayAttr::get(generator.getContext(), {}),
                         TypeArrayAttr::get(generator.getContext(), {}),
                         generator.getFunctionType(), generator.getMetadata()),
      generator.getAlwaysInlineLevel());

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  symtab.modify([newFunc, it = generator->getIterator()](SymbolTable &symtab) {
    symtab.insert(newFunc, it);
  });

  // Clone the body of the generator into the function.
  // TODO: is there a nice way for us to avoid cloning this?
  IRMapping map;
  generator.getBodyRegion().cloneInto(&newFunc.getBodyRegion(), map);

  // Map from the generator to the new function for the parameter graph copy.
  map.map(generator.getOperation(), newFunc.getOperation());
  // Copy over the parameter use-def graph for this clone.
  ParameterUseDefGraph childGraph = genNodeGraph->copy(map);

  // The node for this new func is simply the child of the node for the
  // generator.
  auto childNode = std::make_unique<ImplNode>(
      newFunc, genNode, std::move(childGraph), std::move(baseName));
  g.concreteNodes.modify(
      [newFunc, node = childNode.get()](DenseMap<FuncOp, ImplNode *> &map) {
        map.try_emplace(newFunc, node);
      });
  ImplNode *newFuncNode = childNode.get();
  genNode->impls.push_back(std::move(childNode));
  ParameterUseDefGraph &uses = newFuncNode->paramGraph;

  // Kick off the expansion for the new function.

  auto funcScope = logger.scope("Specializing Function: @", newFunc.getName());
  logger.logOp("Function", newFunc);

  std::vector<Operation *> opsToRewrite;
  llvm::append_range(opsToRewrite, llvm::reverse(uses.paramOps));
  opsToRewrite.push_back(newFunc);
  collectOpsToProcess(&newFunc.getBodyRegion(), uses, opsToRewrite);

  ImplNode::WorkItem item{std::move(opsToRewrite), std::move(evaluator),
                          [](ImplNode *) { return success(); }};
  newFuncNode->stack.push_back(std::move(item));

  initialScheduleImplNode(newFuncNode);
  return ElaborationState::skipNode();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::run
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::run(ModuleOp theModule,
                                  ArrayRef<GeneratorOp> primaryGenerators) {
  LLVM_DEBUG(logger << "Starting Elaboration\n");

  // Detect common errors early and report them cleanly.
  for (auto &op : theModule.getOps())
    if (op.getName().getStringRef() == "lit.func")
      return op.emitError("unlowered lit.func discovered in KGEN elaborator");

  auto emptyInputParamKey = ArrayAttr::get(theModule.getContext(), {});
  bool failed = false;
  for (auto gen : primaryGenerators) {
    LLVM_DEBUG(logger.logOp("Elaborating primary generator", gen));
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    ParamNode *generatorNode =
        g.getOrCreate(runtime, emptyInputParamKey, gen, /*depth=*/0);

    // Now we can begin to construct the expansion tree rooted at this
    // generator. Emit as many errors as possible.
    if (ErrorTreeOrSuccess err = specializeGeneratorAndWait(generatorNode);
        err.isError()) {
      err.takeError().emit([](Location loc) { return mlir::emitError(loc); });
      failed = true;
    }
  }
  // Wait for all work to complete.
  signalWorklist();
  LLCL::await(g.worklistCh);
  if (failed)
    return failure();

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  DenseMap<StringAttr, StringAttr> funcsToRename;
  for (ParamNode &node :
       llvm::make_pointee_range(llvm::make_second_range(g.nodes.get()))) {
    FuncOp first;
    // Erase all erroneous functions.
    for (ImplNode &impl : llvm::make_pointee_range(node.impls)) {
      if (impl.error)
        symtab.get().erase(impl.func);
      else if (!first)
        first = impl.func;
    }

    // Rename the first successful function for concrete top-level generators,
    // if there is one.
    symtab.get().remove(node.gen);
    if (node.inputParams.empty() && first) {
      StringAttr newName = node.gen.getSymNameAttr();
      funcsToRename[first.getNameAttr()] = newName;
      symtab.get().remove(first);
      first.setSymNameAttr(newName);
      symtab.get().insert(first);
    }
  }

  // Erase all generators.
  for (auto gen : llvm::make_early_inc_range(theModule.getOps<GeneratorOp>()))
    symtab.get().erase(gen);

  // Perform any renaming at the end.  We cannot use the
  // SymbolTable::replaceAllSymbolUses method, because it doesn't tolerate
  // unregistered operations.  It also doesn't support batch renaming.
  theModule->walk([&](KGENCallOpInterface call) {
    // If this is a reference to a function that got renamed, update its
    // target.
    auto callee = cast<SymbolConstantAttr>(call.getCallee());
    if (StringAttr newName = funcsToRename.lookup(
            cast<FlatSymbolRefAttr>(callee.getSymbol()).getAttr())) {
      call.updateCallee(SymbolConstantAttr::get(FlatSymbolRefAttr::get(newName),
                                                callee.getType()));
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// M::KGEN::elaborateGenerators
//===----------------------------------------------------------------------===//

LogicalResult M::elaborateGenerators(mlir::SymbolTableAnalysis &symtab,
                                     ParameterCollector::Analysis &paramCache,
                                     LLCL::Runtime &runtime,
                                     TargetInfoAttr target,
                                     ArrayRef<GeneratorOp> primaryGenerators,
                                     EvaluatorExecutorFnRef evaluatorExecutorFn,
                                     const ElaboratorConfig &config) {
  TimeTraceScope<> traceScope("elaborate-generators");
  ModuleOp theModule = symtab.getTopLevelOp<ModuleOp>();

  AsyncSideEffectMap asyncMap(runtime);

  auto transformCacheBackendOr = Cache::getLocalDefaultBackendChain(
      runtime, ".kgen_cache/transform", KGEN_VERSION_STRING);
  if (failed(transformCacheBackendOr))
    return theModule->emitError() << transformCacheBackendOr.getError();
  auto regionCacheBackendOr = Cache::getLocalDefaultBackendChain(
      runtime, ".kgen_cache/region", KGEN_VERSION_STRING);
  if (failed(regionCacheBackendOr))
    return theModule->emitError() << regionCacheBackendOr.getError();

  auto transformCache =
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
          transformCacheBackendOr.takeValue());
  auto regionCache =
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>>::create(
          regionCacheBackendOr.takeValue());

  // Now, construct and run the elaborator.
  ElaboratorImpl impl(
      symtab.getTopLevelSymbolTable(), paramCache, target,
      config.enableSearch ? evaluatorExecutorFn
                          : [](FuncOp, const SymbolTable &, TargetInfoAttr,
                               ArrayRef<FuncOp>) { return 0; },
      runtime, config);
  return impl.run(theModule, primaryGenerators);
}

//===----------------------------------------------------------------------===//
// ElaborateGeneratorsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ELABORATEGENERATORS
#define GEN_PASS_DEF_RESOLVEINCLUDES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Run the elaborator as a pass. The elaborator requires imports to be
/// resolved, so first resolve imports and then elaborate.
class ElaborateGeneratorsPass
    : public KGEN::impl::ElaborateGeneratorsBase<ElaborateGeneratorsPass> {
public:
  ElaborateGeneratorsPass(const ElaborateGeneratorsOptions &options = {},
                          LLCL::Runtime *runtime = nullptr,
                          TargetInfoAttr target = nullptr,
                          BuildInfoAttr build = nullptr,
                          EvaluatorExecutorFn evaluatorExecutorFn = {})
      : ElaborateGeneratorsBase(options), runtime(runtime), target(target),
        build(build), evaluatorExecutorFn(std::move(evaluatorExecutorFn)) {}

  LogicalResult initialize(MLIRContext *ctx) override {
    // Default to the host target if one was not specified
    if (!target) {
      ErrorOr<TargetInfoAttr> targetOr =
          getTargetInfoFor(ctx, llvm::sys::getDefaultTargetTriple(),
                           llvm::sys::getHostCPUName(), getHostCPUFeatures());
      if (targetOr.isError())
        return mlir::emitError(UnknownLoc::get(ctx), targetOr.getError());
      target = targetOr.takeValue();
    }
    // Default to the host build if one was not specified
    if (!build)
      build = BuildInfoAttr::getForCurrentBuild(ctx);
    // Default the evaluator to selecting the first specialization.
    if (!evaluatorExecutorFn) {
      evaluatorExecutorFn = [](KGEN::FuncOp evaluator,
                               const SymbolTable &symtab, TargetInfoAttr target,
                               ArrayRef<KGEN::FuncOp> specializations) {
        return 0;
      };
    }
    return success();
  }

  void runOnOperation() override {
    auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
        runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
        LLCL::createSingleThreadWorkQueue());

    ModuleOp theModule = getOperation();

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

    // Collect exports - we don't want to elaborate generators that are not
    // exported.
    DenseSet<GeneratorOp> exports;
    for (auto e : theModule.getOps<ExportOp>())
      if (auto gen = analysis.getTopLevelSymbolTable().lookup<GeneratorOp>(
              cast<FlatSymbolRefAttr>(e.getExported()).getValue()))
        exports.insert(gen);

    // Extract the top-level, parameterless generators from the main module.
    // These are the only generators that will be elaborated.
    SmallVector<GeneratorOp> primaryGenerators;
    for (auto gen : theModule.getOps<GeneratorOp>())
      if (gen.getInputParams().empty() &&
          (exports.empty() || exports.contains(gen)))
        primaryGenerators.push_back(gen);

    // Elaboration is the compilation phase in which the IR goes from
    // target-non-specific to target-specific: in order to fully concretize the
    // IR, we must evaluate compile-time expressions, which is a target-specific
    // operation. Make the IR target-specific by attaching the required target
    // specification.
    if (TargetInfoAttr tgt = getTargetInfo(theModule)) {
      if (tgt != target) {
        theModule->emitError("target did not match, expected ")
            << target << " but got " << tgt;
        return signalPassFailure();
      }
    } else {
      setTargetInfo(theModule, target);
    }

    // Same for the target info, the build info is concretized in the IR.
    if (BuildInfoAttr bld = getBuildInfo(theModule)) {
      if (bld != build) {
        mlir::emitError(theModule.getLoc(), "build did not match, expected ")
            << build << " but got " << bld;
        return signalPassFailure();
      }
    } else {
      setBuildInfo(theModule, build);
    }

    if (failed(elaborateGenerators(
            analysis, paramCache, *rt, target, primaryGenerators,
            evaluatorExecutorFn,
            ElaboratorConfig{shouldDoSearch, testDiagnostics, maxDepth})))
      return signalPassFailure();
  }

private:
  /// An optional LLCL runtime pointer.
  LLCL::Runtime *runtime;
  /// The compilation target.
  TargetInfoAttr target;
  /// The build target.
  BuildInfoAttr build;
  /// The functor used for evaluating generator specializations.
  EvaluatorExecutorFn evaluatorExecutorFn;
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(LLCL::Runtime &runtime, TargetInfoAttr target,
                                BuildInfoAttr build,
                                const ElaborateGeneratorsOptions &options,
                                EvaluatorExecutorFn evaluatorExecutorFn) {
  return std::make_unique<ElaborateGeneratorsPass>(
      options, &runtime, target, build, std::move(evaluatorExecutorFn));
}
