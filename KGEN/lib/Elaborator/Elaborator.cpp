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
  ImplNode(FuncOp func, ParamNode *parent, IREvaluator &&evaluator,
           ParameterUseDefGraph &&graph)
      : func(func), parent(parent), evaluator(std::move(evaluator)),
        paramGraph(std::move(graph)) {}

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

  /// This function represents a concrete instantiation of a generator.
  FuncOp func;
  /// The parent expansion tree node.
  ParamNode *parent;

  /// The evaluator is shared with scopes below, but not scopes above,
  /// generally. That's why it's copied rather than taking a reference.
  IREvaluator evaluator;
  /// Keep track of the nested parameter scopes within this function.
  ParameterUseDefGraph paramGraph;

  /// Calls to the same interface/generator should resolve to the same thing in
  /// each func.
  DenseMap<std::pair<ArrayAttr, GeneratorOp>, ImplNode *> bindings;
  /// When you have result parameters, we need to store them to access them from
  /// outer scopes.
  ArrayAttr resultParams;
  /// An error contained by this node. This allows us to delay error handling in
  /// cases where an error is recoverable.
  std::optional<ErrorTree> error;

  /// The current state of elaboration on the function.
  enum { FRESH, IN_PROGRESS, DONE } status = FRESH;
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
  ParamNode(GeneratorOp gen, ArrayAttr vals) : gen(gen), inputParams(vals) {}

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

  /// The children of a node are specializations. They may not be fully concrete
  /// in the case of e.g. an interface - where the children are generators that
  /// themselves have children.
  std::vector<std::unique_ptr<ImplNode>> impls;

  /// The current state of the node. This flag is used to break recursion.
  enum { FRESH, IN_PROGRESS, DONE } status = FRESH;
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
  /// Map from generator instantiation to expansion tree node.
  DenseMap<std::pair<ArrayAttr, GeneratorOp>, std::unique_ptr<ParamNode>> nodes;

  /// Map from concrete function to implementation node.
  DenseMap<FuncOp, ImplNode *> concreteNodes;

  /// Fork the expansion of a concrete node.
  ImplNode *fork(ImplNode *cur, IRMapping &map, SymbolTable &symtab) {
    auto clone = cast<FuncOp>(cur->func->clone(map));
    symtab.insert(clone, ++cur->func->getIterator());
    auto n = std::make_unique<ImplNode>(clone, cur->parent,
                                        IREvaluator(cur->evaluator),
                                        cur->paramGraph.copy(map));
    ImplNode *result = n.get();
    cur->parent->impls.push_back(std::move(n));
    concreteNodes.try_emplace(clone, result);
    return result;
  }

  /// Get or create the node for a generator instantiation.
  ParamNode *getOrCreate(ArrayAttr values, GeneratorOp gen) {
    std::unique_ptr<ParamNode> &n = nodes[{values, gen}];
    if (!n)
      n = std::make_unique<ParamNode>(gen, values);
    return n.get();
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
static ErrorTreeOrSuccess processParamDeclareOp(IREvaluator &evaluator,
                                                ParamDeclareOp op) {
  // Simplify the input expression.
  auto errorOrValue =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getValue());
  if (errorOrValue.isError())
    return errorOrValue.takeError();

  // Bind it to the parameter declaration it is setting.
  evaluator.setOrOverwriteParameterValue(op.getParamDecl(),
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
static ErrorTreeOrSuccess processParamResultBindOp(ParamResultBindOp op,
                                                   ImplNode *parentNode) {
  // Concretize the result parameter values.
  IREvaluator &evaluator = parentNode->evaluator;
  SmallVector<Attribute> resultParams;

  // Retrieve the required parameter decls from the nearest declaration.
  // However, if it refers to the function being elaborated, the declarations
  // are in the generator.
  ArrayRef<ParamDeclAttr> resultParamDecls;
  auto parentDecl = op->getParentOfType<DeclInterface>();
  bool isFunc = isa<FuncOp>(parentDecl.getOperation());
  if (isFunc)
    resultParamDecls = parentNode->parent->gen.getResultParams();
  else
    resultParamDecls = parentDecl.getResultParams();

  for (auto [decl, value] : llvm::zip(resultParamDecls, op.getParameters())) {
    ErrorTreeOr<Attribute> concValue =
        evaluator.concretizeParameterExpr(op.getLoc(), value);
    if (concValue.isError())
      return concValue.takeError();
    resultParams.push_back(concValue.takeValue());
    evaluator.setOrOverwriteParameterValue(decl, resultParams.back());
  }

  // If this operation binds values for the result parameters of the generator,
  // set them in the node.
  if (isFunc)
    parentNode->resultParams = ArrayAttr::get(op.getContext(), resultParams);

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// processRebindOp
//===----------------------------------------------------------------------===//

static ErrorTreeOrSuccess processRebindOp(IREvaluator &evaluator, RebindOp op) {
  ErrorTreeOr<Type> outType =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getType());
  if (outType.isError())
    return outType.takeError();
  if (outType.getValue() != op.getInput().getType()) {
    return ErrorTree(op.getLoc(), "operand and result type of rebind operation "
                                  "did not concretize to the same type");
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
static ErrorTreeOrSuccess processParamAssertOp(IREvaluator &evaluator,
                                               ParamAssertOp op) {
  // Check the condition expression.
  auto errorOrValue =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getCond());
  if (errorOrValue.isError())
    return errorOrValue.takeError();

  // If the constraint evaluated to zero then the assert fails.
  auto resultInt = cast<IntegerAttr>(errorOrValue.takeValue());
  if (resultInt.getValue().isZero()) {
    // Evaluate the string to report it.
    errorOrValue =
        evaluator.concretizeParameterExpr(op.getLoc(), op.getMessage());

    StringAttr message;
    if (!errorOrValue.isError())
      message = dyn_cast<StringAttr>(errorOrValue.takeValue());

    return ErrorTree(op.getLoc(),
                     "constraint failed: " +
                         (message ? message.getValue() : "<unknown>"));
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
static ErrorTreeOrSuccess processGenericOp(IREvaluator &evaluator,
                                           Operation *op) {
  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    ErrorTreeOr<Attribute> value = evaluator.concretizeParameterExpr(
        op->getLoc(), namedAttr.getValue(), /*allowUnknown=*/true);
    if (value.isError())
      return value.takeError();

    newAttrs.emplace_back(namedAttr.getName(), value.takeValue());
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  if (ErrorTreeOrSuccess err = processLocation(evaluator, op); err.isError())
    return err.takeError();

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults()) {
    ErrorTreeOr<Type> type =
        evaluator.concretizeParameterExpr(op->getLoc(), result.getType());
    if (type.isError())
      return type.takeError();
    result.setType(type.takeValue());
  }

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Value arg : block.getArguments()) {
        ErrorTreeOr<Type> type =
            evaluator.concretizeParameterExpr(op->getLoc(), arg.getType());
        if (type.isError())
          return type.takeError();
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
// printParameterValue
//===----------------------------------------------------------------------===//

/// Pretty-print the value of a parameter.
static void printParameterValue(TypedAttr value, raw_ostream &os) {
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
static StringAttr mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<Attribute> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return b.getStringAttr(generator.getName() + "_concrete");

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getInputParamsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterValue(cast<TypedAttr>(value), os);
  }
  return b.getStringAttr(result);
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
  llvm::append_range(opsToRewrite, llvm::reverse(defOps.getArrayRef()));
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl Declaration
//===----------------------------------------------------------------------===//

/// This class provides the elaborator, which constructs the expansion tree as
/// it walks the IR and specializes operations. This outputs IR that has been
/// fully specialized/concretized, with the appropriate functions
/// multi-versioned.
namespace {
class ElaboratorImpl : public Elaborator {
public:
  ElaboratorImpl(mlir::SymbolTableAnalysis &analysis,
                 ParameterCollector::Analysis &paramCache,
                 TargetInfoAttr target,
                 EvaluatorExecutorFnRef evaluatorExecutorFn,
                 const ElaboratorConfig &config)
      : Elaborator(analysis, paramCache, target, evaluatorExecutorFn),
        config(config) {}

  ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) override;

  ErrorTreeOrSuccess
  getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                          ArrayRef<TypedAttr> paramValues,
                          std::vector<FuncOp> &funcs) override;

  /// Lookup a symbol of type T in the symbol table collection.
  template <typename T>
  T lookup(SymbolRefAttr symbol) {
    return analysis.getSymbolTables().lookupSymbolIn<T>(
        analysis.getTopLevelOp<ModuleOp>(), symbol);
  }

  /// Lookup a symbol of any type in the symbol table collection.
  Operation *lookup(StringAttr symbol) {
    return analysis.getTopLevelSymbolTable().lookup(symbol);
  }
  Operation *lookup(SymbolRefAttr symbol) {
    return lookup(symbol.getRootReference());
  }

  /// Once a concrete function has finished specializing, finish processing the
  /// function and call the verifier.
  void finalizeAndVerifyFunction(mlir::SymbolTableAnalysis &analysis,
                                 ImplNode *node);

  /// Process a kgen.param.fork op. This will create a clone for each value of
  /// the parameter search, and will mark the parent as an error. This results
  /// in a very clean model where the parent of the current parent (a generator)
  /// will have its children be the successfully concretized parameter search
  /// nodes.
  ErrorTreeOrSuccess
  processParamForkOp(ImplNode *parent, ParamForkOp op,
                     ArrayRef<Operation *> remainingWorklist);

  /// Spawn a clone for kgen.param.fork. This creates a new FuncOp that is a
  /// sibling to the parent of the kgen.param.fork op. It replaces the
  /// kgen.param.fork with a param.declare to allow specialization to succeed.
  ErrorTreeOrSuccess
  spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                      ImplNode *forkParentNode,
                      ArrayRef<Operation *> remainingWorklist);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  ErrorTreeOrSuccess processCallOp(KGENCallOpInterface call, ImplNode *parent,
                                   ArrayRef<Operation *> remainingWorklist);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  ErrorTreeOrSuccess
  processGeneratorUser(KGENCallOpInterface user,
                       SymbolConstantAttr calleeSymbol, ImplNode *parent,
                       ArrayRef<Operation *> remainingWorklist);

  /// Complete processing of a generator user by resolving any bound result
  /// types or parameters in the parent scope. This is the step that propagates
  /// result parameters from the inner scope to the outer scope.
  ErrorTreeOrSuccess completeCallProcessing(KGENCallOpInterface user,
                                            ArrayRef<ParamDeclAttr> decls,
                                            ImplNode *thisNode,
                                            ImplNode *parentNode,
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
  ErrorTreeOrSuccess
  processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                              ImplNode *parent);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  ErrorTreeOrSuccess processParamIfOp(ParamIfOp op, ImplNode *parent);

  /// Process a worklist of ops. Returns failure if the scope produced an error.
  LogicalResult processScope(ImplNode *parentNode,
                             ArrayRef<Operation *> worklist);

  /// Specializes the generator at `genNode`. Essentially instantiates a new
  /// function with the same body, and specializes it. The new function is by
  /// definition the expansion tree child of this generator.
  ErrorTreeOrSuccess specializeGenerator(ParamNode *genNode);

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ArrayRef<GeneratorOp> primaryGenerators);

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
  DenseMap<GeneratorOp, std::unique_ptr<ParameterUseDefGraph>> knownGraphs;

  /// The callgraph being expanded.
  ExpansionGraph g;

  /// The current depth of the elaborator.
  size_t depth = 0;

  /// The elaborator config.
  ElaboratorConfig config;

  /// Remove parameter declare regions after generator elaboration.
  DenseMap<StringAttr, ParameterUseDefGraph *> knownRegions;
  SmallVector<OwningOpRef<ParamDeclareRegionOp>> paramDeclareRegionOps;
};
} // namespace

//===----------------------------------------------------------------------===//
// finalizeAndVerifyFunction
//===----------------------------------------------------------------------===//

void ElaboratorImpl::finalizeAndVerifyFunction(
    mlir::SymbolTableAnalysis &analysis, ImplNode *node) {
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
ElaboratorImpl::getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                                    ArrayRef<TypedAttr> paramValues) {
  auto gen = lookup<GeneratorOp>(symbolRef);

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(vals, gen);
  // If the node has already been elaborated, just use that result.
  if (node->status != ParamNode::DONE) {
    if (ErrorTreeOrSuccess err = specializeGenerator(node); err.isError())
      return err.takeError();
  }

  return node->getFirstConcreteFunc();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getAllConcreteFunctions
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
ElaboratorImpl::getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                                        ArrayRef<TypedAttr> paramValues,
                                        std::vector<FuncOp> &funcs) {
  auto gen = lookup<GeneratorOp>(symbolRef);

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(vals, gen);
  if (node->status != ImplNode::DONE) {
    if (ErrorTreeOrSuccess err = specializeGenerator(node); err.isError())
      return err.takeError();
  }
  node->getAllConcreteFuncs(funcs);
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamForkOp
//===----------------------------------------------------------------------===//

/// Process a kgen.param.fork op.
ErrorTreeOrSuccess
ElaboratorImpl::processParamForkOp(ImplNode *parent, ParamForkOp op,
                                   ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Processing ParamForkOp");
  LLVM_DEBUG(logger.scope("Options") << op.getValuesAttr() << "\n");

  // Loop over all the possible candidates that we will search over, spawning
  // N possibilities to explore.
  SmallVector<ErrorTree> errors;
  DenseSet<Attribute> seenValues;

  ErrorTreeOr<Attribute> errorOrValue =
      parent->evaluator.concretizeParameterExpr(op.getLoc(),
                                                op.getValuesAttr());
  if (errorOrValue.isError())
    return errorOrValue.takeError();

  auto forkValuesAttr = cast<VariadicAttr>(errorOrValue.takeValue());

  if (forkValuesAttr.getValues().empty())
    return ErrorTree(op.getLoc(), "no candidates found");

  bool atLeastOneSuccessful = false;
  for (Attribute candidate : forkValuesAttr.getValues()) {
    // Simplify the input expressions.
    ErrorTreeOr<Attribute> errorOrValue =
        parent->evaluator.concretizeParameterExpr(op.getLoc(), candidate);
    if (errorOrValue.isError()) {
      errors.push_back(errorOrValue.takeError());
      continue;
    }

    Attribute value = errorOrValue.takeValue();

    // If we've already seen this concrete value before,
    // ignore the duplicate.
    if (!seenValues.insert(value).second)
      continue;

    // Otherwise, spawn a clone for this value. If that fails, continue.
    if (ErrorTreeOrSuccess err =
            spawnParamForkClone(op, value, parent, remainingWorklist);
        err.isError()) {
      errors.push_back(err.takeError());
      continue;
    }

    // If search is disabled, break after the first successful parameter.
    atLeastOneSuccessful = true;
    if (!config.enableSearch)
      break;
  }

  // If we don't have at least one successful candidate, fail.
  if (!atLeastOneSuccessful)
    return ErrorTree(op.getLoc(), "some expansions failed", errors);

  // The parent has to be deleted.
  parent->setToError(ErrorTree(op.getLoc(), "param fork base node"));
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::spawnParamForkClone
//===----------------------------------------------------------------------===//

/// Spawn a clone from a kgen.param.fork op.
ErrorTreeOrSuccess
ElaboratorImpl::spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                                    ImplNode *forkParentNode,
                                    ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Spawning ParamForkClone for '", value, "'");

  // Start by cloning the current WIP func to a new copy of it.
  IRMapping map;

  // Hook this new clone up correctly.
  ImplNode *newFuncNode =
      g.fork(forkParentNode, map, analysis.getTopLevelSymbolTable());

  // Change the future of this func by resolving the forkOp in the new func
  // to the specified value.
  auto newFork = cast<ParamForkOp>(map.lookup(forkOp.getOperation()));

  LLVM_DEBUG(logger << "Setting '" << newFork.getParamDecl() << "' = '" << value
                    << "'\n");

  // Update the evaluator.
  newFuncNode->evaluator.setOrOverwriteParameterValue(newFork.getParamDecl(),
                                                      value);
  newFork->erase();

  // Map to the new ops.
  auto remaining = map_to_vector(remainingWorklist,
                                 [&](Operation *op) { return map.lookup(op); });

  // And finally, process the rest of the worklist in this new scope. If we've
  // hit an error case, don't try and finish processing. Return to the upper
  // function that this hit an error.
  newFuncNode->status = ImplNode::IN_PROGRESS;
  LogicalResult result = processScope(newFuncNode, remaining);
  newFuncNode->status = ImplNode::DONE;
  if (failed(result))
    return newFuncNode->error->copy();

  finalizeAndVerifyFunction(analysis, newFuncNode);
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processGeneratorUser
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ElaboratorImpl::processGeneratorUser(
    KGENCallOpInterface user, SymbolConstantAttr calleeSymbol, ImplNode *parent,
    ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Processing Generator User");
  LLVM_DEBUG(logger.logOp("User", user));

  assert(remainingWorklist.empty() || remainingWorklist.front() != user);

  // Add in the mapping for parameters in the calls.
  auto resolvedCallParamsOr = resolveCallInputParams(
      user, parent->evaluator, calleeSymbol.getParamValues());
  if (resolvedCallParamsOr.isError())
    return resolvedCallParamsOr.takeError();
  ArrayAttr inputParamKey = *resolvedCallParamsOr;

  // Lookup the callee.
  auto calleeOp = lookup(calleeSymbol.getSymbol());
  if (!calleeOp) {
    return ErrorTree(user.getLoc(), "could not find callee '" +
                                        mlir::debugString(calleeSymbol) + "'");
  }

  ArrayRef<ParamDeclAttr> decls = user.getParamDecls();
  if (auto func = dyn_cast<FuncOp>(calleeOp)) {
    auto it = g.concreteNodes.find(func);
    if (it == g.concreteNodes.end())
      return ErrorTree(user.getLoc(), "concrete callee doesn't have a node "
                                      "(compiler bug -- file an issue!)");
    return completeCallProcessing(user, decls, it->second, parent, logger);
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

  // Find the tree node that corresponds to the thing we're calling.
  ParamNode *calleeNode = g.getOrCreate(inputParamKey, gen);
  if (ErrorTreeOrSuccess err = specializeGenerator(calleeNode); err.isError())
    return ErrorTree(user.getLoc(), "call expansion failed", err.takeError());

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
    for (ImplNode &impl : llvm::make_pointee_range(calleeNode->impls))
      if (impl.error)
        out.addCause(impl.error->copy());
    return std::move(out);
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
    ImplNode *newNode = g.fork(parent, map, analysis.getTopLevelSymbolTable());
    newNode->bindings = parent->bindings;
    // Bind this concrete impl to this callee for this node.
    newNode->bindings[{inputParamKey, gen}] = c;

    if (ErrorTreeOrSuccess err = completeCallProcessing(
            cast<KGENCallOpInterface>(map.lookup(user.getOperation())), decls,
            c, newNode, logger);
        err.isError())
      return err;

    LLVM_DEBUG(newNode->print(logger << "New Op "));

    // We have to finish specializing this thing now. Map to the new ops and
    // process the remaining scope.
    auto remaining = map_to_vector(
        remainingWorklist, [&](Operation *op) { return map.lookup(op); });

    // Process the rest of the worklist in this new scope. If the scope
    // processing failed, do nothing.
    newNode->status = ImplNode::IN_PROGRESS;
    if (succeeded(processScope(newNode, remaining)))
      finalizeAndVerifyFunction(analysis, newNode);
    newNode->status = ImplNode::DONE;
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
static ErrorTreeOrSuccess processParamApplyOp(ParamApplyOp op, FuncOp func,
                                              IREvaluator &evaluator) {
  SmallVector<TypedAttr> operands;
  for (TypedAttr operand : op.getOperands()) {
    ErrorTreeOr<Attribute> value =
        evaluator.concretizeParameterExpr(op.getLoc(), operand);
    if (value.isError())
      return value.takeError();
    operands.push_back(cast<TypedAttr>(value.takeValue()));
  }
  ErrorTreeOr<TypedAttr> result = evaluator.evaluateFunction(func, operands);
  if (result.isError())
    return result.takeError();

  // Bind the result and erase the operation.
  evaluator.setOrOverwriteParameterValue(op.getParamDecl(), result.takeValue());
  op.erase();
  return {};
}

ErrorTreeOrSuccess ElaboratorImpl::completeCallProcessing(
    KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls, ImplNode *thisNode,
    ImplNode *parentNode, Logger &logger) {

  // Add the callee's bindings to the parent of the call. This ensures that we
  // don't re-bind something we've already bound.
  for (const auto &[k, v] : thisNode->bindings) {
    auto &oldV = parentNode->bindings[k];
    assert(!oldV || oldV == v);
    oldV = v;
  }

  if (thisNode->error)
    return {};

  FuncOp newCalleeFunc = thisNode->func;

  // If this is a `kgen.param.apply`, bind its result here.
  if (auto apply = dyn_cast<ParamApplyOp>(*user))
    return processParamApplyOp(apply, newCalleeFunc, parentNode->evaluator);

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (auto result : user->getResultTypes()) {
    ErrorTreeOr<Type> typeOr =
        parentNode->evaluator.concretizeParameterExpr(user.getLoc(), result);
    if (typeOr.isError()) {
      thisNode->setToError(typeOr.takeError());
      return {};
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

  // If we don't have the result parameters yet, then either no result
  // parameters are necessary, or we have another problem entirely wherein we
  // could not complete the callee's result parameter resolution at all - likely
  // meaning we're in an infinite recursive loop. Essentially, we came back to
  // the same combination of generator + input parameters without resolving the
  // result parameters yet.
  ArrayAttr resultParams = thisNode->resultParams;
  if (!resultParams && !decls.empty()) {
    thisNode->setToError(
        ErrorTree(userLoc, "could not resolve callee's necessary result "
                           "parameters, infinite recursive loop?"));
    return {};
  }
  // No decls, so we don't have to do anything.
  if (decls.empty())
    return {};

  // Bind the result parameters to the output parameter decls.
  assert(decls.size() == resultParams.size());
  for (auto [decl, bindValue] : llvm::zip(decls, resultParams)) {
    LLVM_DEBUG(logger << "Binding " << decl << " to " << bindValue << "\n");
    parentNode->evaluator.setOrOverwriteParameterValue(decl, bindValue);
  }
  return {};
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
ErrorTreeOrSuccess
ElaboratorImpl::processCallOp(KGENCallOpInterface call, ImplNode *parent,
                              ArrayRef<Operation *> remainingWorklist) {
  ErrorTreeOr<Attribute> symbol = parent->evaluator.concretizeParameterExpr(
      call.getLoc(), call.getCallee());
  if (symbol.isError())
    return symbol.takeError();

  if (auto sym = dyn_cast<SymbolConstantAttr>(*symbol))
    return processGeneratorUser(call, sym, parent, remainingWorklist);

  auto decl = dyn_cast<RegionAttr>(*symbol);
  if (!decl)
    return ErrorTree(
        call.getLoc(),
        "concrete parameter must be a SymbolConstantAttr or a RegionAttr");

  // OK we found a region, put it into the machinery.
  ParameterUseDefGraph *regionGraph = knownRegions.lookup(decl.getRegionName());
  Region *region = regionGraph->scope;
  LLVM_DEBUG(logger.logOp("Inlining call to parameter region:",
                          region->getParentOp()));

  // Inline the call now. We clone them now so that we don't modify the original
  // region in case it's re-used.
  IRMapping map;
  inlineCallToConcreteRegion(call, region, map, AlwaysInlineLevel::Enabled);

  // Collect all the ops to process *in the region*.
  std::vector<Operation *> opsToRewriteInRegion;
  collectOpsToProcess(region, *regionGraph, opsToRewriteInRegion);
  llvm::append_range(opsToRewriteInRegion, regionGraph->paramOps);
  auto opsToRewrite = map_to_vector(
      opsToRewriteInRegion, [&](Operation *op) { return map.lookup(op); });

  // Process the ops we just collected.
  {
    llvm::SaveAndRestore<IREvaluator> save(parent->evaluator);
    parent->evaluator.clearCache();

    // Set any bindings on the region in the evaluator context.
    for (auto [decl, value] :
         llvm::zip(region->getParentOfType<DeclInterface>().getInputParams(),
                   decl.getParamValues()))
      parent->evaluator.setOrOverwriteParameterValue(decl, value);

    if (failed(processScope(parent, opsToRewrite)))
      return parent->error->copy();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamDeclareRegionOp
//===----------------------------------------------------------------------===//

/// Process a param.declare.region by creating a generator for the contained
/// region.
ErrorTreeOrSuccess
ElaboratorImpl::processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                                            ImplNode *parent) {
  StringAttr regionName = getMangledRegionParamName(regionDecl);
  // Set the region's parameter decl as the value for this name. That will
  // signal to the call_param handler that it needs to inline the region it
  // finds for that decl. The region attr itself holds onto a bit that knows if
  // the region is isolated from above (in SSA-land) or not.
  parent->evaluator.setOrOverwriteParameterValue(
      regionDecl.getParamDecl().getName(),
      RegionAttr::get(
          regionName, {},
          BoolAttr::get(regionDecl.getContext(),
                        operationIsIsolatedFromAbove(regionDecl)),
          cast<SignatureType>(regionDecl.getParamDecl().getType())));
  auto found =
      parent->paramGraph.nestedScopes.find(&regionDecl.getBodyRegion());
  assert(found != parent->paramGraph.nestedScopes.end() &&
         "must have a nested region");
  LLVM_DEBUG(logger << "Storing known region: " << regionName << "\n");
  knownRegions[regionName] = &found->getSecond();
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamIfOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ElaboratorImpl::processParamIfOp(ParamIfOp op,
                                                    ImplNode *parent) {
  // Check the condition expression.
  auto errorOrValue =
      parent->evaluator.concretizeParameterExpr(op.getLoc(), op.getCond());
  if (errorOrValue.isError())
    return errorOrValue.takeError();

  // Take whichever branch the condition indicated, and simply inline those ops
  // then elaborate them. We can do this by splicing the op list into the parent
  // block. We splice it this way to avoid remapping the ops when we process
  // them later.
  Region *toProcess = nullptr;
  auto resultInt = cast<IntegerAttr>(cast<Attribute>(errorOrValue.takeValue()));
  // Get the appropriate region.
  if (!resultInt.getValue().isZero())
    toProcess = &op.getThenRegion();
  else
    toProcess = &op.getElseRegion();

  auto foundNestedScope = parent->paramGraph.nestedScopes.find(toProcess);
  if (foundNestedScope == parent->paramGraph.nestedScopes.end())
    return ErrorTree(op.getLoc(), "expected a nested parameter scope");

  ParameterUseDefGraph &uses = foundNestedScope->getSecond();

  LLVM_DEBUG(logger << "Elaborating block:\n";
             toProcess->front().print(logger));

  // Only process the ops in the branch that we ended up taking.
  std::vector<Operation *> opsToRewrite;
  collectOpsToProcess(toProcess, uses, opsToRewrite);
  for (Operation *paramOp : uses.paramOps) {
    // Check if this op is in a region that is a child of the region we care
    // about. If not, don't process it.
    if (!toProcess->isAncestor(paramOp->getParentRegion()))
      continue;

    opsToRewrite.push_back(paramOp);
  }

  SmallVector<Attribute> resultParamValues;
  {
    llvm::SaveAndRestore<IREvaluator> save(parent->evaluator);
    parent->evaluator.clearCache();
    if (failed(processScope(parent, opsToRewrite)))
      return parent->error->copy();
    for (ParamDeclAttr resultParam : op.getResultParams())
      resultParamValues.push_back(
          parent->evaluator.getParameterValues().at(resultParam.getName()));
  }
  for (auto [resultParam, value] :
       llvm::zip(op.getResultParams(), resultParamValues))
    parent->evaluator.setOrOverwriteParameterValue(resultParam, value);

  // Splice the ops into the parent. Grab the terminator before the iterators
  // invalidate.
  Block::iterator iter = op->getIterator();
  Operation *terminator = toProcess->front().getTerminator();
  op->getBlock()->getOperations().splice(iter,
                                         toProcess->front().getOperations());

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
    // Drop all uses of the if op because any of its uses will be null and void
    // at this point.
    op->dropAllDefinedValueUses();
  } else {
    return ErrorTree(terminator->getLoc(),
                     "unknown terminator kind for kgen.param.if");
  }

  // We always erase this op and its nested scopes from the parameter graph -
  // it's been handled, and we don't want anyone else touching it later
  // considering we're about to delete the op itself.
  ParameterUseDefGraph &paramGraph = parent->paramGraph;
  auto eraseIfScopes = [op](ParameterUseDefGraph &graph) mutable {
    // Erase any regions from the nested scopes that belong either to this op or
    // under this op.
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
  op->erase();
  LLVM_DEBUG(
      logger.logOp("param.if parent scope (after processing)", parent->func));
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processScope
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::processScope(ImplNode *parentNode,
                                           ArrayRef<Operation *> worklist) {
  LLVM_DEBUG({
    auto _ = logger.scope("Operations to Rewrite");
    for (Operation *op : worklist)
      logger << *op << "\n";
  });
  TimeTraceScope<EnableTracing> traceScope(
      "processScope", std::to_string(worklist.size()) + " ops");

  // Processing an op may generate more stuff, or even delete the op being
  // processed.
  SmallVector<ParamDeclareRegionOp> declareRegionOps;
  for (auto iter = worklist.begin(), end = worklist.end(); iter != end;
       ++iter) {
    Operation *op = *iter;
    if (!op->getBlock()->isEntryBlock() && op->getBlock()->hasNoPredecessors())
      continue;

    ArrayRef<Operation *> remainingWorklist(iter + 1, end);
    auto _ = logger.scope("Processing: '", op->getName(), "'");
    logger.logOp("Op", op);

    ErrorTreeOrSuccess result = success();
    if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamDeclareOp");
      result = processParamDeclareOp(parentNode->evaluator, declare);
    } else if (auto paramDeclareRegionOp = dyn_cast<ParamDeclareRegionOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamDeclareRegionOp");
      result = processParamDeclareRegionOp(paramDeclareRegionOp, parentNode);
      declareRegionOps.push_back(paramDeclareRegionOp);
    } else if (auto bind = dyn_cast<ParamResultBindOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamResultBindOp");
      result = processParamResultBindOp(bind, parentNode);
    } else if (auto fork = dyn_cast<ParamForkOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamForkOp");
      result = processParamForkOp(parentNode, fork, remainingWorklist);
    } else if (auto rebindOp = dyn_cast<RebindOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processRebindOp");
      result = processRebindOp(parentNode->evaluator, rebindOp);
    } else if (auto assertOp = dyn_cast<ParamAssertOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamAssertOp");
      result = processParamAssertOp(parentNode->evaluator, assertOp);
    } else if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processParamIfOp");
      result = processParamIfOp(ifOp, parentNode);
    } else if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      TimeTraceScope<EnableTracing> traceScope("processCallOp");
      result = processCallOp(call, parentNode, remainingWorklist);
    } else {
      TimeTraceScope<EnableTracing> traceScope("processGenericOp");
      result = processGenericOp(parentNode->evaluator, op);
    }

    // If the parent node was set to error, then just bail.
    if (parentNode->error)
      return failure();

    // If we have an error, log it and set the parent node to an error. This
    // will perform any required cleanup.
    if (result.isError()) {
      LLVM_DEBUG(logger.scope("Result: Failure")
                 << result.getError().getError());
      parentNode->setToError(result.takeError());
      return failure();
    }
  }

  for (ParamDeclareRegionOp paramDeclareRegionOp : declareRegionOps) {
    paramDeclareRegionOp->remove();
    paramDeclareRegionOps.push_back(
        OwningOpRef<ParamDeclareRegionOp>(paramDeclareRegionOp));
  }
  LLVM_DEBUG(parentNode->print(logger << "Completed processing "));
  return success();
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

ErrorTreeOrSuccess ElaboratorImpl::specializeGenerator(ParamNode *genNode) {
  ++depth;
  auto decDepth = llvm::make_scope_exit([&] { --depth; });
  if (depth > config.maxDepth) {
    return ErrorTree(genNode->gen.getLoc(),
                     "elaborator expansion is " + Twine(config.maxDepth + 1) +
                         " levels deep - infinite recursion?");
  }

  // If this generator node is already concrete and has no error, don't
  // re-concretize.
  if (genNode->status == ParamNode::DONE ||
      genNode->status == ParamNode::IN_PROGRESS) {
    if (config.testDiagnostics)
      genNode->gen.emitRemark("Generator has already been specialized");
    return success();
  }

  genNode->status = ParamNode::IN_PROGRESS;
  GeneratorOp generator = genNode->gen;

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
  ParameterUseDefGraph *genNodeGraph;
  auto foundGraph = knownGraphs.find(generator);
  if (foundGraph != knownGraphs.end()) {
    genNodeGraph = foundGraph->getSecond().get();
  } else {
    auto &uses = knownGraphs[generator] =
        std::make_unique<ParameterUseDefGraph>(generator.getBodyRegion());
    uses->calculate(paramCache);
    genNodeGraph = uses.get();
  }

  // Bind all parameter values in this scope.
  ArrayRef<Attribute> inputParamValues = genNode->inputParams.getValue();
  auto inputParamDecls = generator.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  IREvaluator evaluator(*this);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setOrOverwriteParameterValue(decl, val);

  // If the generator's constraints don't satisfy, set an error and move on.
  if (auto err =
          KGEN::evaluateConstraints(generator.getConstraints(), evaluator))
    return std::move(err.value());

  StringAttr mangledName = mangleParameterValues(generator, inputParamValues);

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(generator);
  auto newFunc = b.create<FuncOp>(
      generator.getLoc(), mangledName,
      SignatureType::get(TypeArrayAttr::get(generator.getContext(), {}),
                         TypeArrayAttr::get(generator.getContext(), {}),
                         generator.getFunctionType(), generator.getMetadata()),
      generator.getAlwaysInlineLevel());

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  analysis.getTopLevelSymbolTable().insert(newFunc, generator->getIterator());

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
      newFunc, genNode, std::move(evaluator), std::move(childGraph));
  g.concreteNodes.try_emplace(newFunc, childNode.get());
  ImplNode *newFuncNode = childNode.get();
  genNode->impls.push_back(std::move(childNode));
  ParameterUseDefGraph &uses = newFuncNode->paramGraph;

  // Kick off the expansion for the new function.

  auto funcScope = logger.scope("Specializing Function: @", newFunc.getName());
  logger.logOp("Function", newFunc);

  std::vector<Operation *> opsToRewrite;
  collectOpsToProcess(&newFunc.getBodyRegion(), uses, opsToRewrite);
  opsToRewrite.push_back(newFunc);
  llvm::append_range(opsToRewrite, uses.paramOps);

  // Process the worklist. Only finalize the function if this succeeded.
  newFuncNode->status = ImplNode::IN_PROGRESS;
  if (succeeded(processScope(newFuncNode, opsToRewrite)))
    finalizeAndVerifyFunction(analysis, newFuncNode);
  newFuncNode->status = ImplNode::DONE;

  genNode->status = ParamNode::DONE;
  ErrorTree err(genNode->gen.getLoc(), "no viable expansions found");
  for (ImplNode &impl : llvm::make_pointee_range(genNode->impls)) {
    if (!impl.error)
      return success();
    err.addCause(impl.error->copy());
  }
  return std::move(err);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::run
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::run(ArrayRef<GeneratorOp> primaryGenerators) {
  LLVM_DEBUG(logger << "Starting Elaboration\n");

  ModuleOp theModule = analysis.getTopLevelOp<ModuleOp>();
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
    ParamNode *generatorNode = g.getOrCreate(emptyInputParamKey, gen);

    // Now we can begin to construct the expansion tree rooted at this
    // generator. Emit as many errors as possible.
    if (ErrorTreeOrSuccess err = specializeGenerator(generatorNode);
        err.isError()) {
      err.takeError().emit([](Location loc) { return mlir::emitError(loc); });
      failed = true;
    }
  }
  if (failed)
    return failure();

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  DenseMap<StringAttr, StringAttr> funcsToRename;
  // We have to walk all the generators in the module because we have to rename
  // each one to its first implementation.
  SymbolTable &symtab = analysis.getTopLevelSymbolTable();
  for (auto gen : llvm::make_early_inc_range(theModule.getOps<GeneratorOp>())) {
    ParamNode *genNode = g.getOrCreate(emptyInputParamKey, gen);

    // Add all concrete functions, and rename the first one.
    std::vector<FuncOp> concreteFuncs;
    genNode->getAllConcreteFuncs(concreteFuncs);
    for (FuncOp c : concreteFuncs)
      analysis.getTopLevelSymbolTable().insert(c, genNode->gen->getIterator());

    if (!concreteFuncs.empty())
      funcsToRename[concreteFuncs.front().getNameAttr()] =
          genNode->gen.getSymNameAttr();
    symtab.erase(gen);
  }

  for (ParamNode &node :
       llvm::make_pointee_range(llvm::make_second_range(g.nodes))) {
    for (ImplNode &impl : llvm::make_pointee_range(node.impls))
      if (impl.error)
        symtab.erase(impl.func);
  }

  // Perform any renaming at the end.  We cannot use the
  // SymbolTable::replaceAllSymbolUses method, because it doesn't tolerate
  // unregistered operations.  It also doesn't support batch renaming.
  theModule->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    // If this op is a ParamDeclareRegionOp, delete it. It must be fully handled
    // at this phase.
    if (auto regionParam = dyn_cast<ParamDeclareRegionOp>(op)) {
      regionParam.erase();
      return WalkResult::skip();
    }

    // If this is a func being renamed, rename it.
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (auto newName = funcsToRename.lookup(func.getNameAttr())) {
        // Keep the symbol table up-to-date with the new name.
        symtab.remove(func);
        func.setSymNameAttr(newName);
        symtab.insert(func, op->getIterator());
      }
      return WalkResult::advance();
    }

    // If this is a reference to a function that got renamed, update its
    // target.
    TypeSwitch<Operation *>(op).Case([&](KGENCallOpInterface call) {
      auto callee = cast<SymbolConstantAttr>(call.getCallee());
      auto newName = funcsToRename.lookup(
          cast<FlatSymbolRefAttr>(callee.getSymbol()).getAttr());
      if (newName)
        call.updateCallee(SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(newName), callee.getType()));
    });
    return WalkResult::advance();
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
  ElaboratorImpl impl(symtab, paramCache, target, evaluatorExecutorFn, config);
  return impl.run(primaryGenerators);
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
      evaluatorExecutorFn =
          [](KGEN::FuncOp evaluator, SymbolTable &symtab, TargetInfoAttr target,
             ArrayRef<KGEN::FuncOp> specializations) { return 0; };
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
