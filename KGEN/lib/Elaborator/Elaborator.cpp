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

#include "Elaborator.h"
#include "IREvaluator.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/Package/Package.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Support/ForkJoin.h"
#include "LLCL/Support/UnknownLocationDecoder.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/STLExtras.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "kgen-elaborator"

using namespace M;
using namespace KGEN;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Elaborator
//===----------------------------------------------------------------------===//

static ModuleOp cloneAsEmpty(const SymbolTable &symtab) {
  ModuleOp newModule = cast<ModuleOp>(symtab.getOp()).cloneWithoutRegions();
  newModule.getBodyRegion().push_back(new Block);
  return newModule;
}

Elaborator::Elaborator(SymbolTable &oldSymTab, TargetInfoAttr target,
                       const ElaborateGeneratorsOptions &config)
    : target(target), config(config), newModule(cloneAsEmpty(oldSymTab)),
      newSymTab(SymbolTable(*newModule)), oldSymTab(oldSymTab),
      env(oldSymTab.getOp()->getAttrOfType<EnvAttr>(
          EnvAttr::getEnvAttrName())) {
  assert(env && "expected an environment attribute");
}

FuncOp Elaborator::lookupConcreteFunction(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  return newSymTab.read([name](const SymbolTable &symtab) {
    return symtab.lookup<FuncOp>(name);
  });
}

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

std::string KGEN::mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<TypedAttr> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return generator.getName().str();

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getInputParamsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues))
    os << ',' << inputDecl.getName().str() << '=' << getParamAsString(value);
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

ImplNode::ImplNode(ParamNode *parent)
    : parent(parent), paramGraph(parent->gen.getBodyRegion()),
      evaluator(parent->gen.getContext()) {}

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
    for (const auto &[_, bind] : bindings.get()) {
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

void ParamNode::andThenAsync(AsyncValue::Waiter &&waiter) {
  expansionGraph->didAddTask();
  paramCh.andThenAsync([waiter = std::move(waiter), this]() mutable {
    waiter();
    expansionGraph->didCompleteTask();
  });
}

void ParamNode::andThenSync(AsyncValue::Waiter &&waiter) {
  paramCh.andThenSync(std::forward<AsyncValue::Waiter>(waiter));
}

void ParamNode::emplace() {
  std::lock_guard<std::mutex> guard(mu);
  if (!isReady())
    paramCh.copy().emplace();
}

AsyncValueRef<Chain> ParamNode::copy() const { return paramCh.copy(); }

void ParamNode::setToError() {
  std::lock_guard<std::mutex> guard(mu);
  isError = true;
  if (!isReady())
    paramCh.copy().emplace();
}

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

ExpansionGraph::~ExpansionGraph() {
  {
    std::lock_guard<std::mutex> guard(quiesceMu);
    // If we have outstanding tasks at destruction time, construct the chain to
    // trigger waiter completion.
    if (numOutstandingResources > 0) {
      for (auto &[key, node] : nodes.get())
        node->setToError();
    }
  }
  LLCL::await(quiesce());
}

ParamNode *ExpansionGraph::getOrCreate(LLCL::Runtime &runtime,
                                       ParameterExprArrayAttr values,
                                       GeneratorOp gen, size_t depth) {
  // TODO: Split this into `get` and `create` methods, so that some can be
  // read-only accesses.
  return nodes.modify([&](auto &map) {
    std::unique_ptr<ParamNode> &n = map[{values, gen}];
    if (!n)
      n = std::make_unique<ParamNode>(runtime, gen, values, depth, this);
    return n.get();
  });
}

void ExpansionGraph::didCompleteTask() {
  std::lock_guard<std::mutex> guard(quiesceMu);
  assert(numOutstandingResources > 0 &&
         "mismatched didAddTask/didCompleteTask calls");
  if (--numOutstandingResources == 0 && quiesceChain)
    std::move(quiesceChain).emplace();
}

void ExpansionGraph::didAddTask() {
  std::lock_guard<std::mutex> guard(quiesceMu);
  assert(!quiesceChain && "cannot create new task using a "
                          "runtime which is being quiesced");
  ++numOutstandingResources;
}

AsyncValueRef<Chain> ExpansionGraph::quiesce() {
  std::lock_guard<std::mutex> guard(quiesceMu);
  assert(!quiesceChain && "already waiting for ParamNodeRuntime to quiesce");
  quiesceChain =
      numOutstandingResources == 0
          ? AsyncValueRef<Chain>::createReady(worklistCh.getRuntime())
          : AsyncValueRef<Chain>::allocate(worklistCh.getRuntime());
  return quiesceChain.copy();
}

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

ErrorTreeOrSuccess ParamNode::collectErrorsOrSuccess() {
  ErrorTree err(gen.getLoc(), "no viable expansions found");
  for (ImplNode &impl : llvm::make_pointee_range(impls)) {
    if (!impl.error)
      return success();
    err.addCause(impl.error->copy());
  }
  return std::move(err);
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
} // namespace

#define HANDLE_EVALUATOR_CONC(VAR, INODE, LOC, EXPR)                           \
  if (auto exprResult =                                                        \
          (INODE)->getEvaluator().concretizeParameterExpr(INODE, LOC, EXPR);   \
      exprResult.isError()) {                                                  \
    (INODE)->setToError(exprResult.takeError());                               \
    return ElaborationState::error();                                          \
  } else if (!*exprResult) {                                                   \
    return ElaborationState::skipNode();                                       \
  } else {                                                                     \
    VAR = *exprResult;                                                         \
  }

//===----------------------------------------------------------------------===//
// processParamDeclareOp
//===----------------------------------------------------------------------===//

/// Process a param.declare op by setting its parameter value in the provided
/// evaluator.
static ElaborationState processParamDeclareOp(ImplNode *inode,
                                              ParamDeclareOp op) {
  // Simplify the input expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getValue());

  // Bind it to the parameter declaration it is setting.
  inode->getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(), value);

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processParamResultBindOp
//===----------------------------------------------------------------------===//

/// Process a `kgen.param.result_bind` operation by setting the result parameter
/// values of the parent operation.
static ElaborationState processParamResultBindOp(ImplNode *node,
                                                 ParamResultBindOp op) {
  // Concretize the result parameter values.
  IREvaluator &evaluator = node->getEvaluator();
  SmallVector<TypedAttr> resultParams;

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
    Attribute resultValue;
    HANDLE_EVALUATOR_CONC(resultValue, node, op.getLoc(), value);
    resultParams.push_back(cast<TypedAttr>(resultValue));
    evaluator.setOrOverwriteParameterValue(decl, resultParams.back());
  }

  // If this operation binds values for the result parameters of the generator,
  // set them in the node.
  if (isFunc)
    node->resultParams =
        ParameterExprArrayAttr::get(op.getContext(), resultParams);

  op.erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processRebindOp
//===----------------------------------------------------------------------===//

static ElaborationState processRebindOp(ImplNode *inode, RebindOp op) {
  Type outType;
  HANDLE_EVALUATOR_CONC(outType, inode, op.getLoc(), op.getType());
  Type inType;
  HANDLE_EVALUATOR_CONC(inType, inode, op.getLoc(), op.getInput().getType());
  if (outType != inType) {
    inode->setToError(ErrorTree(
        op.getLoc(), "error: rebind input type '" + mlir::debugString(inType) +
                         "' does not match result type '" +
                         mlir::debugString(outType) + "'"));
    return failure();
  }
  op.replaceAllUsesWith(op.getOperand());
  op.erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processParamAssertOp
//===----------------------------------------------------------------------===//

/// Process a param.assert op by folding its parameter expression and checking
/// its constraint. Returns the appropriate error if the constraint failed.
static ElaborationState processParamAssertOp(ImplNode *inode,
                                             ParamAssertOp op) {
  // Check the condition expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getCond());

  // If the constraint evaluated to zero then the assert fails.
  auto resultInt = cast<IntegerAttr>(value);
  if (resultInt.getValue().isZero()) {
    // Evaluate the string to report it.
    HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getMessage());
    inode->setToError(
        ErrorTree(op.getLoc(),
                  "constraint failed: " + cast<StringAttr>(value).getValue()));
    return failure();
  }

  // The kgen.param.assert op serves no further purpose, so we can remove it.
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processGenericOp
//===----------------------------------------------------------------------===//

/// Unknown operations are allowed to use types and attributes with parameter
/// references. Substitute in concrete values for their references. Optionally
/// elaborate their locations.
static ElaborationState processGenericOp(ImplNode *parent, Operation *op) {
  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    Attribute value;
    HANDLE_EVALUATOR_CONC(value, parent, op->getLoc(), namedAttr.getValue());
    newAttrs.emplace_back(namedAttr.getName(), value);
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults()) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, parent, op->getLoc(), result.getType());
    result.setType(type);
  }

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Value arg : block.getArguments()) {
        Type type;
        HANDLE_EVALUATOR_CONC(type, parent, op->getLoc(), arg.getType());
        arg.setType(type);
      }
    }
  }

  return ElaborationState::advance();
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

namespace {

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
                 TargetInfoAttr target, ElaboratorCallbacks callbacks,
                 LLCL::Runtime &runtime,
                 const ElaborateGeneratorsOptions &config)
      : Elaborator(symtab, target, config), g(runtime),
        paramCache(paramCache, runtime.getWorkQueue()->getParallelismLevel()),
        callbacks(std::move(callbacks)), runtime(runtime) {}

  std::optional<ErrorTreeOr<FuncOp>>
  getConcreteFunction(ImplNode *parent, Location loc,
                      FlatSymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) override;

  std::optional<ErrorTreeOrSuccess> getAllConcreteFunctions(
      ImplNode *parent, Location loc, FlatSymbolRefAttr symbolRef,
      ArrayRef<TypedAttr> paramValues, std::vector<FuncOp> &funcs) override;

  ElaboratorCompileAsmFnRef
  getCompileAsmFn(Elaborator::ASMFormat format) const override {
    return callbacks.compileAsmFn;
  }

  void addDeferredFunction(OwningOpRef<FuncOp> func) override;

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ModuleOp theModule,
                    ArrayRef<GeneratorOp> primaryGenerators);

private:
  /// Fork the expansion of a concrete node.
  ImplNode *fork(ImplNode *cur, IRMapping &map, StringRef forkParam,
                 Attribute value);

  /// Insert an ImplNode for a function that is already concrete.
  void addConcreteFunc(FuncOp func) {
    g.concreteNodes.modify([this, func](auto &map) mutable {
      auto node = std::make_unique<ImplNode>(
          func, /*parent=*/nullptr, func.getBodyRegion(),
          func.getSymName().str(), IREvaluator(*this));
      map.try_emplace(func, node.get());
      g.elaboratedNodes.push_back(std::move(node));
    });
  }

  /// Implement the evaluator hook. This function ensures that all active work
  /// items on the workqueue are completed or suspended before running the
  /// evaluator, to ensure that, at least with respect to this compiler
  /// instance, the machine is quiet.
  ErrorOrSuccess evaluateFunctions(ImplNode *inode, FuncOp evaluator,
                                   std::vector<FuncOp> options);

  /// Once a concrete function has finished specializing, finish processing the
  /// function and call the verifier.
  void finalizeFunction(ImplNode *node);

  /// Process a kgen.param.fork op. This will create a clone for each value of
  /// the parameter search, and will mark the parent as an error. This results
  /// in a very clean model where the parent of the current parent (a generator)
  /// will have its children be the successfully concretized parameter search
  /// nodes.
  ElaborationState processParamForkOp(ImplNode *parent, ParamForkOp op);

  /// Parameter constants are the only operation that can bridge the parameter
  /// domain into the runtime domain. Use it to root concretization of nested
  /// symbol references.
  template <typename OpT>
  ElaborationState processParamConstantOp(ImplNode *parent, OpT op);

  /// Spawn a clone for kgen.param.fork. This creates a new FuncOp that is a
  /// sibling to the parent of the kgen.param.fork op. It replaces the
  /// kgen.param.fork with a param.declare to allow specialization to succeed.
  void spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                           ImplNode *forkParentNode);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  ElaborationState processCallOp(ImplNode *parent,
                                 GeneratorUserOpInterface call);

  /// Process an evaluate operation by concretizing the evaluator function and
  /// the function candidates.
  ElaborationState processEvaluateOp(ImplNode *parent, ParamEvaluateOp op);

  /// Instantiate a generator reference by retrieving the concrete
  /// implementations of a reference. If this function returns `advance` but
  /// `inputParamKey` is not populated, then the callee is a direct function
  /// reference.
  ElaborationState instantiateGeneratorReference(
      ImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
      ParameterExprArrayAttr &inputParamKey, GeneratorOp &gen,
      std::vector<ImplNode *> &concrete,
      function_ref<bool(ParamNode *)> shouldWait = [](ParamNode *) {
        return true;
      });

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  ElaborationState processGeneratorUser(GeneratorUserOpInterface user,
                                        SymbolConstantAttr calleeSymbol,
                                        ImplNode *parent);

  /// Complete processing of a generator user by resolving any bound result
  /// types or parameters in the parent scope. This is the step that propagates
  /// result parameters from the inner scope to the outer scope.
  ///
  /// See function definition for the meaning of `invertLockOrder`.
  ElaborationState completeCallProcessing(GeneratorUserOpInterface user,
                                          ArrayRef<ParamDeclAttr> decls,
                                          ImplNode *thisNode, ImplNode *node,
                                          bool invertLockOrder = false);

  /// Complete generator user processing with a list of valid concrete
  /// implementations with consistent bindings. Multi-version the currente node
  /// if required.
  ElaborationState completeGeneratorUserProcessing(
      GeneratorUserOpInterface user, ArrayRef<ParamDeclAttr> decls,
      ImplNode *parent, ParameterExprArrayAttr inputParamKey, GeneratorOp gen,
      ArrayRef<ImplNode *> concrete);

  /// Given a user of a completed parameter node, collect concrete
  /// implementations whose bindings are consistent with the current node.
  LogicalResult
  collectConcreteImplementations(Operation *user, ImplNode *parent,
                                 ParamNode *calleeNode,
                                 std::vector<ImplNode *> &concrete);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  ElaborationState processParamIfOp(ImplNode *parent, ParamIfOp op);

  /// Schedule an implementation node on the LLCL work queue and increment the
  /// initial counters.
  void initialScheduleImplNode(ImplNode *inode) {
    ++inode->parent->numActive;
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(inode);
  }
  /// Signal the worklist to tell it a job has completed or has been taken off
  /// the workqueue.
  void signalWorklist() {
    if (g.numWorkItems.fetch_sub(1) == 1)
      g.worklistCh.copy().emplace();
  }
  /// Schedule an implementation node on the LLCL work queue.
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

  /// Request specialization of the generator at `genNode`. If the node is ready
  /// complete, then the function returns `advance` and the concrete functions
  /// can be retrieved from the node. Otherwise, the function returns
  /// `skipNode`, indicating that elaboration of the current function should be
  /// suspended.
  ElaborationState specializeGenerator(ImplNode *inode, ParamNode *genNode,
                                       ParamNode *from, bool addWaiter);

  /// Attempt to fulfill specialization of a generator by looking for an
  /// implementation in a precompiled module.
  void specializeFromPackage(ImplNode *parent, ParamNode *genNode,
                             FlatSymbolRefAttr linkRef, bool addWaiter);

  /// Specialize a generator by scheduling an elaboration task.
  void specializeFromSource(ImplNode *inode, ParamNode *genNode,
                            bool addWaiter);

  /// Process all deferred search functions serially, re-queuing nodes as it
  /// completes. Returns true if there were deferred search functions.
  bool processDeferredSearchFns();

  /// Attempt to diagnose concrete recursion and break recursion where possible.
  /// Return true if recursion was broken at least once. The "generation" is
  /// used to know whether a visited flag is valid. It must be at least 1,
  /// because the initial generation is always 0.
  bool diagnoseAndBreakRecursion(unsigned generation,
                                 ArrayRef<ParamNode *> roots);

  /// A logger used to emit information during the elaboration process.
  Logger logger;

  /// Hash table of known ParameterUseDefGraphs. This ensures we only compute a
  /// graph once for each generator. This is extra state generated by
  /// specializeGenerator that is *required for correctness* - this will cause
  /// issues with caching (though it would be easy to simply recompute) unless
  /// we create a ParametricNode or something we can use to store these in a
  /// proper data structure.
  Shared<DenseMap<GeneratorOp, std::unique_ptr<ParameterUseDefGraph>>>
      knownGraphs;

  /// The callgraph being expanded.
  ExpansionGraph g;

  /// The mutex to use when verifying elaborated function candidates.
  llvm::sys::SmartRWMutex<true> verifyMutex;

  /// This is the cached parameter collector analysis.
  ThreadLocalCache<ParameterCollector::Analysis> paramCache;

  /// Callbacks to use for JIT functionalities.
  ElaboratorCallbacks callbacks;

  /// This struct contains information about a deferred search job.
  struct DeferredSearch {
    /// Deferred search functor.
    ElaboratorSearchFn searchFn;
    /// The node that was suspended.
    ImplNode *inode;
    /// The candidates.
    std::vector<FuncOp> candidates;
  };
  /// The deferred search jobs.
  Shared<std::vector<DeferredSearch>> deferredSearchFns;

  /// The LLCL runtime instance to use.
  LLCL::Runtime &runtime;

  /// Deferred generated symbols to append to the module.
  SmallVector<mlir::SymbolOpInterface> deferredSymbols;

  /// State for a package reader.
  struct PackageReaderState {
    PackageReaderState(MLIRContext *ctx, mlir::AsmResourceBlob *blob);
    ~PackageReaderState() {
      if (failed(reader.finalize()))
        llvm::report_fatal_error("failed to finalize bytecode reader");
    }

    LogicalResult initialize();

    std::shared_ptr<llvm::SourceMgr> sourceMgr;
    mlir::ParserConfig config;
    mlir::BytecodeReader reader;
    Block block;
    ModuleOp module;
    std::optional<SymbolTable> symtab;
    llvm::sys::SmartRWMutex<true> mutex;
  };

  /// This struct represents the state of a package being elaborated.
  struct PackageState {
    /// Completion chain of the package state.
    LLCL::AsyncValueRef<LLCL::Chain> ch;

    /// The package link op.
    PackageLinkOp link;

    /// An error if the package encountered one during loading.
    std::optional<Error> error;

    /// The package reader state when the package is ready. If this is nullopt
    /// when the chain is set, that means no precompiled symbols are
    /// available.
    std::unique_ptr<PackageReaderState> reader;
  };

  /// States for all packages encountered during elaboration.
  Shared<DenseMap<StringAttr, std::unique_ptr<PackageState>>> packages;
};
} // namespace

//===----------------------------------------------------------------------===//
// ElaboratorImpl::fork
//===----------------------------------------------------------------------===//

ImplNode *ElaboratorImpl::fork(ImplNode *cur, IRMapping &map,
                               StringRef forkParam, Attribute value) {
  // Clone the function and generate a unique name for it.
  auto clone = cast<FuncOp>(cur->func->clone(map));
  std::string name = cur->baseName;
  llvm::raw_string_ostream os(name);
  os << ',';
  if (!forkParam.empty())
    os << forkParam << '=' << getParamAsString(value);
  else
    os << '@' << cast<StringAttr>(value).getValue();
  clone.setSymName(StringAttr::get(value.getContext(), name));

  // Update the subprogram information.
  if (auto scope = clone.getSubprogramScope()) {
    DebugInfo::SourceNameAttr name = scope.getName();
    SmallVector<StringAttr> values = llvm::to_vector(name.getParamValues());
    if (!forkParam.empty())
      values.push_back(getParamTypeAsString(cast<TypedAttr>(value)));
    else
      values.push_back(cast<StringAttr>(value));
    name = DebugInfo::SourceNameAttr::get(name.getName(), name.getParamTypes(),
                                          name.getArgTypes(), values,
                                          name.getParent(), name.getKind());
    DebugInfo::updateSubprogram(clone, clone.getSymNameAttr(), name);
  }

  // Insert the new function at a location relative to the current one. This
  // ensures all forks are inserted in a deterministic order, regardless of
  // which occur first.
  newSymTab.modify([clone, func = cur->func](SymbolTable &symtab) {
    symtab.insert(clone, std::next(func->getIterator()));
  });

  // Fork the node and its bindings.
  auto n =
      std::make_unique<ImplNode>(clone, cur->parent, cur->paramGraph.copy(map),
                                 std::move(name), cur->evaluator);
  n->bindings.get() = cur->bindings.read([](auto &map) { return map; });

  // Copy over the current work stack.
  for (const ImplNode::WorkItem &item : cur->stack) {
    std::vector<Operation *> clonedOps;
    for (Operation *op : item.ops)
      clonedOps.push_back(map.lookup(op));
    n->stack.push_back(
        ImplNode::WorkItem{std::move(clonedOps), item.onComplete});
  }

  // Track the new node as a new child and concrete node.
  ImplNode *result = n.get();
  ParamNode *p = cur->parent;
  {
    // Multiple forks can happen at the same time.
    llvm::sys::SmartScopedWriter<true> guard(p->implsMutex);
    p->impls.push_back(std::move(n));
  }
  g.concreteNodes.modify([clone, result](DenseMap<FuncOp, ImplNode *> &map) {
    map.try_emplace(clone, result);
  });

  // Clone the current dependencies too. Set the number of dependencies to the
  // current number of waiting callee nodes, plus 1 to gate.
  result->numDependencies = 1;
  for (auto [call, genNode] : cur->dependencies) {
    result->dependencies.emplace_back(
        cast<GeneratorUserOpInterface>(map.lookup(&*call)), genNode);
    // Add a new dependent on the callee node. If it is already complete, it
    // will immediately decrement `numDependencies`.
    if (genNode->state.addWaiter()) {
      ++result->numDependencies;
      genNode->andThenAsync(
          [this, inode = result] { completeImplNodeProcessing(inode); });
    }
  }
  assert(result->numDependencies >= 1 && "fork could not have completed");

  return result;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::finalizeFunction
//===----------------------------------------------------------------------===//

void ElaboratorImpl::finalizeFunction(ImplNode *node) {
  CompilerTimeTraceScope traceScope("finalizeFunction");
  // Erase everything but the entry blocks of each region.
  FuncOp func = node->func;
  func.walk<mlir::WalkOrder::PreOrder>([](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(region)))
        block.erase();
  });
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getConcreteFunction
//===----------------------------------------------------------------------===//

std::optional<ErrorTreeOr<FuncOp>>
ElaboratorImpl::getConcreteFunction(ImplNode *parent, Location loc,
                                    FlatSymbolRefAttr symbolRef,
                                    ArrayRef<TypedAttr> paramValues) {
  StringAttr name = symbolRef.getAttr();
  Operation *gen = oldSymTab.lookup(name);
  // If this doesn't reference anything in the existing module, then it must
  // refer to a concrete function in the new module.
  if (!gen) {
    FuncOp concrete = newSymTab.read([name](const SymbolTable &symtab) {
      return symtab.lookup<FuncOp>(name);
    });
    assert(concrete && "expected to find a concrete function");
    return concrete;
  }

  auto vals = ParameterExprArrayAttr::get(symbolRef.getContext(), paramValues);

  // Lookup the node if it already exists.
  ParamNode *node =
      g.getOrCreate(runtime, vals, cast<GeneratorOp>(gen), /*depth=*/0);
  // If the node has already been elaborated, just use that result.
  ElaborationState result =
      specializeGenerator(parent, node, /*from=*/nullptr, /*addWaiter=*/true);
  if (result.shouldSkipNode())
    return std::nullopt;
  return node->getFirstConcreteFunc();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getAllConcreteFunctions
//===----------------------------------------------------------------------===//

std::optional<ErrorTreeOrSuccess> ElaboratorImpl::getAllConcreteFunctions(
    ImplNode *parent, Location loc, FlatSymbolRefAttr symbolRef,
    ArrayRef<TypedAttr> paramValues, std::vector<FuncOp> &funcs) {
  GeneratorOp gen = oldSymTab.lookup<GeneratorOp>(symbolRef.getAttr());

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(
      runtime, ParameterExprArrayAttr::get(loc.getContext(), paramValues), gen,
      /*depth=*/0);
  ElaborationState result =
      specializeGenerator(parent, node, /*from=*/nullptr, /*addWaiter=*/true);
  if (result.shouldSkipNode())
    return std::nullopt;
  node->getAllConcreteFuncs(funcs);
  if (funcs.empty())
    return node->collectErrorsOrSuccess();
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::addDeferredFunction
//===----------------------------------------------------------------------===//

void ElaboratorImpl::addDeferredFunction(OwningOpRef<FuncOp> func) {
  FuncOp op = func.release();
  addConcreteFunc(op);
  newSymTab.modify([this, op](SymbolTable &symtab) {
    symtab.insert(op);
    deferredSymbols.push_back(op);
  });
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::evaluateFunctions
//===----------------------------------------------------------------------===//

ErrorOrSuccess ElaboratorImpl::evaluateFunctions(ImplNode *inode,
                                                 FuncOp evaluator,
                                                 std::vector<FuncOp> options) {
  CompilerTimeTraceScope traceScope("evaluateFunctions", [evaluator, options] {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "evaluator: " << FuncOp(evaluator).getSymName() << "\n";
    for (FuncOp opt : options)
      os << " - " << opt.getSymName();
    return os.str();
  });

  // Cheeky copy. The state of the symbol right at this moment is sufficient to
  // produce a standalone object for the functions being JIT'd.
  SymbolTable symtabCopy = newSymTab.read(
      [](const SymbolTable &symtab) -> SymbolTable { return symtab; });
  ErrorOr<ElaboratorSearchFn> searchFn =
      callbacks.evaluateFn(evaluator, symtabCopy, getTarget(), options);
  if (searchFn.isError())
    return searchFn.takeError();
  // Suspend elaboration. The search has to be performed in isolation.
  deferredSearchFns.modify([inode, fn = searchFn.takeValue(),
                            candidates =
                                std::move(options)](auto &fns) mutable {
    fns.push_back(DeferredSearch{std::move(fn), inode, std::move(candidates)});
  });
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamForkOp
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::processParamForkOp(ImplNode *parent,
                                                    ParamForkOp op) {
  auto _ = logger.scope("Processing ParamForkOp");
  LLVM_DEBUG(logger.scope("Options") << op.getValuesAttr() << "\n");

  Attribute value;
  HANDLE_EVALUATOR_CONC(value, parent, op.getLoc(), op.getValuesAttr());

  auto forkValuesAttr = cast<VariadicAttr>(value);

  if (forkValuesAttr.getValues().empty()) {
    parent->setToError(ErrorTree(op.getLoc(), "no candidates found"));
    return failure();
  }

  // Loop over all the possible candidates that we will search over, spawning
  // N possibilities to explore.
  SmallVector<ErrorTree> errors;
  DenseSet<Attribute> seenValues;
  seenValues.reserve(forkValuesAttr.getValues().size());
  for (Attribute value : forkValuesAttr.getValues().drop_front()) {
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
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamConstantOp
//===----------------------------------------------------------------------===//

template <typename OpT>
ElaborationState ElaboratorImpl::processParamConstantOp(ImplNode *parent,
                                                        OpT op) {
  Attribute attr;
  HANDLE_EVALUATOR_CONC(attr, parent, op->getLoc(), op.getValue());
  auto value = cast<TypedAttr>(attr);

  // Root elaboration at the constant value and concretize any generator
  // references inside it. Multi-versioning is disallowed.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](SymbolConstantAttr cst) -> std::pair<Attribute, WalkResult> {
        // Ignore parametric constants.
        if (!cst.getType().getInputParamTypes().empty())
          return {cst, WalkResult::advance()};
        std::optional<ErrorTreeOr<FuncOp>> func = getConcreteFunction(
            parent, op.getLoc(), cast<FlatSymbolRefAttr>(cst.getSymbol()),
            cst.getParamValues());
        if (!func) {
          return {cst, WalkResult::interrupt()};
        }
        if (func->isError()) {
          parent->setToError(func->takeError());
          return {cst, WalkResult::interrupt()};
        }

        return {SymbolConstantAttr::get(
                    FlatSymbolRefAttr::get(func->takeValue().getSymNameAttr()),
                    cst.getType()),
                WalkResult::advance()};
      });
  value = cast_or_null<TypedAttr>(replacer.replace(value));
  if (parent->error)
    return ElaborationState::error();
  if (!value)
    return ElaborationState::skipNode();

  op.getResult().setType(value.getType());
  op.setValueAttr(value);
  return ElaborationState::advance();
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
  ImplNode *newFuncNode =
      fork(forkParentNode, map, forkOp.getParamDecl().getName(), value);

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
  assert(forkParentNode->parent->numActive != 0 &&
         "forking a completed parameter node?");
  initialScheduleImplNode(newFuncNode);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::instantiateGeneratorReference
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::instantiateGeneratorReference(
    ImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
    ParameterExprArrayAttr &inputParamKey, GeneratorOp &gen,
    std::vector<ImplNode *> &concrete,
    function_ref<bool(ParamNode *)> shouldWait) {
  // Lookup the callee.
  StringAttr name = cast<FlatSymbolRefAttr>(calleeSymbol.getSymbol()).getAttr();
  Operation *calleeOp = oldSymTab.lookup(name);

  if (!calleeOp) {
    auto func = newSymTab.read([name](const SymbolTable &symtab) {
      return symtab.lookup<FuncOp>(name);
    });
    assert(func && "could not find referenced generator, invalid IR?");
    ImplNode *node =
        g.concreteNodes.read([func](const DenseMap<FuncOp, ImplNode *> &map) {
          return map.lookup(func);
        });
    assert(node && "concrete callee is missing a node?");
    concrete.push_back(node);
    return ElaborationState::advance();
  }

  // Add in the mapping for parameters in the calls.
  {
    LLVM_DEBUG(logger.logOp("Resolving Call Input Param", user);
               logger << " with input bindings: ";
               llvm::interleaveComma(calleeSymbol.getParamValues(), logger);
               logger << "\n");
    inputParamKey = ParameterExprArrayAttr::get(user->getContext(),
                                                calleeSymbol.getParamValues());
  }

  LLVM_DEBUG({
    logger.logOp("Callee", calleeOp);
    logger << "Input Params: " << inputParamKey << "\n";
  });

  // If we already have a binding for this, we're done.
  gen = cast<GeneratorOp>(calleeOp);
  if (ImplNode *existing =
          parent->bindings.read([inputParamKey, gen](auto &map) {
            return map.lookup({inputParamKey, gen});
          })) {
    LLVM_DEBUG(existing->print(logger.scope("Result: Existing Binding")));
    concrete.push_back(existing);
    return ElaborationState::advance();
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
  ElaborationState result = specializeGenerator(
      parent, calleeNode, parent->parent, shouldWait(calleeNode));
  if (result.shouldSkipNode())
    return ElaborationState::skipNode();

  return collectConcreteImplementations(user, parent, calleeNode, concrete);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::collectConcreteImplementations
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::collectConcreteImplementations(
    Operation *user, ImplNode *parent, ParamNode *calleeNode,
    std::vector<ImplNode *> &concrete) {
  // Get all valid implementations of the callee node.
  calleeNode->getAllConcreteNodes(concrete);

  // If the concrete thing has bindings, they must be consistent with the
  // parent's bindings for us to consider it. Remove nodes from the vector that
  // have bindings that are inconsistent with the parent.
  //
  // NOTE: Concurrent access here is very unlikely. It can only happen after
  // breaking recursion. Use coarse-grained locking for simplicity.
  parent->bindings.read([&](auto &parentBindings) {
    auto newEnd = llvm::remove_if(concrete, [&](ImplNode *node) {
      return node->bindings.read([&](auto &nodeBindings) {
        bool hasConsistentBindings = llvm::all_of(nodeBindings, [&](auto pair) {
          // The binding is only inconsistent if it (a) does exist and (b)
          // is different.
          if (ImplNode *found = parentBindings.lookup(pair.first))
            return found == pair.second;
          // Otherwise, we're good.
          return true;
        });
        // If it has inconsistent bindings
        if (!hasConsistentBindings && !nodeBindings.empty() &&
            !parentBindings.empty()) {
          LLVM_DEBUG(logger << "Removing node for inconsistent bindings: ";
                     node->print(logger));
          return true;
        }
        return false;
      });
    });
    concrete.erase(newEnd, concrete.end());
  });

  // If there are no implementations, return the callee's errors.
  if (concrete.empty()) {
    ErrorTree out(user->getLoc(),
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
  return success();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processGeneratorUser
//===----------------------------------------------------------------------===//

ElaborationState
ElaboratorImpl::processGeneratorUser(GeneratorUserOpInterface user,
                                     SymbolConstantAttr calleeSymbol,
                                     ImplNode *parent) {
  auto _ = logger.scope("Processing Generator User");
  LLVM_DEBUG(logger.logOp("User", user));

  // Not all operations can verify their callee type, if for instance, it is a
  // generic type. Verify here as a fallback.
  if (!calleeSymbol.getType().getInputParamTypes().empty()) {
    parent->setToError(
        ErrorTree(user.getLoc(), "cannot reference parametric function"));
    return ElaborationState::error();
  }

  std::vector<ImplNode *> concrete;
  ParameterExprArrayAttr inputParamKey;
  GeneratorOp gen;
  bool wasSkipped = false;
  ParamNode *calleeNode;
  ElaborationState result = instantiateGeneratorReference(
      parent, user, calleeSymbol, inputParamKey, gen, concrete,
      [&](ParamNode *genNode) {
        calleeNode = genNode;
        return wasSkipped = !genNode->gen.getResultParams().empty() ||
                            isa<ParamApplyOp>(user);
      });
  if (result.isError() || (result.shouldSkipNode() && wasSkipped))
    return result;

  for (auto [i, resultType] : llvm::enumerate(user->getResultTypes())) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, parent, user.getLoc(), resultType);
    user->getResult(i).setType(type);
  }

  // We don't have to suspend elaboration of this node if the instantiation of a
  // generator with no result parameters is not yet ready. Process it later.
  if (result.shouldSkipNode()) {
    assert(parent->numDependencies >= 1 && "impossible for impl to be done");
    parent->dependencies.emplace_back(user, calleeNode);
    if (calleeNode->state.addWaiter()) {
      ++parent->numDependencies;
      calleeNode->andThenAsync(
          [this, parent] { completeImplNodeProcessing(parent); });
    }
    return ElaborationState::advance();
  }

  // If this resolved to a direct function call, there are no parameters.
  ArrayRef<ParamDeclAttr> decls = user.getParamDecls();
  if (!inputParamKey)
    return completeCallProcessing(user, decls, concrete.front(), parent);

  return completeGeneratorUserProcessing(user, decls, parent, inputParamKey,
                                         gen, concrete);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::completeGeneratorUserProcessing
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::completeGeneratorUserProcessing(
    GeneratorUserOpInterface user, ArrayRef<ParamDeclAttr> decls,
    ImplNode *parent, ParameterExprArrayAttr inputParamKey, GeneratorOp gen,
    ArrayRef<ImplNode *> concrete) {
  // There are more concrete things, we have to multi-version the parent!
  for (auto *c : llvm::drop_begin(concrete)) {
    // Clone the parent.
    IRMapping map;

    auto _ = logger.scope("New Multi-Versioning Op");
    logger.logOp("Op", c->func);
    LLVM_DEBUG(c->print(logger << "Concrete Implementation "));

    // This is a sibling to the parent, and it clones the parent's evaluator.
    ImplNode *newNode = fork(parent, map, "", c->func.getNameAttr());
    // Bind this concrete impl to this callee for this node.
    newNode->bindings.modify([=](auto &map) { map[{inputParamKey, gen}] = c; });

    // The call operation in the cloned function wil be handled by
    // `completeCallProcessing` below, so take it off the clone's worklist
    // beforehand, since new ops may be added.
    if (!newNode->stack.empty()) {
      assert(map.lookup(&*user) == newNode->stack.back().ops.back());
      newNode->stack.back().ops.pop_back();
    }

    ElaborationState result = completeCallProcessing(
        cast<GeneratorUserOpInterface>(map.lookup(user.getOperation())), decls,
        c, newNode);
    if (result.isError()) {
      // If call processing completion failed, then don't enqueue this node.
      assert(newNode->error && "expected an error on new node");
      continue;
    }
    if (result.shouldSkipNode())
      return result;

    LLVM_DEBUG(newNode->print(logger << "New Op "));

    // Process the rest of the worklist in this new scope. If the scope
    // processing failed, do nothing.
    assert(newNode->parent->numActive != 0 &&
           "forking a completed parameter node?");
    initialScheduleImplNode(newNode);
  }

  // Bind this concrete impl to this callee for this node.
  parent->bindings.modify([&](auto &map) {
    map[{inputParamKey, gen}] = concrete.front();
  });

  // Call completeGeneratorUserProcessing on the first concrete thing. This will
  // flow nested bindings upward correctly.
  return completeCallProcessing(user, decls, concrete.front(), parent);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::completeCallProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
static ElaborationState processParamApplyOp(ImplNode *inode, ParamApplyOp op,
                                            FuncOp func) {
  SmallVector<TypedAttr> operands;
  for (TypedAttr operand : op.getOperands()) {
    Attribute value;
    HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), operand);
    operands.push_back(cast<TypedAttr>(value));
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
  return ElaborationState::advance();
}

ElaborationState ElaboratorImpl::completeCallProcessing(
    GeneratorUserOpInterface user, ArrayRef<ParamDeclAttr> decls,
    ImplNode *thisNode, ImplNode *node, bool invertLockOrder) {
  // Add the callee's bindings to the parent of the call. This ensures that we
  // don't re-bind something we've already bound.
  //
  // NOTE: Concurrent access here is very unlikely. It can only happen after
  // breaking recursion. Use coarse-grained locking for simplicity. Also, the
  // nodes may be the same in cases of recursion, in which case nothing needs to
  // be done and avoid deadlocking.
  if (thisNode != node) {
    // NOTE: The mutexes on `bindings` must be acquired in the same order as in
    // `collectConcreteImplementations`, where the parent (caller node) acquires
    // the mutexes of callee nodes. Otherwise, a deadlock can occur if two
    // threads each hit one of the two pieces of code with the same nodes at the
    // same time. In normal call completion, acquire the parent lock first. When
    // breaking recursion, however, `invertLockOrder` is set to reverse the
    // order, because in breaking recursion, the child node becomes a parent.
    auto addBindings = [](auto &nodeBindings, auto &thisNodeBindings) {
      for (const auto &[k, v] : thisNodeBindings) {
        auto &oldV = nodeBindings[k];
        assert(!oldV || oldV == v);
        oldV = v;
      }
    };
    if (invertLockOrder) {
      thisNode->bindings.read([node, addBindings](auto &thisNodeBindings) {
        node->bindings.modify([&](auto &nodeBindings) {
          addBindings(nodeBindings, thisNodeBindings);
        });
      });
    } else {
      node->bindings.modify([thisNode, addBindings](auto &nodeBindings) {
        thisNode->bindings.read([&](auto &thisNodeBindings) {
          addBindings(nodeBindings, thisNodeBindings);
        });
      });
    }
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

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  mlir::IRRewriter b{OpBuilder(user)};
  auto newCallee = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
      newCalleeFunc.getSignature());
  user.concretizeCallee(b, newCallee);

  if (decls.empty())
    return ElaborationState::advance();

  // If we don't have the result parameters yet, then either no result
  // parameters are necessary, or we have another problem entirely wherein we
  // could not complete the callee's result parameter resolution at all - likely
  // meaning we're in an infinite recursive loop. Essentially, we came back to
  // the same combination of generator + input parameters without resolving the
  // result parameters yet.
  ParameterExprArrayAttr resultParams = thisNode->resultParams;
  assert(resultParams && "expected result parameters to be set");

  // Bind the result parameters to the output parameter decls.
  assert(decls.size() == resultParams.size());
  for (auto [decl, bindValue] : llvm::zip(decls, resultParams)) {
    LLVM_DEBUG(logger << "Binding " << decl << " to " << bindValue << "\n");
    node->getEvaluator().setOrOverwriteParameterValue(decl, bindValue);
  }
  return ElaborationState::advance();
}
//===----------------------------------------------------------------------===//
// ElaboratorImpl::processCallOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
ElaborationState ElaboratorImpl::processCallOp(ImplNode *parent,
                                               GeneratorUserOpInterface call) {
  Attribute symbol;
  HANDLE_EVALUATOR_CONC(symbol, parent, call.getLoc(), call.getCallee());
  return processGeneratorUser(call, cast<SymbolConstantAttr>(symbol), parent);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processEvaluateOp
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::processEvaluateOp(ImplNode *parent,
                                                   ParamEvaluateOp op) {
  Attribute evaluatorFn;
  HANDLE_EVALUATOR_CONC(evaluatorFn, parent, op.getLoc(), op.getEvaluator());

  ParameterExprArrayAttr inputParamKey;
  GeneratorOp gen;
  std::vector<ImplNode *> evaluators;
  ElaborationState result = instantiateGeneratorReference(
      parent, op, cast<SymbolConstantAttr>(evaluatorFn), inputParamKey, gen,
      evaluators);
  if (result.isError() || result.shouldSkipNode())
    return result;

  if (evaluators.size() != 1) {
    parent->setToError(ErrorTree(
        op.getLoc(), "evaluator did not resolve to a single candidate"));
    return ElaborationState::error();
  }

  Attribute candidates;
  HANDLE_EVALUATOR_CONC(candidates, parent, op.getLoc(), op.getCandidates());

  std::vector<ImplNode *> concrete;
  for (TypedAttr value : cast<VariadicAttr>(candidates).getValues()) {
    ElaborationState result = instantiateGeneratorReference(
        parent, op, cast<SymbolConstantAttr>(value), inputParamKey, gen,
        concrete);
    if (result.isError() || result.shouldSkipNode())
      return result;
  }

  std::vector<FuncOp> candidateFns;
  candidateFns.reserve(concrete.size());
  for (ImplNode *node : concrete)
    candidateFns.push_back(node->func);
  if (ErrorOrSuccess evalResult = evaluateFunctions(
          parent, evaluators.front()->func, std::move(candidateFns));
      evalResult.isError()) {
    parent->setToError(ErrorTree(op.getLoc(), evalResult.takeError()));
    return ElaborationState::error();
  }
  // Suspend elaboration. The actual search will be performed in isolation
  // later.
  return ElaborationState::skipNode();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamIfOp
//===----------------------------------------------------------------------===//

ElaborationState ElaboratorImpl::processParamIfOp(ImplNode *parent,
                                                  ParamIfOp op) {
  // Check the condition expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, parent, op.getLoc(), op.getCond());

  // Take whichever branch the condition indicated, and simply inline those ops
  // then elaborate them. We can do this by splicing the op list into the parent
  // block. We splice it this way to avoid remapping the ops when we process
  // them later.
  bool resultBool = cast<BoolAttr>(value).getValue();
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
  ImplNode::WorkItem item{std::move(opsToRewrite), nullptr};

  // When the nested scope completes processing, finish processing the current
  // parameter if.
  item.onComplete = [this, resultBool](ImplNode *node) -> LogicalResult {
    assert(node->stack.size() >= 2 && "expected at least two work items");
    // Retrieve the current state.
    ImplNode::WorkItem &parentFrame = *std::next(node->stack.rbegin());
    auto op = cast<ParamIfOp>(parentFrame.ops.back());
    LLVM_DEBUG(logger << "Parameter if completion callback: " << op);

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

void ElaboratorImpl::completeImplNodeProcessing(ImplNode *inode) {
  ParamNode *p = inode->parent;
  // This waiter was triggered in an error scenario. No further action is needed
  // because we are destroying the tree.
  if (p->getIsError())
    return;
  // If the node resulted in an error or all outstanding dependencies are
  // done, complete node processing. Otherwise, if the node has an error state,
  // it could end up completing early. Avoid double-completion by using a flag.
  //
  // NOTE: This is one of the two spots where an ImplNode may be accessed in
  // parallel. Synchronize the error state check using an atomic. Any data race
  // here is benign but this makes TSAN happy.
  bool hasError = inode->hasError.load();
  if ((!hasError && (--inode->numDependencies != 0)) ||
      inode->done.exchange(true)) {
    signalWorklist();
    return;
  }

  if (!hasError) {
    // Complete processing of outstanding dependencies. Process in reverse with
    // `pop_back` so that forks will end up in the same state.
    while (!inode->dependencies.empty()) {
      auto [call, genNode] = inode->dependencies.back();
      inode->dependencies.pop_back();

      // Check for an existing binding.
      if (ImplNode *existing = inode->bindings.read([p = genNode](auto &map) {
            return map.lookup({p->inputParams, p->gen});
          })) {
        LLVM_DEBUG(existing->print(logger.scope("Result: Existing Binding")));
        ElaborationState result =
            completeCallProcessing(call, {}, existing, inode);
        if (result.isError())
          break;
        assert(!result.shouldSkipNode() && !result.shouldSkipFrame() &&
               "expected all dependencies to be ready");
        continue;
      }
      // Otherwise, get all bound nodes.
      std::vector<ImplNode *> concrete;
      if (failed(
              collectConcreteImplementations(call, inode, genNode, concrete)))
        break;
      // Process the multiple concrete nodes. If this causes multi-versioning,
      // the forks will correctly get rescheduled on the worklists with no
      // stacks, and then immediately fallthrough to this function.
      ElaborationState result = completeGeneratorUserProcessing(
          call, {}, inode, genNode->inputParams, genNode->gen, concrete);
      if (result.isError())
        break;
      assert(!result.shouldSkipNode() && !result.shouldSkipFrame() &&
             "expected all dependencies to be ready");
    }
    if (!inode->error)
      finalizeFunction(inode);
  }

  // If this is the last implementation node for its parent parameter node to
  // complete, then the parameter node is done.
  assert(p->numActive > 0 && "node already done?");
  if (--p->numActive == 0) {
    g.numWorkItems.fetch_add(p->state.markDone());
    p->emplace();
  }
  signalWorklist();
}

void ElaboratorImpl::scheduleImplNode(ImplNode *inode) {
  // Increment the number of scheduled work items.
  runtime.getWorkQueue()->addTask([inode, this] {
    // Process the node. If processing the node got pre-empted, then return. It
    // will get scheduled again later.
    if (succeeded(processImplNode(inode))) {
      g.numWorkItems.fetch_add(1);
      completeImplNodeProcessing(inode);
    }
    // Signal the worklist that the work is complete.
    signalWorklist();
  });
}

LogicalResult ElaboratorImpl::processImplNode(ImplNode *inode) {
  // Check for a root node.
  if (!inode->func) {
    // Begin specialization of the parameter node. Immediately suspend
    // execution by returning `failure`.
    (void)specializeGenerator(inode, inode->parent, /*from=*/nullptr,
                              /*addWaiter=*/true);
    return failure();
  }
  if (inode->stack.empty())
    return success();

  LLVM_DEBUG(inode->print(logger << "Processing implementation node: ",
                          /*printBindings=*/false));
  CompilerTimeTraceScope traceScope(
      "processImplNode", [inode] { return inode->func.getSymName().str(); });

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
      // Node skip indicates to suspend elaboration of the current function
      // and come back later.
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
  return success();
}

ElaborationState ElaboratorImpl::processScope(ImplNode *node,
                                              ImplNode::WorkItem &item) {
  LLVM_DEBUG({
    auto _ = logger.scope("Operations to Rewrite");
    for (Operation *op : item.ops)
      logger << *op << "\n";
  });
  VerboseCompilerTimeTraceScope traceScope("processScope", [&item]() {
    return std::to_string(item.ops.size()) + " ops";
  });

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
    return processParamDeclareOp(node, declare);
  } else if (auto bind = dyn_cast<ParamResultBindOp>(op)) {
    return processParamResultBindOp(node, bind);
  } else if (auto fork = dyn_cast<ParamForkOp>(op)) {
    return processParamForkOp(node, fork);
  } else if (auto constant = dyn_cast<ParamConstantOp>(op)) {
    return processParamConstantOp(node, constant);
  } else if (auto constant = dyn_cast<ParamMaterializeOp>(op)) {
    return processParamConstantOp(node, constant);
  } else if (auto rebindOp = dyn_cast<RebindOp>(op)) {
    return processRebindOp(node, rebindOp);
  } else if (auto assertOp = dyn_cast<ParamAssertOp>(op)) {
    return processParamAssertOp(node, assertOp);
  } else if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
    return processParamIfOp(node, ifOp);
  } else if (auto call = dyn_cast<GeneratorUserOpInterface>(op)) {
    return processCallOp(node, call);
  } else if (auto evaluate = dyn_cast<ParamEvaluateOp>(op)) {
    return processEvaluateOp(node, evaluate);
  } else if (isa<DebugInfo::ValueOp>(op)) {
    // Delay elaboration of the DILocalVariableAttr until when locations are
    // elaborated.
    return ElaborationState::advance();
  } else {
    // NOTE: We only need to elaborate locations manually for generic ops if we
    // don't do it globally.
    return processGenericOp(node, op);
  }
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeGenerator
//===----------------------------------------------------------------------===//

/// Concretizes the attribute that may contains parameters. If unsuccessful,
/// sets the ImplNode to the error state and returns null.
template <typename AttrType>
static AttrType concretizeAttr(AttrType attr, mlir::Location loc,
                               ImplNode *inode) {
  auto exprResult =
      inode->getEvaluator().concretizeParameterExpr(inode, loc, attr);
  if (exprResult.isError()) {
    inode->setToError(exprResult.takeError());
    return {};
  }
  if (LLVM_UNLIKELY(!*exprResult)) {
    inode->setToError(ErrorTree(
        loc, "concretized parameter expression in attribute is null"));
    return {};
  }
  return cast<AttrType>(*exprResult);
}

/// Concretizes the location of an op or a block argument.
template <typename ArgOrOp>
static LogicalResult concretizeLocOf(ArgOrOp &argOrOp, ImplNode *inode) {
  mlir::LocationAttr loc = argOrOp.getLoc();
  if (mlir::LocationAttr newLocAttr =
          concretizeAttr<mlir::LocationAttr>(loc, loc, inode)) {
    argOrOp.setLoc(newLocAttr);
    return success();
  }
  return failure();
};

/// Try extracting a short name from a mangled name.
/// E.g. for the mangled name "$math::$math::log($builtin::$simd::SIMD[type,
/// simd_width])" we want to extract "log". This is the part before the opening
/// brace and after the last ':' before it.
static StringRef tryGettingShortName(StringRef s) {
  return s.split('(').first.rsplit(':').second;
}

ElaborationState ElaboratorImpl::specializeGenerator(ImplNode *inode,
                                                     ParamNode *genNode,
                                                     ParamNode *from,
                                                     bool addWaiter) {
  switch (genNode->state.markInProgress()) {
  case ParamNodeState::DONE:
    return ElaborationState::advance();
  case ParamNodeState::IN_PROGRESS:
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
    if (addWaiter) {
      if (genNode->state.addWaiter()) {
        genNode->andThenSync([inode, this] { scheduleImplNode(inode); });
        return ElaborationState::skipNode();
      }
      // Raced with node completion.
      return ElaborationState::advance();
    }
    return ElaborationState::skipNode();
  default:
    break;
  }

  // If this generator is from a package, then attempt to find its
  // specialization within it.
  if (FlatSymbolRefAttr linkRef = genNode->gen.getPreCompiledModuleRefAttr())
    specializeFromPackage(inode, genNode, linkRef, addWaiter);
  else
    specializeFromSource(inode, genNode, addWaiter);
  return ElaborationState::skipNode();
}

void ElaboratorImpl::specializeFromSource(ImplNode *inode, ParamNode *genNode,
                                          bool addWaiter) {
  GeneratorOp gen = genNode->gen;

  // Bind all parameter values in this scope.
  ArrayRef<TypedAttr> inputParamValues = genNode->inputParams.getValue();
  ArrayRef<ParamDeclAttr> inputParamDecls = gen.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  IREvaluator evaluator(*this);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setOrOverwriteParameterValue(decl, val);

  CompilerTimeTraceScope traceScope(
      "specializeGenerator:" + tryGettingShortName(gen.getName()).str(),
      gen.getName().str());
  auto genScope = logger.scope("Specializing Generator: @", gen.getName());
  logger.logOp("Generator", gen);

  // Get a partial ordering of parameter definitions and uses that are listed
  // "top down" in our evaluation order, if we don't have one already. This
  // should happen exactly once for each  node. This will be tricky to
  // parallelize as-is - we should change the approach a bit to have a
  // ParametricNode (or similar) that doesn't store the input parameters, in
  // which we could store the ParameterUseDefGraph.
  ParameterUseDefGraph *genNodeGraph =
      knownGraphs.read([gen](const auto &map) -> ParameterUseDefGraph * {
        if (auto it = map.find(gen); it != map.end())
          return it->second.get();
        return nullptr;
      });
  if (!genNodeGraph) {
    // Compute a new graph. The computed graph could end up getting discarded if
    // two threads end up here at the same time for the same generator.
    auto newGraph = std::make_unique<ParameterUseDefGraph>(gen.getBodyRegion());
    newGraph->calculate(paramCache.getThreadLocalCache());
    // Make sure to use whichever graph ended up in the map.
    genNodeGraph = knownGraphs.modify(
        [gen, newGraph = std::move(newGraph)](auto &map) mutable {
          return map.try_emplace(gen, std::move(newGraph)).first->second.get();
        });
  }

  std::string baseName = mangleParameterValues(gen, inputParamValues);

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(gen.getContext());
  StringAttr mangledName = b.getStringAttr(baseName);

  auto newFunc = b.create<FuncOp>(
      gen.getLoc(), mangledName,
      SignatureType::get(gen.getFunctionType(),
                         gen.getSignature().getArgConventions(),
                         gen.getSignature().getFnEffects()),
      gen.getInlineLevel(), gen.getExportKind(), gen.getDecorators(),
      gen.getLLVMMetadata());

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  newSymTab.modify([newFunc](SymbolTable &symtab) { symtab.insert(newFunc); });

  // Clone the body of the generator into the function.
  // TODO: is there a nice way for us to avoid cloning this?
  IRMapping map;
  gen.getBodyRegion().cloneInto(&newFunc.getBodyRegion(), map);

  // Map from the generator to the new function for the parameter graph copy.
  map.map(gen.getOperation(), newFunc.getOperation());
  // Copy over the parameter use-def graph for this clone.
  ParameterUseDefGraph childGraph = genNodeGraph->copy(map);

  // The node for this new func is simply the child of the node for the
  // generator.
  auto childNode =
      std::make_unique<ImplNode>(newFunc, genNode, std::move(childGraph),
                                 std::move(baseName), std::move(evaluator));
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

  // Since the function will have a new name, we need to update the linkage name
  // in the subprogram information.
  if (auto scope = newFunc.getSubprogramScope()) {
    SmallVector<StringAttr> paramValues;
    for (TypedAttr value : inputParamValues)
      paramValues.push_back(getParamTypeAsString(value));
    DebugInfo::SourceNameAttr name = scope.getName();
    name = DebugInfo::SourceNameAttr::get(name.getName(), name.getParamTypes(),
                                          name.getArgTypes(), paramValues,
                                          name.getParent(), name.getKind());
    StringRef linkageName = newFunc.getSymName();
    if (inputParamValues.empty())
      linkageName.consume_back("_concrete");
    DebugInfo::updateSubprogram(
        newFunc, StringAttr::get(name.getContext(), linkageName), name);
  }

  std::function<LogicalResult(ImplNode *)> onComplete;
  if (config.elaborateDebugInfo) {
    // We need to recursively elaborate locations within nested regions, both on
    // ops and block arguments. We do this after the worklist is processed, to
    // ensure that all parameter computation is completed, e.g. we have
    // processed all kgen.param.decl ops.
    onComplete = [&](ImplNode *inode) -> LogicalResult {
      inode->func->walk([&](Operation *op) -> WalkResult {
        if (failed(concretizeLocOf(*op, inode)))
          return WalkResult::interrupt();

        // Update the ValueInfo attr since they contain types.
        if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
          value->setAttrs(
              concretizeAttr(value->getAttrDictionary(), op->getLoc(), inode));
        }

        // To be defensive, we only concretize location attributes if we know
        // what we are dealing with.
        if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op)) {
          if (mlir::LocationAttr callLoc = inlined.getCallLocAttr()) {
            inlined.setCallLocAttr(concretizeAttr<mlir::LocationAttr>(
                callLoc, op->getLoc(), inode));
          }
        }

        // When elaboration is complete, only the first block in any region is
        // valid (any other block may be illegal, e.g. due to how kgen.param.if
        // is handled). So we only need to go through the region arguments.
        for (Region &r : op->getRegions()) {
          for (BlockArgument arg : r.getArguments())
            if (failed(concretizeLocOf(arg, inode)))
              return WalkResult::interrupt();
        }

        return WalkResult::advance();
      });

      if (inode->error)
        return failure();
      return success();
    };
  } else {
    onComplete = [](ImplNode *) { return success(); };
  }

  ImplNode::WorkItem item{std::move(opsToRewrite), std::move(onComplete)};
  newFuncNode->stack.push_back(std::move(item));

  if (addWaiter) {
    [[maybe_unused]] bool added = genNode->state.addWaiter();
    assert(added);
    genNode->andThenSync([inode, this] { scheduleImplNode(inode); });
  }
  assert(genNode->numActive == 0 && "expected first implementation");
  initialScheduleImplNode(newFuncNode);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeFromPackage
//===----------------------------------------------------------------------===//

static llvm::MemoryBufferRef getBufferRef(mlir::AsmResourceBlob *blob) {
  return llvm::MemoryBufferRef(
      StringRef(blob->getData().begin(), blob->getData().size()), "");
}

ElaboratorImpl::PackageReaderState::PackageReaderState(
    MLIRContext *ctx, mlir::AsmResourceBlob *blob)
    : sourceMgr(std::make_shared<llvm::SourceMgr>()),
      config(ctx, /*verifyAfterParse=*/false),
      reader(getBufferRef(blob), config, /*lazyLoad=*/true, sourceMgr) {}

LogicalResult ElaboratorImpl::PackageReaderState::initialize() {
  if (failed(reader.readTopLevel(&block)))
    return failure();
  module = cast<ModuleOp>(block.front());
  if (failed(reader.materialize(module)))
    return failure();
  symtab.emplace(module);
  return success();
}

void ElaboratorImpl::specializeFromPackage(ImplNode *parent, ParamNode *genNode,
                                           FlatSymbolRefAttr linkRef,
                                           bool addWaiter) {
  auto [state,
        inserted] = packages.modify([name = linkRef.getAttr(), this](auto &map)
                                        -> std::pair<PackageState *, bool> {
    if (auto it = map.find(name); it != map.end())
      return {it->second.get(), false};
    auto state = std::make_unique<PackageState>();
    state->ch = AsyncValueRef<Chain>::allocate(runtime);
    return {map.try_emplace(name, std::move(state)).first->second.get(), true};
  });

  // Keep the workqueue chain alive while this occurs.
  g.numWorkItems.fetch_add(1);
  state->ch.andThenAsync([state = state, parent, genNode, addWaiter, this] {
    // If the package does not have a precompiled module available, then just
    // specialize from the existing IR.
    if (!state->error && !state->reader) {
      specializeFromSource(parent, genNode, addWaiter);
      signalWorklist();
      return;
    }

    // We will complete the ParamNode here, so schedule a waiter if requested.
    if (addWaiter) {
      [[maybe_unused]] bool added = genNode->state.addWaiter();
      assert(added);
      genNode->andThenSync([parent, this] { scheduleImplNode(parent); });
    }

    // If the package is in an error state, create a dummy error.
    if (state->error) {
      auto impl = std::make_unique<ImplNode>(genNode);
      impl->setToError(ErrorTree(state->link.getLoc(), state->error->copy()));
      genNode->impls.push_back(std::move(impl));
    } else {
      // Otherwise, we are good to start reading into the concrete module.
      // First, check if the specialization was already loaded.
      StringAttr name = genNode->gen.getSymNameAttr();
      FuncOp func = newSymTab.read([name](const SymbolTable &symtab) {
        return symtab.lookup<FuncOp>(name);
      });

      auto addImplNode = [genNode, this](FuncOp func) {
        assert(genNode->impls.empty() && "could not be already specialized");
        genNode->impls.push_back(std::make_unique<ImplNode>(
            func, genNode, ParameterUseDefGraph(/*scope=*/nullptr), "",
            IREvaluator(*this)));
        return genNode->impls.back().get();
      };
      if (!func) {
        assert(state->reader && "expected a valid package");
        PackageReaderState &reader = *state->reader;

        // This function wasn't already pulled in. Load the specialization now
        // into the concrete module.
        func = reader.symtab->lookup<FuncOp>(name);
        assert(func && "missing function in bytecode module");
        ImplNode *inode = addImplNode(func);

        // FIXME: Avoid locking the symbol table for the whole duration here.
        auto loadFunc = [func, &reader](SymbolTable &symtab) {
          llvm::sys::SmartScopedWriter<true> guard(reader.mutex);
          func->remove();
          symtab.insert(func);
          return loadSymbolsFromBytecode(func, reader.reader, symtab,
                                         *reader.symtab);
        };
        if (failed(newSymTab.modify(loadFunc))) {
          inode->setToError(
              ErrorTree(state->link.getLoc(),
                        Error("failed to read body from bytecode")));
        }
      } else if (genNode->impls.empty()) {
        // The function was pulled in as a transitive dependency of another
        // function. Just ensure an ImplNode is created for it.
        addImplNode(func);
      }
    }

    // Complete the ParamNode.
    assert(genNode->numActive == 0 && "counter should always have been 0");
    g.numWorkItems.fetch_add(genNode->state.markDone());
    genNode->emplace();
    signalWorklist();
  });

  // The first thread to access a package state gets to initialize it. Other
  // threads will just go wait on the chain.
  if (!inserted)
    return;

  // Find the package link operation in the non-concrete module.
  auto link = oldSymTab.lookup<PackageLinkOp>(linkRef.getAttr());
  assert(link && "package reference does not refer to a package link op");
  state->link = link;

  auto setToError = [state = state](Error err) {
    // Given an error, propagate it to the package state so that waiters can
    // read it.
    state->error = std::move(err);
    // The package is done processing.
    state->ch.copy().emplace();
  };

  PackageArchiveAttr archive;

  // If the package link supplies a precompiled package, use it.
  if (std::optional<PackageArchiveAttr> maybeArchive =
          link.getArchivesAttr().getTargetArchive(target))
    archive = *maybeArchive;
  if (!archive) {
    // Otherwise, invoke the package handler callback to attempt on-demand
    // compilation. The callback is at liberty to modify the link op, and only
    // one thread can access each link op at a time.
    ErrorOr<PackageArchiveAttr> result =
        callbacks.packageHandlerFn(link, target);
    if (result.isError())
      return setToError(result.takeError());
    archive = result.takeValue();
  }

  // Set up the package reader.
  if (!archive) {
    // No bytecode is avaiable. Just signal waiters to call into
    // `specializeFromSource`.
    state->ch.copy().emplace();
    return;
  } else {
    // Add the concretized link op into the concrete module.
    OpBuilder b(link.getContext());
    auto newLink =
        b.create<LinkOp>(link.getLoc(), link.getSymNameAttr(),
                         archive.getArchive(), archive.getDependencies());
    newSymTab.modify([newLink, this](SymbolTable &symtab) {
      symtab.insert(newLink);
      deferredSymbols.push_back(newLink);
    });
  }

  mlir::AsmResourceBlob *blob =
      archive.getElaboratedModule().getRawHandle().getBlob();
  if (!blob)
    return setToError("unable to find the post-elaboration module blob");

  // Prepare the package state for bytecode reading.
  auto reader = std::make_unique<PackageReaderState>(link.getContext(), blob);
  if (failed(reader->initialize()))
    return setToError("failed to read top-level bytecode module");
  state->reader = std::move(reader);

  // This package state is ready to be read.
  state->ch.copy().emplace();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processedDeferredSearchFns
//===----------------------------------------------------------------------===//

bool ElaboratorImpl::processDeferredSearchFns() {
  if (deferredSearchFns.get().empty())
    return false;

  std::vector<ImplNode *> reschedule;
  reschedule.reserve(deferredSearchFns.get().size());
  for (DeferredSearch &search : deferredSearchFns.get()) {
    ImplNode *parent = search.inode;
    auto op = cast<ParamEvaluateOp>(parent->stack.back().ops.back());
    parent->stack.back().ops.pop_back();

    auto completeWithError = [&](Error err) {
      parent->setToError(ErrorTree(op.getLoc(), std::move(err)));
      g.numWorkItems.fetch_add(1);
      completeImplNodeProcessing(parent);
    };

    ErrorOr<ssize_t> bestIdx = search.searchFn();
    if (bestIdx.isError()) {
      completeWithError(bestIdx.takeError());
      continue;
    }
    if (*bestIdx == -1) {
      completeWithError("user-provided evaluator returned failure (-1)");
      continue;
    }
    if (*bestIdx < 0 ||
        *bestIdx >= static_cast<ssize_t>(search.candidates.size())) {
      completeWithError(
          "user-provided evaluator returned an out-of-bounds result: " +
          Twine(*bestIdx));
      continue;
    }
    FuncOp best = search.candidates[*bestIdx];
    LLVM_DEBUG(logger.logOp("best specialization", best));
    parent->getEvaluator().setParameterValue(
        op.getParamDecl(),
        SymbolConstantAttr::get(SymbolRefAttr::get(best.getSymNameAttr()),
                                best.getSignature()));
    // Handle the operation.
    reschedule.push_back(parent);
    op.erase();
  }
  deferredSearchFns.get().clear();

  // Now reschedule the nodes.
  for (ImplNode *inode : reschedule) {
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(inode);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::diagnoseAndBreakRecursion
//===----------------------------------------------------------------------===//

bool ElaboratorImpl::diagnoseAndBreakRecursion(unsigned generation,
                                               ArrayRef<ParamNode *> roots) {
  std::function<bool(ParamNode *)> visitParamNode = nullptr;
  std::vector<ImplNode *> reschedule;
  std::vector<ImplNode *> errComplete;

  std::function<void(ImplNode *)> visitImplNode = [&](ImplNode *inode) {
    // Skip completed nodes.
    if (inode->numDependencies == 0 || inode->error)
      return;

    llvm::BitVector completed(inode->dependencies.size());
    bool anyBroken = false;
    for (auto [idx, dep] : llvm::enumerate(inode->dependencies)) {
      auto [call, genNode] = dep;
      if (!visitParamNode(genNode))
        continue;
      // This `genNode` is cyclic. Handle the cycle. First, enforce that
      // recursive functions cannot have multiple implementations.
      if (genNode->impls.size() > 1) {
        for (ImplNode &impl : llvm::make_pointee_range(genNode->impls)) {
          if (impl.error)
            continue;
          inode->setToError(ErrorTree(
              call.getLoc(),
              "recursive call to function with more than 1 implementation"));
          errComplete.push_back(inode);
          break;
        }
      } else {
        assert(genNode->impls.size() == 1 && "expected at least 1 child");
        // Break the cycle. Set `invertLockOrder` as recursion processes nodes
        // in reverse order.
        (void)completeCallProcessing(call, {}, genNode->impls.front().get(),
                                     inode, /*invertLockOrder=*/true);
        completed.set(idx);
        anyBroken = true;
      }
    }
    if (anyBroken) {
      // Complete the broken dependencies and reschedule the node.
      std::vector<std::pair<GeneratorUserOpInterface, ParamNode *>> newDeps;
      for (auto [idx, dep] : llvm::enumerate(inode->dependencies)) {
        if (!completed.test(idx))
          newDeps.push_back(dep);
      }
      inode->dependencies = std::move(newDeps);
      inode->numDependencies -= (completed.count() - 1);
      reschedule.push_back(inode);
      return;
    }

    // Check if the node got stuck on a recursive call to something with result
    // parameters, since that's illegal but won't show up in `dependencies`.
    if (inode->stack.empty())
      return;
    if (auto call =
            dyn_cast<GeneratorUserOpInterface>(inode->stack.back().ops.back());
        call && !call.getCalleeSignature().getResultParamTypes().empty()) {
      // We should be able to lookup to concrete callee in the evaluator cache.
      ParameterEvaluator &eval = inode->getEvaluator();
      auto callee =
          cast<SymbolConstantAttr>(eval.getReboundAttribute(call.getCallee()));
      ParamNode *calleeNode = g.getOrCreate(
          runtime,
          ParameterExprArrayAttr::get(call.getContext(),
                                      callee.getParamValues()),
          oldSymTab.lookup<GeneratorOp>(
              cast<FlatSymbolRefAttr>(callee.getSymbol()).getAttr()),
          /*depth=*/0);
      if (visitParamNode(calleeNode)) {
        inode->setToError(
            ErrorTree(call.getLoc(),
                      "recursive call to function with result parameters"));
        errComplete.push_back(inode);
      }
    }
  };

  visitParamNode = [&](ParamNode *pnode) -> bool {
    if (pnode->state.getValue() == ParamNodeState::DONE) {
      return false;
    } else if (pnode->cycleGeneration != generation) {
      pnode->cycleGeneration = generation;
      pnode->cycleState = ParamNode::VISITED;
    } else if (pnode->cycleState == ParamNode::VISITED) {
      return true;
    } else {
      assert(pnode->cycleState == ParamNode::DONE);
      return false;
    }

    for (ImplNode &inode : llvm::make_pointee_range(pnode->impls))
      visitImplNode(&inode);
    pnode->cycleState = ParamNode::DONE;
    return false;
  };

  for (ParamNode *root : roots)
    visitParamNode(root);

  if (reschedule.empty() && errComplete.empty()) {
    // As a last ditch attempt, check all the nodes for any "islands", because
    // not all dependencies are tracked by `dependencies`. It's not worth paying
    // the cost for that dependency tracking when recursion is uncommon.
    for (auto &[_, pnode] : g.nodes.get())
      visitParamNode(pnode.get());
  }

  // Now reschedule the nodes outside the loop to avoid races.
  for (ImplNode *inode : reschedule) {
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(inode);
  }
  for (ImplNode *inode : errComplete) {
    g.numWorkItems.fetch_add(1);
    inode->stack.clear();
    completeImplNodeProcessing(inode);
  }
  return !reschedule.empty() || !errComplete.empty();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::run
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::run(ModuleOp theModule,
                                  ArrayRef<GeneratorOp> primaryGenerators) {
  LLVM_DEBUG(logger << "Starting Elaboration\n");
  MLIRContext *ctx = theModule.getContext();

  // Find any kgen.func we have already - they're already elaborated, and we do
  // not want to re-process them. Add concrete ImplNodes for each one.
  auto moveSymbol = [this](Operation *op) {
    oldSymTab.remove(op);
    op->remove();
    newSymTab.get().insert(op);
  };
  for (Operation &op : llvm::make_early_inc_range(theModule.getOps())) {
    if (auto func = dyn_cast<FuncOp>(op))
      addConcreteFunc(func);
    if (!isa<GeneratorOp, PackageLinkOp>(op) &&
        isa<mlir::SymbolOpInterface>(op))
      moveSymbol(&op);
  }

  auto emptyInputParamKey = ParameterExprArrayAttr::get(ctx, {});
  std::vector<AnyAsyncValueRef> primaryChs;
  std::vector<std::unique_ptr<ImplNode>> rootNodes;
  std::vector<ParamNode *> primaryNodes;
  primaryChs.reserve(primaryGenerators.size());
  primaryNodes.reserve(primaryGenerators.size());
  for (GeneratorOp gen : primaryGenerators) {
    LLVM_DEBUG(logger.logOp("Elaborating primary generator", gen));
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    ParamNode *genNode =
        g.getOrCreate(runtime, emptyInputParamKey, gen, /*depth=*/0);
    primaryNodes.push_back(genNode);

    // Create a special root node for this primary generator.
    ImplNode *root =
        rootNodes.emplace_back(std::make_unique<ImplNode>(genNode)).get();

    // Now we can begin to construct the expansion tree rooted at this
    // generator. Emit as many errors as possible.
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(root);
    primaryChs.push_back(genNode->copy());
  }

  // Process all current work.
  {
    CompilerTimeTraceScope traceScope("doElaboration");
    unsigned cycleGeneration = 0;
    while (true) {
      signalWorklist();
      LLCL::await(g.worklistCh);
      assert(g.numWorkItems == 0);

      // Check if all primary generators are done. If so, break.
      if (llvm::all_of(primaryChs, [](auto &ch) { return ch.isReady(); }))
        break;
      g.numWorkItems = 1;

      // Re-initialize the worklist chain.
      g.worklistCh = AsyncValueRef<Chain>::allocate(runtime);

      // Check for deferred search.
      if (processDeferredSearchFns())
        continue;
      // The only other possibility is a cycle due to recursion.
      if (diagnoseAndBreakRecursion(++cycleGeneration, primaryNodes))
        continue;
      // Anything else indicates a bug/race condition.
      llvm_unreachable("no work left, no deferred search, and no recursion?");
    }
  }

  // Check for any errors and emit them. Emit as many errors as possible.
  bool failed = false;
  for (ParamNode *genNode : primaryNodes) {
    ErrorTreeOrSuccess err = genNode->collectErrorsOrSuccess();
    if (err.isError()) {
      failed = true;
      err.takeError().emit([](Location loc) { return mlir::emitError(loc); });
    }
    if (!config.allowMultiplePrimaryImpls &&
        llvm::count_if(genNode->impls,
                       [](auto &impl) { return !impl->error; }) > 1) {
      InFlightDiagnostic diag = mlir::emitError(
          genNode->gen.getLoc(),
          "primary generator with more than one successful implementation");
      diag.attachNote() << "select one implementation using search or remove "
                           "forks in the implementation";
      failed = true;
    }
  }
  if (failed)
    return failure();

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  {
    CompilerTimeTraceScope traceScope("eraseFuncs");
    LLCL::ForkJoin eraseState(runtime);
    auto eraseFunc = [&eraseState](Operation *op, SymbolTable &symtab) {
      symtab.remove(op);
      op->remove();
      eraseState.fork([op] { op->erase(); });
    };
    // Sort instantiations of each generator to ensure we have a deterministic
    // output in multithreaded execution.
    struct SuccessfulFuncs {
      std::string paramStr;
      SmallVector<std::pair<StringRef, FuncOp>, 1> funcs;
    };
    llvm::MapVector<GeneratorOp, std::vector<SuccessfulFuncs>>
        genInstantiations;
    for (auto gen : theModule.getOps<GeneratorOp>())
      genInstantiations[gen];
    for (ParamNode &node :
         llvm::make_pointee_range(llvm::make_second_range(g.nodes.get()))) {
      CompilerTimeTraceScope traceScope(
          "processGen", [name = node.gen.getSymName()] { return name.str(); });
      FuncOp first;
      // Erase all erroneous functions.
      SmallVector<std::pair<StringRef, FuncOp>, 1> successfulFuncs;
      for (ImplNode &impl : llvm::make_pointee_range(node.impls)) {
        if (impl.error) {
          if (config.diagAllFailures) {
            mlir::emitRemark(impl.func.getLoc(), "other failed instantiations");
            std::move(*impl.error).emit([](Location loc) {
              return mlir::emitRemark(loc);
            });
          }
          eraseFunc(impl.func, newSymTab.get());
          continue;
        }
        if (!first)
          first = impl.func;
        successfulFuncs.emplace_back(impl.func.getSymName(), impl.func);
      }

      // Sort the successful instantiations, if there are more than 1.
      if (successfulFuncs.size() > 1) {
        llvm::sort(successfulFuncs,
                   [](auto &lhs, auto &rhs) { return lhs.first > rhs.first; });
      }
      genInstantiations[node.gen].push_back(SuccessfulFuncs{
          mlir::debugString(node.inputParams), std::move(successfulFuncs)});
    }

    // Now reorder all instantiations of each generator to be deterministic.
    Block *newBlock = newModule->getBody();
    for (auto &[gen, instantiations] : genInstantiations) {
      CompilerTimeTraceScope traceScope(
          "sortInstantiations",
          [name = gen.getSymNameAttr()] { return name.str(); });
      llvm::sort(instantiations, [](auto &lhs, auto &rhs) {
        return lhs.paramStr < rhs.paramStr;
      });
      for (auto &[_, implFuncs] : instantiations)
        for (FuncOp func : llvm::make_second_range(implFuncs))
          func->moveBefore(newBlock, newBlock->end());
    }

    // Erase all generators.
    cast<ModuleOp>(oldSymTab.getOp())
        .getBodyRegion()
        .takeBody(newModule->getBodyRegion());

    eraseState.join();
  }

  // Sort and then push on all the deferred functions.
  llvm::sort(deferredSymbols,
             [](mlir::SymbolOpInterface lhs, mlir::SymbolOpInterface rhs) {
               return lhs.getName() < rhs.getName();
             });
  for (mlir::SymbolOpInterface symbol : deferredSymbols) {
    symbol->remove();
    theModule.push_back(symbol);
  }

  // Update the symbol table with the new one.
  oldSymTab = std::move(newSymTab.get());
  // HACK: Need a `SymbolTable::setOp` to properly avoid recomputing the table.
  *((Operation **)&oldSymTab) = theModule;
  return success();
}

//===----------------------------------------------------------------------===//
// elaborateGenerators
//===----------------------------------------------------------------------===//

static LogicalResult elaborateGenerators(
    mlir::SymbolTableAnalysis &symtab, ParameterCollector::Analysis &paramCache,
    LLCL::Runtime &runtime, TargetInfoAttr target,
    ArrayRef<GeneratorOp> primaryGenerators, ElaboratorCallbacks callbacks,
    const ElaborateGeneratorsOptions &config) {
  CompilerTimeTraceScope traceScope("elaborate-generators");
  ModuleOp theModule = symtab.getTopLevelOp<ModuleOp>();

  auto noopEvaluator = [](FuncOp, const SymbolTable &, TargetInfoAttr,
                          ArrayRef<FuncOp>) { return [] { return 0; }; };
  if (!config.enableSearch)
    callbacks.evaluateFn = noopEvaluator;

  // Now, construct and run the elaborator.
  ElaboratorImpl impl(symtab.getTopLevelSymbolTable(), paramCache, target,
                      std::move(callbacks), runtime, config);
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
                          EvaluatorExecutorFn evaluatorExecutorFn = {},
                          ElaboratorCompileAsmFn compileAsmFn = {},
                          PackageLinkHandlerFn packageHandlerFn = {})
      : ElaborateGeneratorsBase(options), runtime(runtime), target(target),
        evaluatorExecutorFn(std::move(evaluatorExecutorFn)),
        compileAsmFn(std::move(compileAsmFn)),
        packageHandlerFn(std::move(packageHandlerFn)) {}

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

    // Default the evaluator to selecting the first specialization.
    if (!evaluatorExecutorFn) {
      evaluatorExecutorFn = +[](FuncOp, const SymbolTable &, TargetInfoAttr,
                                ArrayRef<FuncOp>) { return [] { return 0; }; };
    }

    // Default compile assembly hook will just error.
    if (!compileAsmFn) {
      compileAsmFn = +[](GeneratorOp, SymbolConstantAttr, StringAttr,
                         const SymbolTable &, TargetInfoAttr, EmissionKind) {
        return Error("internal error: cannot compile assembly without a JIT");
      };
    }

    // Default package handler does nothing.
    if (!packageHandlerFn) {
      packageHandlerFn =
          +[](PackageLinkOp, TargetInfoAttr) { return PackageArchiveAttr(); };
    }
    return success();
  }

  void runOnOperation() override {
    auto rt =
        ConditionallyOwnedPointer<LLCL::Runtime>::takeIfNeeded(runtime, []() {
          return LLCL::createUniqueRuntime(LLCL::RuntimeOptions().forDebug())
              .release();
        });
    ModuleOp theModule = getOperation();

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

    // Root elaboration on exports and global variables. These are the
    // generators that elaboration will start from. If there are no such
    // generators, then elaborate anything with no input parameters.
    DenseSet<GeneratorOp> roots;
    auto addAsRoot = [&](SymbolRefAttr ref) {
      roots.insert(analysis.getTopLevelSymbolTable().lookup<GeneratorOp>(
          cast<FlatSymbolRefAttr>(ref).getValue()));
    };
    for (Operation &op : theModule.getOps()) {
      if (auto gen = dyn_cast<GeneratorOp>(op); gen && gen.isExported()) {
        roots.insert(gen);
      } else if (auto global = dyn_cast<GlobalOp>(op);
                 global && global.getCtor()) {
        addAsRoot(*global.getCtor());
        addAsRoot(*global.getDtor());
      }
    }

    // Extract the top-level, parameterless generators from the main module.
    // These are the only generators that will be elaborated.
    SmallVector<GeneratorOp> primaryGenerators;
    for (auto gen : theModule.getOps<GeneratorOp>())
      if (gen.getInputParams().empty() &&
          (roots.empty() || roots.contains(gen)))
        primaryGenerators.push_back(gen);

    // Elaboration is the compilation phase in which the IR goes from
    // target-non-specific to target-specific: in order to fully concretize the
    // IR, we must evaluate compile-time expressions, which is a target-specific
    // operation. Make the IR target-specific by attaching the required target
    // specification.
    if (TargetInfoAttr targetInfo = getTargetInfo(theModule))
      target = targetInfo;
    else
      setTargetInfo(theModule, target);

    // If the module is missing an environment attribute, set an empty one.
    if (!theModule->hasAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName())) {
      theModule->setAttr(EnvAttr::getEnvAttrName(),
                         EnvAttr::get(DictionaryAttr::get(&getContext())));
    }

    ElaboratorCallbacks callbacks{evaluatorExecutorFn, compileAsmFn,
                                  packageHandlerFn};
    ElaborateGeneratorsOptions config{enableSearch, allowMultiplePrimaryImpls,
                                      maxDepth, elaborateDebugInfo,
                                      diagAllFailures};
    if (failed(elaborateGenerators(analysis, paramCache, *rt, target,
                                   primaryGenerators, callbacks, config)))
      return signalPassFailure();
  }

private:
  /// An optional LLCL runtime pointer.
  LLCL::Runtime *runtime;
  /// The compilation target.
  TargetInfoAttr target;
  /// The functor used for evaluating generator specializations.
  EvaluatorExecutorFn evaluatorExecutorFn;
  /// The functor used to compile a module to assembly.
  ElaboratorCompileAsmFn compileAsmFn;
  /// The functor used to on-demand compile a package.
  PackageLinkHandlerFn packageHandlerFn;
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(LLCL::Runtime &runtime, TargetInfoAttr target,
                                const ElaborateGeneratorsOptions &options,
                                EvaluatorExecutorFn evaluatorExecutorFn,
                                ElaboratorCompileAsmFn compileAsmFn,
                                PackageLinkHandlerFn packageHandlerFn) {
  return std::make_unique<ElaborateGeneratorsPass>(
      options, &runtime, target, std::move(evaluatorExecutorFn),
      std::move(compileAsmFn), std::move(packageHandlerFn));
}
