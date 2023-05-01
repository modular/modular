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
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/Semaphore.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/STLExtras.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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
// ExpansionTreeNode
//===----------------------------------------------------------------------===//

namespace {
/// This struct is a node in the expansion tree that describes the elaboration.
/// In general, we try to limit effects to a single subtree. The only exception
/// is that creating new generators/funcs generally are children of the root -
/// this is because they're semi-independent of the current node and will
/// elaborate to something concrete we can simply refer to. We try to track
/// dependencies in order to make that graph explicit.
struct ExpansionTreeNode {
public:
  static ExpansionTreeNode *
  create(Operation *op, ArrayAttr inputParams, ExpansionTreeNode *parent,
         const IREvaluator &evaluator, unsigned expansionDepth,
         std::optional<ParameterUseDefGraph> paramGraph = std::nullopt);

private:
  /// Construct an expansion tree node. The node adds itself to its parent's
  /// children list, and inherits its parent's evaluator.
  ExpansionTreeNode(Operation *op, ArrayAttr inputParams,
                    ExpansionTreeNode *parent, const IREvaluator &evaluator,
                    std::optional<ParameterUseDefGraph> paramGraph,
                    unsigned expansionDepth);

public:
  /// Construct an expansion tree node without a parent. This should only be
  /// used for the tree root.
  ExpansionTreeNode(ModuleOp op, const IREvaluator &evaluator,
                    llvm::SpecificBumpPtrAllocator<ExpansionTreeNode> &alloc,
                    Logger &logger)
      : op(op), evaluator(evaluator), paramGraph(std::nullopt), alloc(alloc),
        expansionDepth(0), logger(logger) {}

  /// Return the name of the operation this node represents.
  StringAttr getNameAttr() { return op.getNameAttr(); }

  /// Get the root of the tree. This is useful for when you need to add a new
  /// generator to multi-version a call, for example. There is always a single
  /// root.
  ExpansionTreeNode *getRoot();

  /// Compute the distance of this node from the root of the tree. Used to
  /// enforce invariants about the tree's structure regarding its depth.
  size_t distanceFromRoot();

  /// Check if we are at the root of the tree. There must be a single root.
  bool isRoot() { return getRoot() == this; }

  /// Return true if this can be resolved to at least one concrete value.
  bool isConcrete();

  /// Return the first concrete node in the subtree rooted on `this`. This is
  /// often called from a node that is either concrete, or only has one
  /// concretization. For generality in cases where the full list of concrete
  /// nodes is required, use getAllConcrete below. Returns an error if there are
  /// no concretizations of this node.
  ErrorTreeOr<ExpansionTreeNode *> getFirstConcreteNode();

  /// Get the first concrete FuncOp. This finds the first concrete node in the
  /// subtree, and returns its op cast to a FuncOp. This is always safe because
  /// if the node has been concretized, then the op is a FuncOp.
  ErrorTreeOr<FuncOp> getFirstConcrete() {
    auto concNode = getFirstConcreteNode();
    if (concNode.isError())
      return concNode.takeError();

    return cast<FuncOp>((*concNode)->op);
  }

  /// Get all the concrete nodes in the tree rooted on `this`. This is useful
  /// when you have something like a GeneratorInterface that can concretize to
  /// multiple valid generators, and from that multiple functions.
  void getAllConcreteNodes(std::vector<ExpansionTreeNode *> &concrete);

  /// Get all the concrete functions in the tree rooted on `this`. Exactly the
  /// same as `getAllConcreteNodes` above, but only returns the FuncOp. Useful
  /// when you don't need the full ExpansionTreeNode.
  void getAllConcrete(std::vector<FuncOp> &concrete);

  /// Trims children that failed out of the subtree rooted on `this`. This is
  /// useful for looking at the final state of a tree after resolution has
  /// occurred - specifically in cases where we may have a failed expansion due
  /// to a param.assert, for example.
  void trimFailedExpansions(DenseSet<Operation *> &toErase);

  /// Collect all the errors in my subtree into a single error tree.
  void collectSubtreeErrors(ErrorTree &tree);

  /// Take the provided node as a child and set the parent to `this`.
  void takeExpansion(ExpansionTreeNode *child) {
    // Erase the child from the vector of its original parent.
    auto found = llvm::find(child->parent->expansions, child);
    child->parent->expansions.erase(found);
    // Re-parent it under this parent.
    child->parent = this;
    expansions.push_back(child);
  }

  /// Take the provided error and set this node to an `error` state. Erase all
  /// state dominated by this node.
  void setToError(ErrorTree &&err);

  /// Update the debug info on the concretization of the current node. This
  /// means, if there are child nodes, update their debug info. This should be
  /// called after the operations are renamed at the end of the pass.
  void updateDebugInfo();

  /// Print this tree to the provided indented stream. This preserves any
  /// indentation provided by the caller to make it possible to nest things
  /// nicely.
  void print(mlir::raw_indented_ostream &os, bool printBindings = true);

  /// Dump this node to llvm::errs().
  LLVM_DUMP_METHOD void dump() {
    mlir::raw_indented_ostream os(llvm::errs());
    print(os);
  }

  /// Each node is rooted on an op that defines a symbol.
  mlir::SymbolOpInterface op;
  /// The tree node is uniqued by the pair [op, inputParams]. This is the input
  /// parameter storage.
  ArrayAttr inputParams;
  /// When you have result parameters, we need to store them to access them from
  /// outer scopes.
  ArrayAttr resultParams;
  /// The evaluator is shared with scopes below, but not scopes above,
  /// generally. That's why it's copied rather than taking a reference.
  IREvaluator evaluator;

  /// Keep track of the nested parameter scopes within this op. This is optional
  /// because for example, the root module does not have one of these.
  std::optional<ParameterUseDefGraph> paramGraph;

  /// Set of known parameter regions. Maps a string attr (name) to a
  /// ParameterUseDefGraph for the original region in its original context. This
  /// is needed because we inline the region directly and collect the ops we
  /// need to process from the original region.
  DenseMap<StringAttr, ParameterUseDefGraph *> knownRegions;

  /// Parent node. This is useful for setting parameters on the parent's scope,
  /// for example. Each node must have one parent.
  ExpansionTreeNode *parent = nullptr;

  /// Calls to the same interface/generator should resolve to the same thing in
  /// each func.
  DenseMap<std::pair<Operation *, ArrayAttr>, ExpansionTreeNode *> bindings;

  /// An error contained by this node. This allows us to delay error handling in
  /// cases where an error is recoverable.
  std::optional<ErrorTree> error;

  /// The children of a node are specializations. They may not be fully concrete
  /// in the case of e.g. an interface - where the children are generators that
  /// themselves have children.
  std::vector<ExpansionTreeNode *> expansions;

  /// The allocator to use for allocating new children.
  llvm::SpecificBumpPtrAllocator<ExpansionTreeNode> &alloc;

  /// The expansion depth - we use this to track recursion.
  unsigned expansionDepth;

  /// A logger to use for this node.
  Logger &logger;
};
} // namespace

// TODO: need to find a way to order these, insert them right before/after
// something else maybe?
ExpansionTreeNode *
ExpansionTreeNode::create(Operation *op, ArrayAttr inputParams,
                          ExpansionTreeNode *parent,
                          const IREvaluator &evaluator, unsigned expansionDepth,
                          std::optional<ParameterUseDefGraph> paramGraph) {
  auto *out = new (parent->alloc.Allocate())
      ExpansionTreeNode(op, inputParams, parent, evaluator,
                        std::move(paramGraph), expansionDepth + 1);
  assert(out->distanceFromRoot() <= 3 &&
         "Should have at most 3 hops to the root");
  parent->expansions.push_back(out);
  return parent->expansions.back();
}

ExpansionTreeNode::ExpansionTreeNode(
    Operation *op, ArrayAttr inputParams, ExpansionTreeNode *parent,
    const IREvaluator &evaluator,
    std::optional<ParameterUseDefGraph> paramGraph, unsigned expansionDepth)
    : op(op), inputParams(inputParams), evaluator(evaluator),
      paramGraph(std::move(paramGraph)), parent(parent),
      bindings(parent->bindings), alloc(parent->alloc),
      expansionDepth(expansionDepth), logger(parent->logger) {
  // TODO: make this configurable?
  if (expansionDepth > 128) {
    error = ErrorTree(
        op->getLoc(),
        "elaborator expansion is 129 levels deep - infinite recursion?");
  }
  LLVM_DEBUG(print(logger << "Constructing "));
}

bool ExpansionTreeNode::isConcrete() {
  if (expansions.empty())
    return !error.has_value();

  for (auto &c : expansions)
    if (c->isConcrete())
      return true;

  return false;
}

ExpansionTreeNode *ExpansionTreeNode::getRoot() {
  if (parent)
    return parent->getRoot();

  return this;
}

size_t ExpansionTreeNode::distanceFromRoot() {
  size_t result = 0;
  ExpansionTreeNode *ptr = this;
  while (ptr != getRoot()) {
    ptr = ptr->parent;
    ++result;
  }

  return result;
}

ErrorTreeOr<ExpansionTreeNode *> ExpansionTreeNode::getFirstConcreteNode() {
  // If I have no children, then I am error or concrete myself.
  if (expansions.empty()) {
    if (error)
      return error->copy();
    return this;
  }

  // Return the first successful child.
  for (auto &c : expansions) {
    auto concNode = c->getFirstConcreteNode();
    if (!concNode.isError())
      return concNode;
  }

  // Otherwise, collect up all the errors in my children and report them.
  ErrorTree out(op.getLoc(), "no successful concrete nodes");
  for (auto &c : expansions)
    out.addCause(c->error->copy());

  return out;
}

// TODO: make these iterative, not recursive?
void ExpansionTreeNode::getAllConcreteNodes(
    std::vector<ExpansionTreeNode *> &concrete) {
  // Only deal with leaves - we have to check and see if the error has been
  // set for this leaf.
  if (expansions.empty() && !error)
    return concrete.push_back(this);

  for (auto &ch : expansions)
    ch->getAllConcreteNodes(concrete);
}

void ExpansionTreeNode::getAllConcrete(std::vector<FuncOp> &concrete) {
  // If I am concrete, add my concrete impl and return. If I have an error,
  // then do nothing.
  if (expansions.empty() && !error)
    return concrete.push_back(cast<FuncOp>(op));

  // Otherwise, recurse into my children.
  for (auto &ch : expansions)
    ch->getAllConcrete(concrete);
}

void ExpansionTreeNode::trimFailedExpansions(DenseSet<Operation *> &toErase) {
  // If I am concrete, hold an error, and am unique, then erase my op and move
  // along.
  if (expansions.empty()) {
    if (!error)
      return;
    LLVM_DEBUG(print(logger.scope("Erasing Failed Node")));
    if (op)
      toErase.insert(op);
    return;
  }
  auto _ = logger.scope("Trimming Failed Children");
  // Only log the op if we have it - it may have already been deleted.
  LLVM_DEBUG({
    if (op)
      logger.logOp("Op", op);
  });

  // Post-order trimming here - visit children first.
  for (auto &ch : expansions)
    ch->trimFailedExpansions(toErase);

  size_t numChildren = expansions.size();
  // Erase children that failed.
  auto newEnd = llvm::remove_if(expansions,
                                [](auto &ch) { return ch->error.has_value(); });

  // If I've just erased all my children, then I have failed. Propagate the
  // error up.
  if (newEnd == expansions.begin() && numChildren != 0) {
    error = ErrorTree(op.getLoc(), "no viable expansions found");
    collectSubtreeErrors(*error);
  }

  // Finally, actually erase the vector.
  expansions.erase(newEnd, expansions.end());
}

void ExpansionTreeNode::collectSubtreeErrors(ErrorTree &tree) {
  if (expansions.empty() && error)
    tree.addCause(error->copy());

  for (auto &ch : expansions)
    ch->collectSubtreeErrors(tree);
}

void ExpansionTreeNode::setToError(ErrorTree &&err) {
  // Take the error as the error in this node.
  this->error = std::move(err);
  // Trim the children if this has any by setting them to error.
  for (ExpansionTreeNode *child : expansions)
    child->setToError(this->error->copy());

  // Erase the op if it's a FuncOp - can't erase generators cause different
  // input params can mean different things.
  if (llvm::isa_and_present<FuncOp>(op)) {
    op->erase();
    op = nullptr;
  }
}

void ExpansionTreeNode::updateDebugInfo() {
  for (auto &child : expansions)
    child->updateDebugInfo();

  if (!expansions.empty() || !op || error.has_value())
    return;

  auto oldFuncSp = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(op);
  if (!oldFuncSp)
    return;

  auto newFuncSp = DebugInfo::DISubprogramAttr::get(
      oldFuncSp.getContext(), oldFuncSp.getCompileUnit(), oldFuncSp.getScope(),
      op.getNameAttr(), op.getNameAttr(), oldFuncSp.getFile(),
      oldFuncSp.getLine(), oldFuncSp.getScopeLine(),
      oldFuncSp.getSubprogramFlags(), oldFuncSp.getType());
  DebugInfo::DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DebugInfo::DISubprogramAttr attr) { return newFuncSp; });
  replacer.recursivelyReplaceElementsIn(op);
}

void ExpansionTreeNode::print(mlir::raw_indented_ostream &os,
                              bool printBindings) {
  // If we don't have an op, don't bother printing anything.
  if (!op) {
    // Only print the top level error.
    if (error)
      os << "Error: " << error->getError() << "\n";
    return;
  }

  bool isRoot = (parent == nullptr);
  os << "ExpansionTreeNode <" << (isRoot ? "Root" : getNameAttr().getValue())
     << ">";
  auto _ = os.scope(" {\n", "}\n");

  // Don't print the operation if this is the root (no need to dump the whole
  // module).
  if (!isRoot) {
    {
      auto opScope = os.scope("Op: {\n", "}\n");
      op->print(os);
      os << "\n";
    }
    if (inputParams && !inputParams.empty())
      os << "InputParams: " << inputParams << "\n";
    if (resultParams && !resultParams.empty())
      os << "ResultParams: " << resultParams << "\n";
    {
      auto regionScope = os.scope("Known Regions: {\n", "}\n");
      for (auto &[name, _] : knownRegions)
        os << name << "\n";
    }
  }

  // Print the bindings only if requested - this is so we don't recurse
  // infinitely to print the bindings of a recursive call.
  if (printBindings) {
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

  // Print the children.
  if (!expansions.empty()) {
    auto childrenScope = os.scope("Children: {\n", "}\n");
    for (auto &child : expansions)
      child->print(os);
  }
}

//===----------------------------------------------------------------------===//
// processParamDeclareOp
//===----------------------------------------------------------------------===//

/// Process a param.declare op by setting its parameter value in the provided
/// evaluator.
static std::optional<ErrorTree> processParamDeclareOp(IREvaluator &evaluator,
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
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// processParamResultBindOp
//===----------------------------------------------------------------------===//

/// Process a `kgen.param.result_bind` operation by setting the result parameter
/// values of the parent operation.
static std::optional<ErrorTree>
processParamResultBindOp(ParamResultBindOp op, ExpansionTreeNode *parentNode) {
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
    resultParamDecls =
        cast<GeneratorOp>(parentNode->parent->op).getResultParams();
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
  return {};
}

//===----------------------------------------------------------------------===//
// processRebindOp
//===----------------------------------------------------------------------===//

static std::optional<ErrorTree> processRebindOp(IREvaluator &evaluator,
                                                RebindOp op) {
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
  return {};
}

//===----------------------------------------------------------------------===//
// processParamAssertOp
//===----------------------------------------------------------------------===//

/// Process a param.assert op by folding its parameter expression and checking
/// its constraint. Returns the appropriate error if the constraint failed.
static std::optional<ErrorTree> processParamAssertOp(IREvaluator &evaluator,
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
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// processLocation
//===----------------------------------------------------------------------===//

/// Handle location concretization.
static std::optional<ErrorTree> processLocation(IREvaluator &evaluator,
                                                Operation *op) {
  ErrorTreeOr<Attribute> value = evaluator.concretizeParameterExpr(
      op->getLoc(), op->getLoc(), /*allowUnknown=*/true);
  if (value.isError())
    return value.takeError();
  op->setLoc(cast<Location>(value.takeValue()));
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// processGenericOp
//===----------------------------------------------------------------------===//

/// Unknown operations are allowed to use types and attributes with parameter
/// references.  Substitute in concrete values for their references.
static std::optional<ErrorTree> processGenericOp(IREvaluator &evaluator,
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

  if (std::optional<ErrorTree> err = processLocation(evaluator, op))
    return std::move(*err);

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

  return std::nullopt;
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
// completeCallProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
static std::optional<ErrorTree>
processParamApplyOp(ParamApplyOp op, FuncOp func, ExpansionTreeNode *parent) {
  SmallVector<TypedAttr> operands;
  for (TypedAttr operand : op.getOperands()) {
    ErrorTreeOr<Attribute> value =
        parent->evaluator.concretizeParameterExpr(op.getLoc(), operand);
    if (value.isError())
      return value.takeError();
    operands.push_back(cast<TypedAttr>(value.takeValue()));
  }
  ErrorTreeOr<TypedAttr> result =
      parent->evaluator.evaluateFunction(func, operands);
  if (result.isError())
    return result.takeError();

  // Bind the result and erase the operation.
  parent->evaluator.setOrOverwriteParameterValue(op.getParamDecl(),
                                                 result.takeValue());
  op.erase();
  return {};
}

/// Complete processing of a generator user by resolving any bound result types
/// or parameters in the parent scope. This is the step that propagates result
/// parameters from the inner scope to the outer scope.
static std::optional<ErrorTree>
completeCallProcessing(KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls,
                       ExpansionTreeNode *thisNode,
                       ExpansionTreeNode *parentNode, Logger &logger) {

  // Add the callee's bindings to the parent of the call. This ensures that we
  // don't re-bind something we've already bound.
  for (const auto &[k, v] : thisNode->bindings) {
    auto &oldV = parentNode->bindings[k];
    assert(!oldV || oldV == v);
    oldV = v;
  }

  ErrorTreeOr<FuncOp> newCalleeFuncOr = thisNode->getFirstConcrete();
  if (newCalleeFuncOr.isError())
    return {};

  FuncOp newCalleeFunc = *newCalleeFuncOr;

  // If this is a `kgen.param.apply`, bind its result here.
  if (auto apply = dyn_cast<ParamApplyOp>(*user))
    return processParamApplyOp(apply, newCalleeFunc, parentNode);

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

  // Get the result params from the concretization of this node, if we have
  // them.
  auto concreteNodeOr = thisNode->getFirstConcreteNode();
  assert(!concreteNodeOr.isError() &&
         "This should be called from a concrete node in this case");

  // If we don't have the result parameters yet, then either no result
  // parameters are necessary, or we have another problem entirely wherein we
  // could not complete the callee's result parameter resolution at all - likely
  // meaning we're in an infinite recursive loop. Essentially, we came back to
  // the same combination of generator + input parameters without resolving the
  // result parameters yet.
  ArrayAttr resultParams = (*concreteNodeOr)->resultParams;
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
    LLVM_DEBUG(thisNode->logger << "Binding " << decl << " to " << bindValue
                                << "\n");
    parentNode->evaluator.setOrOverwriteParameterValue(decl, bindValue);
  }
  return {};
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
  ElaboratorImpl(
      mlir::SymbolTableAnalysis &analysis,
      ParameterCollector::Analysis &paramCache, TargetInfoAttr target,
      LLCL::Runtime &runtime, LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      EvaluatorExecutorFnRef evaluatorExecutorFn, bool enableSearch = false,
      bool testDiagnostics = false)
      : Elaborator(analysis, paramCache, target, runtime, map,
                   transformCache.copy(), regionCache.copy(),
                   evaluatorExecutorFn, enableSearch),
        root(analysis.getTopLevelOp<ModuleOp>(), IREvaluator(*this), alloc,
             logger),
        evalSemaphore(1), testDiagnostics(testDiagnostics) {}

  ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                      ArrayRef<TypedAttr> paramValues) override;

  std::optional<ErrorTree>
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
                                 ExpansionTreeNode *node, FuncOp func);

  /// Process a kgen.param.fork op. This will create a clone for each value of
  /// the parameter search, and will mark the parent as an error. This results
  /// in a very clean model where the parent of the current parent (a generator)
  /// will have its children be the successfully concretized parameter search
  /// nodes.
  std::optional<ErrorTree>
  processParamForkOp(ExpansionTreeNode *parent, ParamForkOp op,
                     ArrayRef<Operation *> remainingWorklist);

  /// Spawn a clone for kgen.param.fork. This creates a new FuncOp that is a
  /// sibling to the parent of the kgen.param.fork op. It replaces the
  /// kgen.param.fork with a param.declare to allow specialization to succeed.
  std::optional<ErrorTree>
  spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                      ExpansionTreeNode *forkParentNode,
                      ArrayRef<Operation *> remainingWorklist);

  /// Process a call op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  std::optional<ErrorTree>
  processCallOp(KGENCallOpInterface call, ExpansionTreeNode *parent,
                ArrayRef<Operation *> remainingWorklist);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  std::optional<ErrorTree> processGeneratorUser(
      KGENCallOpInterface user, SymbolConstantAttr calleeSymbol,
      ExpansionTreeNode *parent, ArrayRef<Operation *> remainingWorklist);

  /// Create a node for a callee and concretize it if needed.
  ErrorTreeOr<ExpansionTreeNode *>
  createCalleeNode(std::pair<Operation *, ArrayAttr> &&key,
                   SymbolConstantAttr calleeSymbol, ExpansionTreeNode *parent,
                   Operation *user);

  /// Resolve call input parameters - this is a complex function because calls
  /// can have regions. We take the body of those regions and put it into a
  /// generator with a specially prepared ParameterEvaluator scope and elaborate
  /// the region that way.
  ///
  /// Elaborating region parameters is the most non-local part of the elaborator
  /// - we have to interact with the module symbol table to put these regions
  /// into top-level generators.
  ErrorTreeOr<ArrayAttr>
  resolveCallInputParams(Operation *call, ExpansionTreeNode *parentNode,
                         ArrayRef<TypedAttr> inputValues);

  /// Process a param.declare.region op by creating a generator with the correct
  /// captures. We don't specialize the generator until the call-site because we
  /// don't know what the actual input parameters are supposed to be until then.
  std::optional<ErrorTree>
  processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                              ExpansionTreeNode *parent);

  /// Process a param.if op by evaluating the condition and elaborating and
  /// inlining only the branch that was taken. If one of the branches had an
  /// early return, this will split the block after the return and avoid
  /// elaborating the rest of the function.
  std::optional<ErrorTree> processParamIfOp(ParamIfOp op,
                                            ExpansionTreeNode *parent);

  /// Process a worklist of ops. Returns failure if the scope produced an error.
  LogicalResult processScope(ExpansionTreeNode *parentNode,
                             ArrayRef<Operation *> worklist);

  /// Specializes the generator at `genNode`. Essentially instantiates a new
  /// function with the same body, and specializes it. The new function is by
  /// definition the expansion tree child of this generator.
  std::optional<ErrorTree> specializeGenerator(ExpansionTreeNode *genNode);

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

  /// The allocator we use to allocate children in the expansion tree.
  llvm::SpecificBumpPtrAllocator<ExpansionTreeNode> alloc;
  /// The root of the expansion tree.
  ExpansionTreeNode root;
  /// Hash table to speed up lookups of generators in the expansion tree.
  DenseMap<std::pair<Operation *, ArrayAttr>, ExpansionTreeNode *>
      topLevelTrees;
  /// Hash table of known ParameterUseDefGraphs. This ensures we only compute a
  /// graph once for each generator. This is extra state generated by
  /// specializeGenerator that is *required for correctness* - this will cause
  /// issues with caching (though it would be easy to simply recompute) unless
  /// we create a ParametricNode or something we can use to store these in a
  /// proper data structure.
  DenseMap<GeneratorOp, std::unique_ptr<ParameterUseDefGraph>> knownGraphs;

  /// Evaluation semaphore - this ensures we benchmark one thing at a time. We
  /// initialize it to 1 so that the first thing to evaluate something doesn't
  /// block.
  // TODO (#7826): Make this a NamedSemaphore so it's unique across processes
  //               too.
  Semaphore evalSemaphore;

  /// If this is true, emit diagnostics for certain conditions that are
  /// interesting to test for.
  bool testDiagnostics;

  /// Remove parameter declare regions after generator elaboration.
  SmallVector<OwningOpRef<ParamDeclareRegionOp>> paramDeclareRegionOps;
};
} // namespace

//===----------------------------------------------------------------------===//
// finalizeAndVerifyFunction
//===----------------------------------------------------------------------===//
void ElaboratorImpl::finalizeAndVerifyFunction(
    mlir::SymbolTableAnalysis &analysis, ExpansionTreeNode *node, FuncOp func) {
  TimeTraceScope<> traceScope("finalizeAndVerifyFunction");
  // Erase any unreachable blocks that might have arisen.
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
  auto funcItf = lookup<FuncInterface>(symbolRef);
  if (auto func = dyn_cast<FuncOp>(funcItf.getOperation()))
    return func;

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ExpansionTreeNode *node = topLevelTrees.lookup({funcItf, vals});
  // If the node has already been elaborated, just use that result.
  if (node && node->isConcrete())
    return node->getFirstConcrete();

  // Otherwise, if the node doesn't exist, then create a new one.
  if (!node) {
    node =
        ExpansionTreeNode::create(funcItf, vals, &root, IREvaluator(*this), 0);
    topLevelTrees[{funcItf, vals}] = node;
  }

  for (auto [decl, value] :
       llvm::zip(cast<DeclInterface>(*funcItf).getInputParams(), vals))
    node->evaluator.setOrOverwriteParameterValue(decl, value);

  if (auto gen = dyn_cast<GeneratorOp>(funcItf.getOperation())) {
    if (auto err = specializeGenerator(node))
      return std::move(*err);
  }
  return node->getFirstConcrete();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getAllConcreteFunctions
//===----------------------------------------------------------------------===//

std::optional<ErrorTree>
ElaboratorImpl::getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                                        ArrayRef<TypedAttr> paramValues,
                                        std::vector<FuncOp> &funcs) {
  auto funcItf = lookup<FuncInterface>(symbolRef);
  if (auto func = dyn_cast<FuncOp>(funcItf.getOperation())) {
    funcs.push_back(func);
    return std::nullopt;
  }

  SmallVector<Attribute> inputParams;
  for (TypedAttr value : paramValues)
    inputParams.push_back(cast<Attribute>(value));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ExpansionTreeNode *node = topLevelTrees.lookup({funcItf, vals});
  // If the node has already been elaborated, just use that result.
  if (node && node->isConcrete()) {
    node->getAllConcrete(funcs);
    return std::nullopt;
  }

  // Otherwise, if the node doesn't exist, then create a new one.
  if (!node) {
    node =
        ExpansionTreeNode::create(funcItf, vals, &root, IREvaluator(*this), 0);
    topLevelTrees[{funcItf, vals}] = node;
  }

  for (auto [decl, value] :
       llvm::zip(cast<DeclInterface>(*funcItf).getInputParams(), vals))
    node->evaluator.setOrOverwriteParameterValue(decl, value);

  if (auto gen = dyn_cast<GeneratorOp>(funcItf.getOperation())) {
    if (auto err = specializeGenerator(node))
      return std::move(*err);
  }
  node->getAllConcrete(funcs);
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamForkOp
//===----------------------------------------------------------------------===//

/// Process a kgen.param.fork op.
std::optional<ErrorTree>
ElaboratorImpl::processParamForkOp(ExpansionTreeNode *parent, ParamForkOp op,
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
    if (auto err = spawnParamForkClone(op, value, parent, remainingWorklist)) {
      errors.push_back(std::move(*err));
      continue;
    }

    // If search is disabled, break after the first successful parameter.
    atLeastOneSuccessful = true;
    if (!enableSearch)
      break;
  }

  // If we don't have at least one successful candidate, fail.
  if (!atLeastOneSuccessful)
    return ErrorTree(op.getLoc(), "some expansions failed", errors);

  // The parent has to be deleted.
  parent->setToError(ErrorTree(op.getLoc(), "param fork base node"));
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::spawnParamForkClone
//===----------------------------------------------------------------------===//

/// Spawn a clone from a kgen.param.fork op.
std::optional<ErrorTree>
ElaboratorImpl::spawnParamForkClone(ParamForkOp forkOp, Attribute value,
                                    ExpansionTreeNode *forkParentNode,
                                    ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Spawning ParamForkClone for '", value, "'");

  // Start by cloning the current WIP func to a new copy of it.
  IRMapping map;
  auto newFunc = cast<FuncOp>(forkParentNode->op->clone(map));
  // Insert into the symbol table - this will also unique the name for us.
  analysis.getTopLevelSymbolTable().insert(newFunc,
                                           ++forkParentNode->op->getIterator());

  // Hook this new clone up correctly.
  auto newFuncNode = ExpansionTreeNode::create(
      newFunc, forkParentNode->inputParams, forkParentNode->parent,
      forkParentNode->evaluator, forkParentNode->expansionDepth,
      forkParentNode->paramGraph->copy(map));

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
  if (failed(processScope(newFuncNode, remaining)))
    return newFuncNode->error->copy();

  finalizeAndVerifyFunction(analysis, newFuncNode, newFunc);
  return std::nullopt;
}

ErrorTreeOr<ExpansionTreeNode *>
ElaboratorImpl::createCalleeNode(std::pair<Operation *, ArrayAttr> &&key,
                                 SymbolConstantAttr calleeSymbol,
                                 ExpansionTreeNode *parent, Operation *user) {
  // Find the tree node that corresponds to the thing we're calling.
  ExpansionTreeNode *node = topLevelTrees.lookup(key);
  // If we haven't found this callee yet, we have to add it to the tree!
  if (!node) {
    // Use the parent of the call to show the expansion depth, and inherit the
    // evaluator from the root.
    node =
        ExpansionTreeNode::create(key.first, key.second, &root,
                                  IREvaluator(*this), parent->expansionDepth);
    if (node->error)
      return ErrorTreeOr<ExpansionTreeNode *>(std::move(node->error).value());

    topLevelTrees[key] = node;

    // Set the region parameters on the callee by performing the name rebind and
    // handling any parameter captures. This can't be pushed into
    // specializeGenerator below because it requires passing specific values
    // from the caller's node to the callee's node.
    for (TypedAttr bind : calleeSymbol.getParamValues()) {
      ErrorTreeOr<Attribute> attrValue =
          parent->evaluator.concretizeParameterExpr(user->getLoc(), bind);
      if (attrValue.isError())
        return ErrorTreeOr<ExpansionTreeNode *>(attrValue.takeError());

      Attribute value = attrValue.takeValue();
      if (auto region = dyn_cast<RegionAttr>(value)) {
        // Set the known regions correctly by updating the name.
        ParameterUseDefGraph *graph =
            parent->knownRegions.lookup(region.getRegionName());
        node->knownRegions[region.getRegionName()] = graph;

        // Do the same for any potential parameter captures.
        for (ParamDeclRefAttr param : graph->usesFromAbove) {
          ErrorTreeOr<Attribute> attr =
              parent->evaluator.concretizeParameterExpr(user->getLoc(), param);
          if (attr.isError())
            return ErrorTreeOr<ExpansionTreeNode *>(attr.takeError());

          node->evaluator.setOrOverwriteParameterValue(param.getName(), *attr);
        }
      }
    }

    if (isa<GeneratorOp>(key.first)) {
      if (auto err = specializeGenerator(node))
        return ErrorTreeOr<ExpansionTreeNode *>(std::move(err).value());
    }
  }
  return ErrorTreeOr<ExpansionTreeNode *>(node);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processGeneratorUser
//===----------------------------------------------------------------------===//

/// Process a generator user like a call.
std::optional<ErrorTree> ElaboratorImpl::processGeneratorUser(
    KGENCallOpInterface user, SymbolConstantAttr calleeSymbol,
    ExpansionTreeNode *parent, ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Processing Generator User");
  LLVM_DEBUG(logger.logOp("User", user));

  assert(remainingWorklist.empty() || remainingWorklist.front() != user);

  // Add in the mapping for parameters in the calls.
  auto resolvedCallParamsOr =
      resolveCallInputParams(user, parent, calleeSymbol.getParamValues());
  if (resolvedCallParamsOr.isError())
    return resolvedCallParamsOr.takeError();
  ArrayAttr inputParamKey = *resolvedCallParamsOr;

  // Lookup the callee.
  FuncInterface calleeOp =
      cast<FuncInterface>(lookup(calleeSymbol.getSymbol()));
  if (!calleeOp) {
    return ErrorTree(user.getLoc(), "could not find callee '" +
                                        mlir::debugString(calleeSymbol) + "'");
  }

  LLVM_DEBUG({
    logger.logOp("Callee", calleeOp);
    logger << "Input Params: " << inputParamKey << "\n";
  });

  // If we already have a binding for this, we're done.
  ArrayRef<ParamDeclAttr> decls = user.getParamDecls();
  auto found = parent->bindings.find({calleeOp, inputParamKey});
  if (found != parent->bindings.end()) {
    LLVM_DEBUG(
        found->getSecond()->print(logger.scope("Result: Existing Binding")));
    return completeCallProcessing(user, decls, found->getSecond(), parent,
                                  logger);
  }

  // Find the tree node that corresponds to the thing we're calling.
  ErrorTreeOr<ExpansionTreeNode *> maybeCalleeNode =
      createCalleeNode({calleeOp, inputParamKey}, calleeSymbol, parent, user);
  if (maybeCalleeNode.isError())
    return maybeCalleeNode.takeError();
  ExpansionTreeNode *calleeNode = maybeCalleeNode.takeValue();
  LLVM_DEBUG(calleeNode->print(logger));

  // Complete processing for all the leaves of this subtree.
  std::vector<ExpansionTreeNode *> concrete;
  calleeNode->getAllConcreteNodes(concrete);

  // If the concrete thing has bindings, they must be consistent with the
  // parent's bindings for us to consider it. Remove nodes from the vector that
  // have bindings that are inconsistent with the parent.
  auto newEnd = llvm::remove_if(concrete, [&](ExpansionTreeNode *node) {
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
    calleeNode->collectSubtreeErrors(out);
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
    Operation *newOp = parent->op->clone(map);
    // Insert it into the symbol table.
    analysis.getTopLevelSymbolTable().insert(newOp,
                                             ++parent->op->getIterator());

    auto _ = logger.scope("New Multi-Versioning Op");
    logger.logOp("Op", newOp);
    LLVM_DEBUG(c->print(logger << "Concrete Implementation "));

    // This is a sibling to the parent, and it clones the parent's evaluator.
    auto newNode = ExpansionTreeNode::create(
        newOp, parent->inputParams, parent->parent, parent->evaluator,
        parent->expansionDepth, parent->paramGraph->copy(map));
    newNode->bindings = parent->bindings;
    // Bind this concrete impl to this callee for this node.
    newNode->bindings[{calleeOp, inputParamKey}] = c;

    if (std::optional<ErrorTree> err = completeCallProcessing(
            cast<KGENCallOpInterface>(map.lookup(user.getOperation())), decls,
            c, newNode, logger))
      return err;

    LLVM_DEBUG(newNode->print(logger << "New Op "));

    // We have to finish specializing this thing now. Map to the new ops and
    // process the remaining scope.
    auto remaining = map_to_vector(
        remainingWorklist, [&](Operation *op) { return map.lookup(op); });

    // Process the rest of the worklist in this new scope. If the scope
    // processing failed, do nothing.
    if (succeeded(processScope(newNode, remaining)))
      finalizeAndVerifyFunction(analysis, newNode, cast<FuncOp>(newOp));
  }

  // Bind this concrete impl to this callee for this node.
  parent->bindings[{calleeOp, inputParamKey}] = concrete.front();

  // Call completeGeneratorUserProcessing on the first concrete thing. This will
  // flow nested bindings upward correctly.
  return completeCallProcessing(user, decls, concrete.front(), parent, logger);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::resolveCallInputParams
//===----------------------------------------------------------------------===//

/// Resolve input params on a call_param op.
ErrorTreeOr<ArrayAttr>
ElaboratorImpl::resolveCallInputParams(Operation *call,
                                       ExpansionTreeNode *parentNode,
                                       ArrayRef<TypedAttr> inputValues) {
  LLVM_DEBUG(logger.logOp("Resolving Call Input Params", call);
             logger << " with input bindings: ";
             llvm::interleaveComma(inputValues, logger); logger << "\n");

  SmallVector<Attribute> boundInputParams;
  for (TypedAttr param : inputValues) {
    // Fold the parameter expression in this context to a simple constant.
    ErrorTreeOr<Attribute> valueOr =
        parentNode->evaluator.concretizeParameterExpr(call->getLoc(), param);
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
std::optional<ErrorTree>
ElaboratorImpl::processCallOp(KGENCallOpInterface call,
                              ExpansionTreeNode *parent,
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
  ParameterUseDefGraph *regionGraph =
      parent->knownRegions.lookup(decl.getRegionName());
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

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamDeclareRegionOp
//===----------------------------------------------------------------------===//

/// Process a param.declare.region by creating a generator for the contained
/// region.
std::optional<ErrorTree>
ElaboratorImpl::processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                                            ExpansionTreeNode *parent) {
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
      parent->paramGraph->nestedScopes.find(&regionDecl.getBodyRegion());
  assert(found != parent->paramGraph->nestedScopes.end() &&
         "must have a nested region");
  LLVM_DEBUG(logger << "Storing known region: " << regionName << "\n");
  parent->knownRegions[regionName] = &found->getSecond();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamIfOp
//===----------------------------------------------------------------------===//

std::optional<ErrorTree>
ElaboratorImpl::processParamIfOp(ParamIfOp op, ExpansionTreeNode *parent) {
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

  auto foundNestedScope = parent->paramGraph->nestedScopes.find(toProcess);
  if (foundNestedScope == parent->paramGraph->nestedScopes.end())
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
  ParameterUseDefGraph &paramGraph = *parent->paramGraph;
  auto eraseIfScopes = [op](ParameterUseDefGraph &graph) mutable {
    // Erase any regions from the nested scopes that belong either to this op or
    // under this op.
    for (auto [r, _] : graph.nestedScopes)
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
      logger.logOp("param.if parent scope (after processing)", parent->op));
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processScope
//===----------------------------------------------------------------------===//

LogicalResult ElaboratorImpl::processScope(ExpansionTreeNode *parentNode,
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

    std::optional<ErrorTree> result = std::nullopt;
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

    // If we have an error, log it and set the parent node to an error. This
    // will perform any required cleanup.
    if (result) {
      LLVM_DEBUG(logger.scope("Result: Failure") << result->getError());
      parentNode->setToError(std::move(*result));
      return failure();
    }

    // If the parent node was set to error, then just bail.
    if (parentNode->error)
      return failure();
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

std::optional<ErrorTree>
ElaboratorImpl::specializeGenerator(ExpansionTreeNode *genNode) {
  GeneratorOp generator = cast<GeneratorOp>(genNode->op);

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
  IREvaluator &evaluator = genNode->evaluator;
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setOrOverwriteParameterValue(decl, val);

  // If the generator's constraints don't satisfy, set an error and move on.
  if (auto err =
          KGEN::evaluateConstraints(generator.getConstraints(), evaluator)) {
    genNode->setToError(std::move(err.value()));
    return std::nullopt;
  }

  // If this generator node is already concrete and has no error, don't
  // re-concretize.
  if (!genNode->expansions.empty() && !genNode->error.has_value()) {
    LLVM_DEBUG(logger << "Result: Generator has already been specialized (node "
                         "is known to be concrete)\n");
    if (testDiagnostics)
      generator.emitRemark("Generator has already been specialized");
    return std::nullopt;
  }

  // Check if we have a FuncOp parented under the top level module.
  StringAttr mangledName = mangleParameterValues(generator, inputParamValues);
  if (Operation *op = lookup(mangledName)) {
    // See if we can find this node (func + input params) in the root.
    // The top level thing will be a generator, so check its expansions for a
    // node that has this op and these input parameters.
    auto found = llvm::find_if(root.expansions, [&](ExpansionTreeNode *e) {
      return e->op == op && e->inputParams == genNode->inputParams;
    });
    // If we found the node in the root's expansion list, then we should not
    // re-specialize.
    if (found != root.expansions.end()) {
      // Great, already specialized this node - we're all done.
      LLVM_DEBUG(logger << "Result: Generator has already been specialized "
                           "(found concrete func under the root)\n");

      // If we have the node and its parentage is incorrect, then take it as a
      // child.
      if ((*found)->parent != genNode)
        genNode->takeExpansion(*found);

      if (testDiagnostics)
        generator.emitRemark("Generator has already been specialized");
      return std::nullopt;
    }
  }

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

  // The node for this new func is simply the child of the node for the
  // generator.
  auto newFuncNode =
      ExpansionTreeNode::create(newFunc, genNode->inputParams, genNode,
                                evaluator, genNode->expansionDepth);
  newFuncNode->knownRegions = genNode->knownRegions;
  // Map from the generator to the new function for the parameter graph copy.
  map.map(generator.getOperation(), newFunc.getOperation());
  // Copy over the parameter use-def graph for this clone.
  newFuncNode->paramGraph = genNodeGraph->copy(map);
  ParameterUseDefGraph &uses = newFuncNode->paramGraph.value();

  // Inflate the function and then specialize it.
  asyncMap.mapChained(
      newFunc, [newFunc, regionCache = regionCache.copy()](auto ch) mutable {
        return Cache::inflateOp(newFunc, std::move(regionCache), std::move(ch));
      });
  if (auto err = asyncMap.await(newFunc))
    return ErrorTree(newFunc.getLoc(), err.takeError());

  // Kick off the expansion for the new function.
  FuncOp func = cast<FuncOp>(newFuncNode->op);

  auto funcScope = logger.scope("Specializing Function: @", func.getName());
  logger.logOp("Function", func);

  std::vector<Operation *> opsToRewrite;
  collectOpsToProcess(&func.getBodyRegion(), uses, opsToRewrite);
  opsToRewrite.push_back(newFunc);
  llvm::append_range(opsToRewrite, uses.paramOps);

  // Process the worklist. Only finalize the function if this succeeded.
  if (succeeded(processScope(newFuncNode, opsToRewrite)))
    finalizeAndVerifyFunction(analysis, newFuncNode, func);

  return std::nullopt;
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
  for (auto gen : primaryGenerators) {
    LLVM_DEBUG(logger.logOp("Elaborating primary generator", gen));
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    ExpansionTreeNode *generatorNode =
        topLevelTrees.lookup({gen, emptyInputParamKey});
    if (!generatorNode) {
      generatorNode = ExpansionTreeNode::create(gen, emptyInputParamKey, &root,
                                                IREvaluator(*this), 0);
      topLevelTrees[{gen, emptyInputParamKey}] = generatorNode;
    }

    // Now we can begin to construct the expansion tree rooted at this
    // generator.
    if (auto err = specializeGenerator(generatorNode)) {
      err->emit([](Location loc) { return mlir::emitError(loc); });
      return failure();
    }
  }

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  DenseMap<StringAttr, StringAttr> funcsToRename;
  bool failed = !root.isConcrete();
  // We have to walk all the generators in the module because we have to rename
  // each one to its first implementation.
  for (auto gen : theModule.getOps<GeneratorOp>()) {
    ExpansionTreeNode *genNode =
        topLevelTrees.lookup({gen, emptyInputParamKey});
    if (!genNode)
      continue;

    // If this primary generator failed (or we already knew we failed), then the
    // whole thing has failed. Just stop processing at this point.
    if (failed || !genNode->isConcrete()) {
      if (!llvm::is_contained(primaryGenerators, gen)) {
        continue;
      } else {
        failed = true;
        break;
      }
    }

    // Add all concrete functions, and rename the first one.
    std::vector<FuncOp> concreteFuncs;
    genNode->getAllConcrete(concreteFuncs);
    for (auto c : concreteFuncs)
      analysis.getTopLevelSymbolTable().insert(c, genNode->op->getIterator());

    funcsToRename[concreteFuncs.front().getNameAttr()] = genNode->getNameAttr();
  }

  // Trim the expansion tree and erase ops we don't need/want.
  DenseSet<Operation *> toErase;

  // If we have failed, emit your errors and return failure.
  if (failed) {
    for (auto gen : primaryGenerators) {
      ExpansionTreeNode *genNode =
          topLevelTrees.lookup({gen, emptyInputParamKey});
      assert(genNode && "We must have a node for a primary generator");
      genNode->trimFailedExpansions(toErase);
      if (genNode->error)
        genNode->error->emit([](Location loc) { return mlir::emitError(loc); });
    }
    return failure();
  }

  LLVM_DEBUG(root.print(logger.scope("Expansion Tree")));
  root.trimFailedExpansions(toErase);
  LLVM_DEBUG(root.print(logger.scope("Trimmed Expansion Tree")));

  for (Operation *op : toErase)
    op->erase();

  SymbolTable &symtab = analysis.getTopLevelSymbolTable();
  for (Operation &op : llvm::make_early_inc_range(theModule.getOps())) {
    if (isa<GeneratorOp>(op)) {
      symtab.erase(&op);
      continue;
    }

    /// Non viable funcs or inlined funcs will be left with an invalid body.
    /// Remove them at the end of elaboration.
    if (auto func = dyn_cast<FuncOp>(op)) {
      // Make sure all funcs are inflated at the end of this, even if they
      // didn't participate in elaboration.
      asyncMap.map(func, Cache::inflateOp(func, regionCache.copy(),
                                          asyncMap.getChain(func)));
    }
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
        // TODO: We should upstream something for this.
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

  // Update the debug info for everything now that we've done renaming etc.
  root.updateDebugInfo();

  if (root.isConcrete())
    LLVM_DEBUG(logger.logOp("Finished successfully", theModule));

  // We were only successful if the root could be concretized.
  return success(root.isConcrete());
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
                                     bool enableSearch, bool testDiagnostics) {
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
  ElaboratorImpl impl(symtab, paramCache, target, transformCache->getRuntime(),
                      asyncMap, transformCache.copy(), regionCache.copy(),
                      evaluatorExecutorFn, enableSearch, testDiagnostics);
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

    if (failed(elaborateGenerators(analysis, paramCache, *rt, target,
                                   primaryGenerators, evaluatorExecutorFn,
                                   shouldDoSearch, testDiagnostics)))
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
