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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Support/Semaphore.h"
#include "SelectFastestFunction.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/STLExtras.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kgen-elaborator"

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
      os << std::forward<Arg>(arg);
      return os;
    }
  };

  Logger()
#ifdef MODULAR_DEBUG
      : os(llvm::dbgs())
#else
      : os(llvm::nulls())
#endif
  {
  }

  /// Start a new logging scope, using the provided arguments to form a message
  /// on the title line of the scope.
  template <typename... TitleLineArgs>
  DelimitedScope scope(TitleLineArgs... titleLineArgs) {
#ifdef MODULAR_DEBUG
    LLVM_DEBUG({
      ((os << titleLineArgs), ...);
      return DelimitedScope(os, " {\n", "}\n");
    });
#endif
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
  static ExpansionTreeNode *create(Operation *op, ArrayAttr inputParams,
                                   ExpansionTreeNode *parent,
                                   const IREvaluator &evaluator,
                                   unsigned expansionDepth);

  /// Create a node nested under the direct parent, and set its expansion depth.
  static ExpansionTreeNode *create(Operation *op, ArrayAttr inputParams,
                                   ExpansionTreeNode *parent) {
    return ExpansionTreeNode::create(op, inputParams, parent, parent->evaluator,
                                     parent->expansionDepth);
  }

private:
  /// Construct an expansion tree node. The node adds itself to its parent's
  /// children list, and inherits its parent's evaluator.
  ExpansionTreeNode(Operation *op, ArrayAttr inputParams,
                    ExpansionTreeNode *parent, const IREvaluator &evaluator,
                    unsigned expansionDepth);

public:
  /// Construct an expansion tree node without a parent. This should only be
  /// used for the tree root.
  ExpansionTreeNode(ModuleOp op, const IREvaluator &evaluator,
                    llvm::SpecificBumpPtrAllocator<ExpansionTreeNode> &alloc,
                    Logger &logger)
      : op(op), evaluator(evaluator), alloc(alloc), expansionDepth(0),
        logger(logger) {}

  /// Return the name of the operation this node represents.
  StringAttr getNameAttr() { return op.getNameAttr(); }

  /// Given an operation and its input params, find the node in the subtree that
  /// has the correct operation and input params. The max depth is related to
  /// the type of the thing you're looking for - 1 for interfaces, 2 for
  /// generators, and 3 for concretized functions.
  ExpansionTreeNode *find(Operation *op, ArrayAttr inputParams);

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
    expansions.push_back(*found);
  }

  /// Update the debug info on the concretization of the current node. This
  /// means, if there are child nodes, update their debug info. This should be
  /// called after the operations are renamed at the end of the pass.
  void updateDebugInfo();

  /// Print this tree to the provided indented stream. This preserves any
  /// indentation provided by the caller to make it possible to nest things
  /// nicely.
  void print(mlir::raw_indented_ostream &os);

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
ExpansionTreeNode *ExpansionTreeNode::create(Operation *op,
                                             ArrayAttr inputParams,
                                             ExpansionTreeNode *parent,
                                             const IREvaluator &evaluator,
                                             unsigned expansionDepth) {
  auto *out = new (parent->alloc.Allocate())
      ExpansionTreeNode(op, inputParams, parent, evaluator, expansionDepth + 1);
  assert(out->distanceFromRoot() <= 3 &&
         "Should have at most 3 hops to the root");
  parent->expansions.push_back(out);
  return parent->expansions.back();
}

ExpansionTreeNode::ExpansionTreeNode(Operation *op, ArrayAttr inputParams,
                                     ExpansionTreeNode *parent,
                                     const IREvaluator &evaluator,
                                     unsigned expansionDepth)
    : op(op), inputParams(inputParams), evaluator(evaluator), parent(parent),
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

ExpansionTreeNode *ExpansionTreeNode::find(Operation *op,
                                           ArrayAttr inputParams) {
  // Compute the max depth automatically based on the op's type.
  unsigned maxDepth = 0;
  if (isa<GeneratorInterfaceOp>(op))
    maxDepth = 1;
  else if (auto gen = dyn_cast<GeneratorOp>(op))
    maxDepth = gen.getImplements() ? 2 : 1;
  else if (isa<FuncOp>(op))
    maxDepth = 3;

  // Do a depth-limited DFS search of the children.
  std::stack<std::pair<ExpansionTreeNode *, unsigned>> toExplore;
  toExplore.emplace(this, 0);
  for (auto &child : expansions)
    toExplore.emplace(child, 1);

  while (!toExplore.empty()) {
    auto [front, depth] = toExplore.top();
    toExplore.pop();
    // If we found it, great. Return it.
    if (front->op == op && front->inputParams == inputParams)
      return front;

    // If we're going too deep in the stack, then don't explore any further
    // children.
    if (depth >= maxDepth)
      continue;

    // Otherwise, put the children of this child onto the stack.
    for (auto child : front->expansions)
      toExplore.emplace(child, depth + 1);
  }

  return nullptr;
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
    toErase.insert(op);
    return;
  }
  auto _ = logger.scope("Trimming Failed Children");
  LLVM_DEBUG(logger.logOp("Op", op));

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
  if (expansions.empty() && error) {
    tree.addCause(error->copy());
  }

  for (auto &ch : expansions)
    ch->collectSubtreeErrors(tree);
}

void ExpansionTreeNode::updateDebugInfo() {
  for (auto &child : expansions)
    child->updateDebugInfo();

  if (!expansions.empty())
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

void ExpansionTreeNode::print(mlir::raw_indented_ostream &os) {
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
  }

  // Print the bindings.
  {
    auto _ = os.scope("Bindings: {\n", "}\n");
    for (const auto &[_, bind] : bindings) {
      if (bind != this)
        bind->print(os);
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
// processParamConstantOp
//===----------------------------------------------------------------------===//

/// Process a param.constant op by concretizing its parameter value and setting
/// its value attr.
static std::optional<ErrorTree> processParamConstantOp(IREvaluator &evaluator,
                                                       ParamConstantOp op) {
  // ParamConstantOp projects a parameter expression into an SSA value.  We can
  // eventually lower this into lower level operators in the target set, but
  // for now we just simplify their operand.
  auto errorOrValue =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getValue());
  if (errorOrValue.isError())
    return errorOrValue.takeError();

  auto errorOrType =
      evaluator.concretizeParameterExpr(op.getLoc(), op.getType());
  if (errorOrType.isError())
    return errorOrType.takeError();

  op.getResult().setType(errorOrType.takeValue());
  op.setValueAttr(errorOrValue.takeValue());
  return std::nullopt;
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

  auto resultInt = dyn_cast<IntegerAttr>(errorOrValue.takeValue());
  if (!resultInt || resultInt.getValue().getBitWidth() != 1)
    return ErrorTree(op.getLoc(),
                     "constraint evaluation didn't return true or false");
  // If the constraint evaluated to zero then the assert fails.
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
// completeGeneratorUserProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a generator user by resolving any bound result types
/// or parameters in the parent scope. This is the step that propagates result
/// parameters from the inner scope to the outer scope.
static void completeGeneratorUserProcessing(KGENCallOpInterface user,
                                            ArrayRef<ParamDeclAttr> decls,
                                            ExpansionTreeNode *thisNode,
                                            ExpansionTreeNode *parentNode) {
  // Add the callee's bindings to the parent of the call. This ensures that we
  // don't re-bind something we've already bound.
  for (const auto &[k, v] : thisNode->bindings) {
    auto &oldV = parentNode->bindings[k];
    assert(!oldV || oldV == v);
    oldV = v;
  }

  ErrorTreeOr<FuncOp> newCalleeFuncOr = thisNode->getFirstConcrete();
  if (newCalleeFuncOr.isError())
    return;

  FuncOp newCalleeFunc = *newCalleeFuncOr;

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (auto result : user->getResultTypes()) {
    auto typeOr =
        parentNode->evaluator.concretizeParameterExpr(user.getLoc(), result);
    if (typeOr.isError()) {
      thisNode->error = typeOr.takeError();
      return;
    }

    resultTypes.push_back(typeOr.takeValue());
  }

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  mlir::IRRewriter b{OpBuilder(user)};
  if (isa<CallOp, CallParamOp>(user)) {
    b.replaceOpWithNewOp<CallOp>(
        user, resultTypes,
        SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
            newCalleeFunc.getSignature()),
        ArrayRef<ParamDeclAttr>(), user->getOperands());
  } else if (isa<AddressOfOp>(user)) {
    b.replaceOpWithNewOp<AddressOfOp>(
        user, resultTypes.front(),
        SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
            newCalleeFunc.getSignature()),
        ArrayRef<ParamDeclAttr>());
  } else {
    // Update the interface in-place.
    auto itf = cast<GeneratorInterfaceOp>(user);
    itf.setEvaluatorAttr(SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(newCalleeFunc.getSymNameAttr()),
        newCalleeFunc.getSignature()));
  }

  // Get the result params from the concretization of this node, if we have
  // them.
  auto concreteNodeOr = thisNode->getFirstConcreteNode();
  assert(!concreteNodeOr.isError() &&
         "This should be called from a concrete node in this case");
  auto resultParams = (*concreteNodeOr)->resultParams;
  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] : llvm::zip(decls, resultParams)) {
    LLVM_DEBUG(thisNode->logger << "Binding " << decl << " to " << bindValue
                                << "\n");
    parentNode->evaluator.setOrOverwriteParameterValue(decl, bindValue);
  }
}

//===----------------------------------------------------------------------===//
// completeReturnProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a ReturnOp by binding its result parameters to the
/// node's result parameters.
static std::optional<ErrorTree>
completeReturnProcessing(Logger &logger, ReturnOp returnOp,
                         ExpansionTreeNode *parent) {
  LLVM_DEBUG(logger.logOp("Processing ReturnOp", returnOp));
  SmallVector<Attribute> resultParams;
  for (auto param : returnOp.getParameters()) {
    auto concreteOr =
        parent->evaluator.concretizeParameterExpr(returnOp.getLoc(), param);
    if (concreteOr.isError())
      return concreteOr.takeError();

    resultParams.push_back(param);
  }
  parent->resultParams = ArrayAttr::get(returnOp.getContext(), resultParams);

  // Clear the parameters from this function and its return before we try to
  // verify.
  auto func = returnOp->getParentOfType<FuncOp>();
  assert(func && "must call completeReturnProcessing from a FuncOp.");
  func.setSignature(
      SignatureType::get(func.getInputParamDeclsAttr(),
                         ParamDeclArrayAttr::get(func.getContext(), {}),
                         func.getFunctionType(), func.getConventions()));
  returnOp.setParameters({});
  return std::nullopt;
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
  } else if (auto listConstant = dyn_cast<ListAttr>(value)) {
    os << '[';
    llvm::interleave(
        listConstant.getValues(), os,
        [&](TypedAttr value) { printParameterValue(value, os); }, ",");
    os << ']';
  } else {
    value.print(os, /*elideType=*/true);
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

  auto inputParamDecls = generator.getInputParamDeclsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterValue(value, os);
  }
  return b.getStringAttr(result);
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
      mlir::SymbolTableAnalysis &analysis, TargetInfoAttr target,
      LLCL::Runtime &runtime, LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      bool enableSearch = false)
      : Elaborator(analysis, target, runtime, map, transformCache.copy(),
                   regionCache.copy(), enableSearch),
        root(analysis.getTopLevelOp<ModuleOp>(), IREvaluator(*this), alloc,
             logger),
        evalSemaphore(1) {}

  ErrorTreeOr<FuncOp>
  getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                      ArrayRef<ParamBindAttr> paramValues) override;

  std::optional<ErrorTree>
  getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                          ArrayRef<ParamBindAttr> paramValues,
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

  /// Process a param search op. This will create a clone for each value of the
  /// parameter search, and will mark the parent as an error. This results in a
  /// very clean model where the parent of the current parent (a generator) will
  /// have its children be the successfully concretized parameter search nodes.
  std::optional<ErrorTree>
  processParamSearchOp(ExpansionTreeNode *parent, ParamSearchOp op,
                       ArrayRef<Operation *> remainingWorklist);

  /// Spawn a clone for param.search. This creates a new FuncOp that is a
  /// sibling to the parent of the param.search op. It replaces the param.search
  /// with a param.declare to allow specialization to succeeed.
  std::optional<ErrorTree>
  spawnParamSearchClone(ParamSearchOp searchOp, Attribute value,
                        ExpansionTreeNode *searchParentNode,
                        ArrayRef<Operation *> remainingWorklist);

  /// Process a generator user. In general, this is anything that can call into
  /// a generator and might therefore need to be multi-versioned.
  std::optional<ErrorTree>
  processGeneratorUser(ArrayRef<ParamDeclAttr> decls, KGENCallOpInterface user,
                       SymbolConstantAttr calleeSymbol,
                       ExpansionTreeNode *parent,
                       ArrayRef<Operation *> remainingWorklist);

  /// Resolve call input parameters - this is a complex function because calls
  /// can have regions. We take the body of those regions and put it into a
  /// generator with a specially prepared ParameterEvaluator scope and elaborate
  /// the region that way.
  ///
  /// Elaborating region parameters is the most non-local part of the elaborator
  /// - we have to interact with the module symbol table to put these regions
  /// into top-level generators.
  ErrorTreeOr<ArrayAttr> resolveCallInputParams(KGENCallOpInterface call,
                                                ExpansionTreeNode *parentNode,
                                                ParamBindArrayAttr inputValues);

  /// Process a call_param op by binding any necessary input parameters from the
  /// symbol or the call and passing them on to processGeneratorUser.
  std::optional<ErrorTree>
  processCallParamOp(CallParamOp call, ExpansionTreeNode *parent,
                     ArrayRef<Operation *> remainingWorklist);

  /// Process a param.declare.region op by creating a generator with the correct
  /// captures. We don't specialize the generator until the call-site because we
  /// don't know what the actual input parameters are supposed to be until then.
  std::optional<ErrorTree>
  processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                              ExpansionTreeNode *parent);

  /// Process a worklist of ops.
  void processScope(ExpansionTreeNode *parentNode,
                    ArrayRef<Operation *> worklist);

  /// Specializes the function at `funcNode` with parameter declarations
  /// `paramDecls`. Generates a DAG of ops that use parameters and calls
  /// processScope on the list.
  std::optional<ErrorTree>
  specializeFunction(ExpansionTreeNode *funcNode,
                     ArrayRef<ParamDeclAttr> paramDecls);

  /// Specializes the generator at `genNode`. Essentially instantiates a new
  /// function with the same body, and calls `specializeFunction` on it. The new
  /// function is by definition the expansion tree child of this generator.
  std::optional<ErrorTree> specializeGenerator(ExpansionTreeNode *genNode);

  /// Specializes the interface at `itfNode`. The list of interface
  /// implementations are provided because we have precomputed this list
  /// elsewhere, and we can simply provide it. The specialization happens by
  /// getting or creating the node belonging to the generator, and then simply
  /// specializing the generator. Since the generator is a child of the
  /// interface, the concrete implementations of the interface are exactly its
  /// concrete children.
  std::optional<ErrorTree>
  specializeInterface(ExpansionTreeNode *itfNode,
                      ArrayRef<GeneratorOp> interfaceImpls);

  /// Given a list of primary generators (i.e. generators with no input
  /// parameters), run the elaborator. This will generate an expansion tree
  /// rooted on the module with base nodes for each primary generator. Once
  /// specialization completes we will be able to collect all the concrete
  /// implementations for each primary generator and handle any renaming or
  /// fixup that needs to happen to produce the output IR.
  LogicalResult run(ArrayRef<GeneratorOp> primaryGenerators);

private:
  /// Map of interface implementations - we can easily collect these at the
  /// beginning of `run` while generators are still deflated and pass the map as
  /// read-only.
  DenseMap<GeneratorInterfaceOp, SmallVector<GeneratorOp>> implementsMap;

  /// A logger used to emit information during the elaboration process.
  Logger logger;

  /// The allocator we use to allocate children in the expansion tree.
  llvm::SpecificBumpPtrAllocator<ExpansionTreeNode> alloc;
  /// The root of the expansion tree.
  ExpansionTreeNode root;

  /// Evaluation semaphore - this ensures we benchmark one thing at a time. We
  /// initialize it to 1 so that the first thing to evaluate something doesn't
  /// block.
  // TODO (#7826): Make this a NamedSemaphore so it's unique across processes
  //               too.
  Semaphore evalSemaphore;
};
} // namespace

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getConcreteFunction
//===----------------------------------------------------------------------===//

ErrorTreeOr<FuncOp>
ElaboratorImpl::getConcreteFunction(Location loc, SymbolRefAttr symbolRef,
                                    ArrayRef<ParamBindAttr> paramValues) {
  auto funcItf = lookup<FuncInterface>(symbolRef);
  if (auto func = dyn_cast<FuncOp>(funcItf.getOperation()))
    return func;

  SmallVector<Attribute> inputParams;
  for (ParamBindAttr bind : paramValues)
    inputParams.push_back(cast<Attribute>(bind.getValue()));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ExpansionTreeNode *node = root.find(funcItf, vals);
  // If the node has already been elaborated, just use that result.
  if (node && node->isConcrete())
    return node->getFirstConcrete();

  // Otherwise, if the node doesn't exist, then create a new one.
  if (!node)
    node =
        ExpansionTreeNode::create(funcItf, vals, &root, IREvaluator(*this), 0);

  for (auto [decl, value] : llvm::zip(funcItf.getInputParamDecls(), vals))
    node->evaluator.setOrOverwriteParameterValue(decl, value);

  if (auto gen = dyn_cast<GeneratorOp>(funcItf.getOperation())) {
    if (auto err = specializeGenerator(node))
      return std::move(*err);
  } else if (auto itf =
                 dyn_cast<GeneratorInterfaceOp>(funcItf.getOperation())) {
    if (auto err = specializeInterface(node, implementsMap[itf]))
      return std::move(*err);
  }

  return node->getFirstConcrete();
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::getAllConcreteFunctions
//===----------------------------------------------------------------------===//

std::optional<ErrorTree>
ElaboratorImpl::getAllConcreteFunctions(Location loc, SymbolRefAttr symbolRef,
                                        ArrayRef<ParamBindAttr> paramValues,
                                        std::vector<FuncOp> &funcs) {
  auto funcItf = lookup<FuncInterface>(symbolRef);
  if (auto func = dyn_cast<FuncOp>(funcItf.getOperation())) {
    funcs.push_back(func);
    return std::nullopt;
  }

  SmallVector<Attribute> inputParams;
  for (ParamBindAttr bind : paramValues)
    inputParams.push_back(cast<Attribute>(bind.getValue()));

  auto vals = ArrayAttr::get(symbolRef.getContext(), inputParams);

  // Lookup the node if it already exists.
  ExpansionTreeNode *node = root.find(funcItf, vals);
  // If the node has already been elaborated, just use that result.
  if (node && node->isConcrete()) {
    node->getAllConcrete(funcs);
    return std::nullopt;
  }

  // Otherwise, if the node doesn't exist, then create a new one.
  if (!node)
    node =
        ExpansionTreeNode::create(funcItf, vals, &root, IREvaluator(*this), 0);

  for (auto [decl, value] : llvm::zip(funcItf.getInputParamDecls(), vals))
    node->evaluator.setOrOverwriteParameterValue(decl, value);

  if (auto gen = dyn_cast<GeneratorOp>(funcItf.getOperation())) {
    if (auto err = specializeGenerator(node))
      return std::move(*err);
  } else if (auto itf =
                 dyn_cast<GeneratorInterfaceOp>(funcItf.getOperation())) {
    if (auto err = specializeInterface(node, implementsMap[itf]))
      return std::move(*err);
  }

  node->getAllConcrete(funcs);
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamSearchOp
//===----------------------------------------------------------------------===//

/// Process a param.search op.
std::optional<ErrorTree>
ElaboratorImpl::processParamSearchOp(ExpansionTreeNode *parent,
                                     ParamSearchOp op,
                                     ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Processing ParamSearchOp");
  LLVM_DEBUG(logger.scope("Options") << op.getValuesAttr() << "\n");

  // Loop over all the possible candidates that we will search over, spawning
  // N possibilities to explore.
  SmallVector<ErrorTree> errors;
  DenseSet<Attribute> seenValues;
  if (op.getValues().empty())
    return ErrorTree(op.getLoc(), "no candidates found");

  bool atLeastOneSuccessful = false;
  for (Attribute candidate : op.getValues()) {
    // Simplify the input expressions.
    auto errorOrValue =
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
    if (auto err =
            spawnParamSearchClone(op, value, parent, remainingWorklist)) {
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
  parent->error = ErrorTree(op.getLoc(), "search param base node");
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::spawnParamSearchClone
//===----------------------------------------------------------------------===//

/// Spawn a clone from a param.search op.
std::optional<ErrorTree>
ElaboratorImpl::spawnParamSearchClone(ParamSearchOp searchOp, Attribute value,
                                      ExpansionTreeNode *searchParentNode,
                                      ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Spawning ParamSearchClone for '", value, "'");

  // Start by cloning the current WIP func to a new copy of it.
  IRMapping map;
  auto newFunc = cast<FuncOp>(searchParentNode->op->clone(map));
  // Insert into the symbol table - this will also unique the name for us.
  analysis.getTopLevelSymbolTable().insert(
      newFunc, ++searchParentNode->op->getIterator());

  // Hook this new clone up correctly.
  auto newFuncNode = ExpansionTreeNode::create(
      newFunc, searchParentNode->inputParams, searchParentNode->parent,
      searchParentNode->evaluator, searchParentNode->expansionDepth);

  // Change the future of this func by resolving the searchOp in the new func
  // to the specified value.
  auto newSearch = cast<ParamSearchOp>(map.lookup(searchOp.getOperation()));

  LLVM_DEBUG(logger << "Setting '" << newSearch.getParamDecl() << "' = '"
                    << value << "'\n");

  // Update the evaluator.
  newFuncNode->evaluator.setOrOverwriteParameterValue(newSearch.getParamDecl(),
                                                      value);
  newSearch->erase();

  // Map to the new ops.
  auto remaining = llvm::to_vector(llvm::map_range(
      remainingWorklist, [&](Operation *op) { return map.lookup(op); }));

  // And finally, process the rest of the worklist in this new scope.
  processScope(newFuncNode, remaining);

  // If we've hit an error case, don't try and finish processing. Return to the
  // upper function that this hit an error.
  if (newFuncNode->error)
    return newFuncNode->error->copy();

  // And handle the return processing.
  completeReturnProcessing(logger, newFunc.getReturnOp(), newFuncNode);

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processGeneratorUser
//===----------------------------------------------------------------------===//

/// Process a generator user like a call.
std::optional<ErrorTree> ElaboratorImpl::processGeneratorUser(
    ArrayRef<ParamDeclAttr> decls, KGENCallOpInterface user,
    SymbolConstantAttr calleeSymbol, ExpansionTreeNode *parent,
    ArrayRef<Operation *> remainingWorklist) {
  auto _ = logger.scope("Processing Generator User");
  LLVM_DEBUG(logger.logOp("User", user));

  assert(remainingWorklist.empty() || remainingWorklist.front() != user);

  // Add in the mapping for parameters in the calls.
  auto resolvedCallParamsOr =
      resolveCallInputParams(user, parent, calleeSymbol.getParamValues());
  if (resolvedCallParamsOr.isError())
    return resolvedCallParamsOr.takeError();
  auto inputParamKey = *resolvedCallParamsOr;

  // Lookup the callee.
  auto calleeOp = lookup(calleeSymbol.getSymbol());
  if (!calleeOp) {
    return ErrorTree(user.getLoc(), "could not find callee '" +
                                        mlir::debugString(calleeSymbol) + "'");
  }

  LLVM_DEBUG({
    logger.logOp("Callee", calleeOp);
    logger << "Input Params: " << inputParamKey << "\n";
  });

  // If we already have a binding for this, we're done.
  auto found = parent->bindings.find({calleeOp, inputParamKey});
  if (found != parent->bindings.end()) {
    LLVM_DEBUG(
        found->getSecond()->print(logger.scope("Result: Existing Binding")));
    completeGeneratorUserProcessing(user, decls, found->getSecond(), parent);
    return std::nullopt;
  }

  // Find the tree node that corresponds to the thing we're calling.
  auto callee = root.find(calleeOp, inputParamKey);
  // If we haven't found this callee yet, we have to add it to the tree!
  if (!callee) {
    // Use the parent of the call to show the expansion depth, and inherit the
    // evaluator from the root.
    auto calleeNode =
        ExpansionTreeNode::create(calleeOp, inputParamKey, &root,
                                  IREvaluator(*this), parent->expansionDepth);
    if (calleeNode->error)
      return std::move(calleeNode->error);

    if (isa<GeneratorOp>(calleeOp)) {
      if (auto err = specializeGenerator(calleeNode))
        return err;
    } else if (isa<FuncOp>(calleeOp)) {
      if (auto err = specializeFunction(calleeNode, decls))
        return err;
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(calleeOp)) {
      if (auto err = specializeInterface(calleeNode, implementsMap[itf]))
        return err;
    }

    callee = calleeNode;
  }
  LLVM_DEBUG(callee->print(logger));

  // Complete processing for all the leaves of this subtree.
  std::vector<ExpansionTreeNode *> concrete;
  callee->getAllConcreteNodes(concrete);

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
    callee->collectSubtreeErrors(out);
    return std::move(out);
  }

  // If we called into an interface, and search was off, use the first concrete
  // node.
  if (isa<GeneratorInterfaceOp>(calleeOp) && !enableSearch)
    concrete.erase(concrete.begin() + 1, concrete.end());

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
    auto newNode =
        ExpansionTreeNode::create(newOp, parent->inputParams, parent->parent,
                                  parent->evaluator, parent->expansionDepth);
    newNode->bindings = parent->bindings;
    // Bind this concrete impl to this callee for this node.
    newNode->bindings[{calleeOp, inputParamKey}] = c;

    completeGeneratorUserProcessing(map.lookup(user.getOperation()), decls, c,
                                    newNode);

    LLVM_DEBUG(newNode->print(logger << "New Op "));

    // We have to finish specializing this thing now. Map to the new ops and
    // process the remaining scope.
    auto remaining = llvm::to_vector(llvm::map_range(
        remainingWorklist, [&](Operation *op) { return map.lookup(op); }));

    // Process the rest of the worklist in this new scope.
    processScope(newNode, remaining);

    // And handle the return.
    completeReturnProcessing(logger, cast<FuncOp>(newOp).getReturnOp(),
                             newNode);
  }

  // Bind this concrete impl to this callee for this node.
  parent->bindings[{calleeOp, inputParamKey}] = concrete.front();

  // Call completeGeneratorUserProcessing on the first concrete thing. This will
  // flow nested bindings upward correctly.
  completeGeneratorUserProcessing(user, decls, concrete.front(), parent);

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::resolveCallInputParams
//===----------------------------------------------------------------------===//

/// Resolve input params on a call_param op.
ErrorTreeOr<ArrayAttr>
ElaboratorImpl::resolveCallInputParams(KGENCallOpInterface call,
                                       ExpansionTreeNode *parentNode,
                                       ParamBindArrayAttr inputValues) {
  LLVM_DEBUG(logger.logOp("Resolving Call Input Params", call);
             logger << " with input bindings: " << inputValues << "\n");

  SmallVector<Attribute> boundInputParams;
  for (ParamBindAttr param : inputValues) {
    // Fold the parameter expression in this context to a simple constant.
    ErrorTreeOr<Attribute> valueOr =
        parentNode->evaluator.concretizeParameterExpr(call.getLoc(),
                                                      param.getValue());
    if (valueOr.isError())
      return valueOr.takeError();

    Attribute value = valueOr.takeValue();
    boundInputParams.push_back(value);
  }

  return ArrayAttr::get(call->getContext(), boundInputParams);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processCallParamOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
std::optional<ErrorTree>
ElaboratorImpl::processCallParamOp(CallParamOp call, ExpansionTreeNode *parent,
                                   ArrayRef<Operation *> remainingWorklist) {
  auto symbol = parent->evaluator.concretizeParameterExpr(call.getLoc(),
                                                          call.getCallee());
  if (symbol.isError())
    return symbol.takeError();

  auto symbolCst = dyn_cast<SymbolConstantAttr>(*symbol);
  if (!symbolCst)
    return ErrorTree(call.getLoc(), "must be a SymbolConstantAttr");
  return processGeneratorUser(call.getParamDecls(), call, symbolCst, parent,
                              remainingWorklist);
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processParamDeclareRegionOp
//===----------------------------------------------------------------------===//

/// Process a param.declare.region by creating a generator for the contained
/// region.
std::optional<ErrorTree>
ElaboratorImpl::processParamDeclareRegionOp(ParamDeclareRegionOp regionDecl,
                                            ExpansionTreeNode *parent) {
  return ErrorTree(regionDecl.getLoc(),
                   "should never see one of these (for now)");
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::processScope
//===----------------------------------------------------------------------===//

void ElaboratorImpl::processScope(ExpansionTreeNode *parentNode,
                                  ArrayRef<Operation *> worklist) {
  LLVM_DEBUG({
    auto _ = logger.scope("Operations to Rewrite");
    for (Operation *op : worklist)
      logger << *op << "\n";
  });

  // Processing an op may generate more stuff, or even delete the op being
  // processed.
  for (auto iter = worklist.begin(), end = worklist.end(); iter != end;
       ++iter) {
    Operation *op = *iter;
    ArrayRef<Operation *> remainingWorklist(iter + 1, end);
    auto _ = logger.scope("Processing: '", op->getName(), "'");
    logger.logOp("Op", op);

    std::optional<ErrorTree> result = std::nullopt;
    if (auto bind = dyn_cast<ParamDeclareOp>(op)) {
      result = processParamDeclareOp(parentNode->evaluator, bind);
    } else if (auto declare = dyn_cast<ParamDeclareRegionOp>(op)) {
      result = processParamDeclareRegionOp(declare, parentNode);
    } else if (auto search = dyn_cast<ParamSearchOp>(op)) {
      result = processParamSearchOp(parentNode, search, remainingWorklist);
    } else if (auto value = dyn_cast<ParamConstantOp>(op)) {
      result = processParamConstantOp(parentNode->evaluator, value);
    } else if (auto assertOp = dyn_cast<ParamAssertOp>(op)) {
      result = processParamAssertOp(parentNode->evaluator, assertOp);
    } else if (auto addressof = dyn_cast<AddressOfOp>(op)) {
      result = processGeneratorUser(addressof.getParamDecls(), addressof,
                                    addressof.getCallee(), parentNode,
                                    remainingWorklist);
    } else if (auto call = dyn_cast<CallOp>(op)) {
      result =
          processGeneratorUser(call.getParamDecls(), call, call.getCallee(),
                               parentNode, remainingWorklist);
    } else if (auto callParam = dyn_cast<CallParamOp>(op)) {
      result = processCallParamOp(callParam, parentNode, remainingWorklist);
    } else {
      result = processGenericOp(parentNode->evaluator, op);
    }

    // st set the node to an error - we don't want to fail the whole
    // process.
    if (result) {
      LLVM_DEBUG(logger.scope("Result: Failure") << result->getError());
      parentNode->error = std::move(result);
      return;
    }

    // If the parent node was set to error, then just bail.
    if (parentNode->error)
      return;
  }

  LLVM_DEBUG(parentNode->print(logger << "Completed processing "));
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeFunction
//===----------------------------------------------------------------------===//

std::optional<ErrorTree>
ElaboratorImpl::specializeFunction(ExpansionTreeNode *funcNode,
                                   ArrayRef<ParamDeclAttr> paramDecls) {
  FuncOp func = cast<FuncOp>(funcNode->op);

  auto _ = logger.scope("Specializing Function: @", func.getName());
  logger.logOp("Function", func);

  // Get a partial ordering of parameter definitions and uses that are listed
  // "top down" in our evaluation order. Plug in the input parameter values.
  ParameterUseDefGraph uses(func);
  for (auto [decl, val] : llvm::zip(paramDecls, funcNode->inputParams)) {
    uses.decls.try_emplace(decl.getName(),
                           ParamDeclaration{decl.getType(), func});
    uses.defs.try_emplace(decl.getName(), ParamDefinition{val, {}, func, {}});
  }
  uses.calculate();

  // FIXME: The elaborator does not correctly handle the new parameter use-def
  // graph. Process the parameters in reverse: the same operation can define
  // multiple parameters, so punt those according to their most dominated
  // definition.
  std::vector<Operation *> opsToRewrite;
  opsToRewrite.reserve(uses.params.size() + uses.paramOps.size());
  llvm::SetVector<Operation *, SmallVector<Operation *, 8>,
                  SmallPtrSet<Operation *, 8>>
      defOps;
  for (StringAttr param : llvm::reverse(uses.params)) {
    auto it = uses.defs.find(param);
    assert(it != uses.defs.end());
    defOps.insert(it->second.defOp);
  }
  // The only acceptable leaf nodes are input parameters.
  for (auto &[param, decl] : uses.decls) {
    if (uses.defs.find(param) != uses.defs.end())
      continue;
    return ErrorTree(decl.declOp->getLoc(),
                     "unknown parameter-defining operator");
  }
  llvm::append_range(opsToRewrite, llvm::reverse(defOps.takeVector()));
  llvm::append_range(opsToRewrite, uses.paramOps);

  // Process the worklist.
  processScope(funcNode, opsToRewrite);

  // Bail if we hit an error.
  if (funcNode->error)
    return std::nullopt;

  // Store the return parameters.
  if (auto err = completeReturnProcessing(logger, func.getReturnOp(), funcNode))
    return err;

  // Check that the thing we just built is correct IR!  We want to catch any
  // errors produced by the verify pass, we don't want them to actually get
  // emitted.
  std::string verificationErrorStr;
  llvm::raw_string_ostream verificationError(verificationErrorStr);
  Optional<Location> verificationLoc;
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

  // Verify the function and invoke the symbol user verifier on all its
  // contained ops to verify parameter references.
  auto verifySymbolUses = [&](mlir::SymbolUserOpInterface user) -> WalkResult {
    return user.verifySymbolUses(analysis.getSymbolTables());
  };
  if (failed(verify(func)) || func.walk(verifySymbolUses).wasInterrupted()) {
    funcNode->error =
        ErrorTree(*verificationLoc,
                  Twine("verification error: ") + verificationError.str());
    LLVM_DEBUG(logger.scope("Result: Failure")
               << verificationError.str() << "\n");
    return std::nullopt;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeGenerator
//===----------------------------------------------------------------------===//

std::optional<ErrorTree>
ElaboratorImpl::specializeGenerator(ExpansionTreeNode *genNode) {
  GeneratorOp generator = cast<GeneratorOp>(genNode->op);

  auto _ = logger.scope("Specializing Generator: @", generator.getName());
  logger.logOp("Generator", generator);

  // Bind all parameter values in this scope.
  ArrayRef<Attribute> inputParamValues = genNode->inputParams.getValue();
  auto inputParamDecls = generator.getInputParamDecls();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  IREvaluator &evaluator = genNode->evaluator;
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setOrOverwriteParameterValue(decl, val);

  // If the generator's constraints don't satisfy, set an error and move on.
  if (auto err =
          KGEN::evaluateConstraints(generator.getConstraints(), evaluator)) {
    genNode->error = std::move(err.value());
    return std::nullopt;
  }

  // If we already have something in the tree then don't re-specialize.
  StringAttr mangledName = mangleParameterValues(generator, inputParamValues);
  if (Operation *op = lookup(mangledName)) {
    if (auto found = genNode->getRoot()->find(op, genNode->inputParams)) {
      LLVM_DEBUG(logger << "Result: Generator has already been specialized\n");

      // If we have the node, then take it as a child.
      if (found->parent != genNode)
        genNode->takeExpansion(found);
      return std::nullopt;
    }
  }

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(generator);
  auto newFunc = b.create<FuncOp>(
      generator.getLoc(), mangledName,
      SignatureType::get(ParamDeclArrayAttr::get(generator.getContext(), {}),
                         generator.getResultParamsAttr(),
                         generator.getFunctionType(),
                         generator.getConventions()),
      generator.getAlwaysInlineLevel());

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  analysis.getTopLevelSymbolTable().insert(newFunc, generator->getIterator());

  // Await for the generator in the asyncMap - otherwise it won't have the cache
  // attrs we need.
  if (auto err = asyncMap.await(generator))
    return ErrorTree(generator.getLoc(), err.takeError());

  // Set the body on the new func with the hash rather than cloning if at all
  // possible.
  if (auto hashAttr = generator->getAttrOfType<Cache::RegionHashArrayAttr>(
          Cache::getRegionHashAttrName())) {
    newFunc->setAttr(Cache::getRegionHashAttrName(), hashAttr);
  } else {
    // Otherwise, just clone the body.
    // TODO: is there a nice way to not have to clone this?
    IRMapping map;
    generator.getBodyRegion().cloneInto(&newFunc.getBodyRegion(), map);
  }

  // The node for this new func is simply the child of the node for the
  // generator.
  auto newFuncNode =
      ExpansionTreeNode::create(newFunc, genNode->inputParams, genNode,
                                evaluator, genNode->expansionDepth);

  // Inflate the function and then specialize it.
  asyncMap.mapChained(
      newFunc, [newFunc, regionCache = regionCache.copy()](auto ch) mutable {
        return Cache::inflateOp(newFunc, std::move(regionCache), std::move(ch));
      });
  if (auto err = asyncMap.await(newFunc))
    return ErrorTree(newFunc.getLoc(), err.takeError());

  // Kick off the expansion for the new function.
  return specializeFunction(newFuncNode, generator.getInputParamDeclsAttr());
}

//===----------------------------------------------------------------------===//
// ElaboratorImpl::specializeInterface
//===----------------------------------------------------------------------===//

/// Specialize a generator interface.
std::optional<ErrorTree>
ElaboratorImpl::specializeInterface(ExpansionTreeNode *itfNode,
                                    ArrayRef<GeneratorOp> interfaceImpls) {
  auto itf = cast<GeneratorInterfaceOp>(itfNode->op);

  // Bind all the parameter values in this scope.
  ArrayRef<Attribute> inputParamValues = itfNode->inputParams.getValue();
  auto inputParamDecls = itf.getInputParamDecls();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    itfNode->evaluator.setOrOverwriteParameterValue(decl, val);

  if (interfaceImpls.empty()) {
    itfNode->error =
        ErrorTree(itfNode->op.getLoc(), "no implementations of interface '" +
                                            itf.getName() + "' found");
    return std::nullopt;
  }

  // If a default has been provided, and we don't want to do search, then use
  // it.
  std::optional<SymbolConstantAttr> defaultImpl = itf.getDefaultImpl();
  if (!enableSearch && defaultImpl.has_value()) {
    LLVM_DEBUG(logger << "Using default implementation for interface: "
                      << *defaultImpl << "\n");
    // If the SymbolConstant exists, then the callee must exist.
    auto defaultImplCallee = lookup<GeneratorOp>(defaultImpl->getSymbol());
    assert(defaultImplCallee && "expected defaultImpl to exist");
    // If we already have a node for this generator, just use it. Otherwise,
    // create a new one and specialize it.
    auto genNode =
        itfNode->getRoot()->find(defaultImplCallee, itfNode->inputParams);
    if (!genNode) {
      genNode = ExpansionTreeNode::create(
          defaultImplCallee, itfNode->inputParams, itfNode, IREvaluator(*this),
          itfNode->expansionDepth);
    } else if (genNode->parent != itfNode) {
      itfNode->takeExpansion(genNode);
    }
    return specializeGenerator(genNode);
  }

  LLVM_DEBUG(logger << "Kicking off specializations for all generators\n");

  // Kick off specializations for all the generators. We'll use the flattened
  // list of leaves as the inputs to search.
  for (GeneratorOp gen : interfaceImpls) {
    auto genNode = itfNode->getRoot()->find(gen, itfNode->inputParams);
    if (!genNode) {
      genNode = ExpansionTreeNode::create(gen, itfNode->inputParams, itfNode,
                                          IREvaluator(*this),
                                          itfNode->expansionDepth);
    } else if (genNode->parent != itfNode) {
      itfNode->takeExpansion(genNode);
    }

    if (auto err = specializeGenerator(genNode))
      return err;
  }

  // Get all the concrete implementations for this interface.
  std::vector<FuncOp> concrete;
  itfNode->getAllConcrete(concrete);

  // Only one implementation, or search is off, so we're done.
  if (concrete.size() == 1 || !enableSearch)
    return std::nullopt;

  SymbolConstantAttr evaluator = itf.getEvaluatorAttr();
  LLVM_DEBUG(logger << "Evaluator: " << evaluator << "\n");

  // There's no evaluator, we're done.
  if (!evaluator)
    return std::nullopt;

  auto keyBuf = Cache::WriteableBuffer::get();

  // Pull out the elaboration results that succeeded to provide to the search
  // inputs. We also write the bytecode for each input into the key.
  for (const auto &r : concrete)
    mlir::writeBytecodeToFile(r, *keyBuf);

  // Part of the key is the evaluation function. If it has not been
  // elaborated, do it now.
  Operation *eval = lookup(evaluator.getSymbol());
  if (!eval)
    return ErrorTree(itfNode->op.getLoc(), "could not find evaluator '" +
                                               mlir::debugString(evaluator) +
                                               "'");

  LLVM_DEBUG(logger.logOp("Found evaluator", eval));

  // Create a new tree node with the new evaluator and use it. The
  // parameter values we want to use here are the ones stashed on the
  // evaluator itself.
  SmallVector<Attribute> attrs;
  for (auto param : evaluator.getParamValues()) {
    auto valueOr = itfNode->evaluator.concretizeParameterExpr(eval->getLoc(),
                                                              param.getValue());
    if (valueOr.isError())
      return valueOr.takeError();

    attrs.push_back(*valueOr);
  }
  auto evalInputParams = ArrayAttr::get(eval->getContext(), attrs);

  // Ensure the evaluator is elaborated.
  auto node = root.find(eval, evalInputParams);
  if (!node)
    node =
        ExpansionTreeNode::create(eval, evalInputParams, &root,
                                  itfNode->evaluator, itfNode->expansionDepth);

  // Elaborate the evaluator node.
  if (auto err = specializeGenerator(node))
    return err;

  std::vector<FuncOp> concreteEvaluators;
  node->getAllConcrete(concreteEvaluators);
  if (concreteEvaluators.empty()) {
    ErrorTree out(eval->getLoc(), "no viable expansions found");
    node->collectSubtreeErrors(out);
    return out;
  }

  if (concreteEvaluators.size() > 1) {
    return ErrorTree(itf->getLoc(), "evaluator should have one candidate")
        .addCause(eval->getLoc(), "evaluator defined here");
  }

  auto evalFuncOr = node->getFirstConcrete();
  if (evalFuncOr.isError())
    return evalFuncOr.takeError();
  FuncOp evalFunc = *evalFuncOr;
  LLVM_DEBUG(logger.logOp("Chose Evaluation Func", evalFunc));

  mlir::writeBytecodeToFile(evalFunc, *keyBuf);

  // And finally, the target.
  *keyBuf << target;

  // Alright - we want to do search now.
  LLCL::AsyncValue::registerTypes<FuncOp>();

  // This provides the implementation of search. This is the part we actually
  // care about caching because it's the most expensive part.
  auto doSpecialization = [this, evalFunc,
                           concrete](Operation *itfOp,
                                     Cache::WriteableBufferRef toCache,
                                     AnyAsyncValueRef chain) {
    auto out = LLCL::AsyncValueRef<FuncOp>::allocate(runtime);
    std::move(chain).andThenSync(
        [this, evalFunc, concrete, itfOp, out = out.copy(),
         toCache = std::move(toCache)](AnyAsyncValueRef &&chain) mutable {
          auto itf = cast<GeneratorInterfaceOp>(itfOp);

          evalSemaphore.wait();
          ErrorOr<size_t> bestSpecializationIdxOr = evaluateSpecializations(
              evalFunc, analysis.getTopLevelSymbolTable(), runtime, target,
              concrete);
          evalSemaphore.post();

          if (failed(bestSpecializationIdxOr)) {
            return out.setToError(getMLIRDiagnostic(
                bestSpecializationIdxOr.takeError(), itf.getLoc()));
          }

          // Find the fastest one and return just that one.
          FuncOp bestResult = concrete[*bestSpecializationIdxOr];

          // Finally, cache the result.
          *toCache << bestResult.getName();
          return std::move(out).emplace(bestResult);
        });
    return out;
  };

  auto onCacheHit =
      [this, concrete](Operation *itfOp,
                       Cache::BufferRef cacheContents) -> AnyAsyncValueRef {
    auto out = LLCL::AsyncValueRef<FuncOp>::allocate(runtime);
    StringAttr fastestFuncName =
        StringAttr::get(itfOp->getContext(), cacheContents->getBuffer());

    // Find the fastest function by name.
    auto fastest = llvm::find_if(concrete, [&](FuncOp func) {
      return func.getNameAttr() == fastestFuncName;
    });
    if (fastest == concrete.end()) {
      out.setToError(LLCL::getMLIRDiagnostic(
          Error("could not find " + fastestFuncName.getValue()),
          itfOp->getLoc()));
    } else {
      out.copy().emplace(*fastest);
    }

    return std::move(out);
  };

  // Run the transform with the functions we just defined.
  auto xform =
      Cache::cachedTransform(itf, transformCache.copy(),
                             LLCL::AsyncValueRef<Chain>::createReady(runtime),
                             std::move(keyBuf), doSpecialization, onCacheHit);

  LLCL::await(xform);
  if (xform.isError())
    return ErrorTree(itf.getLoc(), xform.getDiagnostic().getMessage().copy());
  else
    concrete = {std::move(xform.get<FuncOp>())};

  // Trim the nodes that we don't want. Check all the concrete implementations
  // and ensure that only the one(s) that we chose are in the expansion tree.
  // Mark the others as having errored so they are removed properly at the end
  // of elaboration.
  for (auto *child : itfNode->expansions) {
    if (!child->isConcrete())
      child->error = ErrorTree(itf.getLoc(), "no viable expansions found");

    for (auto *c : child->expansions) {
      if (!llvm::is_contained(concrete, c->op))
        c->error = ErrorTree(itf.getLoc(), "not chosen in search");
    }
  }

  LLVM_DEBUG(itfNode->print(logger << "Post Search Interface "));

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
  {
    auto _ = logger.scope("Processing Primary Generator Interfaces");
    for (auto gen : theModule.getOps<GeneratorOp>()) {
      if (auto implements = gen.getImplementsAttr()) {
        LLVM_DEBUG(logger << "Found generator '@" << gen.getName()
                          << "' implementing interface '" << implements
                          << "'\n");
        if (auto itf = lookup<GeneratorInterfaceOp>(implements))
          implementsMap[itf].push_back(gen);
      }
    }
  } // The implementsMap map is read-only after *here*.

  for (auto gen : primaryGenerators) {
    LLVM_DEBUG(logger.logOp("Elaborating primary generator", gen));
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    ExpansionTreeNode *generatorNode = root.find(gen, emptyInputParamKey);
    if (!generatorNode)
      generatorNode = ExpansionTreeNode::create(gen, emptyInputParamKey, &root,
                                                IREvaluator(*this), 0);

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
    auto genNode = root.find(gen, emptyInputParamKey);
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
      auto genNode = root.find(gen, emptyInputParamKey);
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

  for (auto op : toErase)
    op->erase();

  SymbolTable &symtab = analysis.getTopLevelSymbolTable();
  for (Operation &op : llvm::make_early_inc_range(theModule.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op)) {
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
  theModule->walk([&](Operation *op) {
    // If this is a func being renamed, rename it.
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (auto newName = funcsToRename.lookup(func.getNameAttr())) {
        // Keep the symbol table up-to-date with the new name.
        // TODO: We should upstream something for this.
        symtab.remove(func);
        func.setSymNameAttr(newName);
        symtab.insert(func, op->getIterator());
      }
      return;
    }

    // If this is a reference to a function that got renamed, update its
    // target.
    TypeSwitch<Operation *>(op).Case<CallOp, AddressOfOp>([&](auto op) {
      SymbolConstantAttr callee = op.getCallee();
      auto newName = funcsToRename.lookup(
          cast<FlatSymbolRefAttr>(callee.getSymbol()).getAttr());
      if (newName)
        op.setCalleeAttr(SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(newName), callee.getType()));
    });
  });

  // Update the debug info for everything now that we've done renaming etc.
  root.updateDebugInfo();

  if (root.isConcrete())
    LLVM_DEBUG(logger.logOp("Finished successfully", theModule));

  // We were only successful if the root could be concretized.
  return success(root.isConcrete());
}

//===----------------------------------------------------------------------===//
// M::KGEN::elaborateGeneratorsV2
//===----------------------------------------------------------------------===//

LogicalResult
M::KGEN::elaborateGeneratorsV2(mlir::SymbolTableAnalysis &analysis,
                               LLCL::Runtime &runtime, TargetInfoAttr target,
                               ArrayRef<GeneratorOp> primaryGenerators,
                               bool enableSearch) {
  TimeTraceScope<> traceScope("elaborate-generators");
  ModuleOp theModule = analysis.getTopLevelOp<ModuleOp>();

  AsyncSideEffectMap asyncMap(runtime);

  auto transformCacheBackendOr =
      Cache::getDefaultBackendChain(runtime, ".kgen_cache/transform");
  if (failed(transformCacheBackendOr))
    return theModule->emitError() << transformCacheBackendOr.getError();
  auto regionCacheBackendOr =
      Cache::getDefaultBackendChain(runtime, ".kgen_cache/region");
  if (failed(regionCacheBackendOr))
    return theModule->emitError() << regionCacheBackendOr.getError();

  auto transformCache =
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
          transformCacheBackendOr.takeValue());
  auto regionCache =
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>>::create(
          regionCacheBackendOr.takeValue());

  // Deflate every generator in the primary module.
  // TODO: We should be able to deflate *everything*
  for (auto op : theModule.getOps<GeneratorOp>()) {
    asyncMap.mapChained(op, [op, regionCache = regionCache.copy()](auto ch) {
      return Cache::deflateOp(op, regionCache.copy(), std::move(ch));
    });
  }

  // Now, construct and run the elaborator.
  ElaboratorImpl impl(analysis, target, transformCache->getRuntime(), asyncMap,
                      transformCache.copy(), regionCache.copy(), enableSearch);
  return impl.run(primaryGenerators);
}
