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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Support/ForkJoin.h"
#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;
using namespace AsyncRT;

//===----------------------------------------------------------------------===//
// mangleParameterValues
//===----------------------------------------------------------------------===//

/// Return a string that is a unique specification of the specified parameter.
static void printParameterMangling(TypedAttr value, raw_ostream &os) {
  // TODO: Don't print #lit.lifetime, they are singletons!
  // if (isa<LifetimeAttr>(value))
  //  return;

  // It is very common to pass types in as parameters, since this is how
  // parameters of Trait type get substituted in.  We'd really like to include
  // just the full name of the type (e.g. something like
  // stdlib::builtin::string_literal::StringLiteral) instead of the full string
  // form of the TypeConstantAttr - which includes a vtable!

  // TODO(MOCO-945): we should have this mangled in in a stable way in the
  // vtable itself.  Until then, we do a "terrible hack" to get it because
  // everything will have a del member, and thus have something like:
  //   @"stdlib::builtin::string_literal::StringLiteral::__del__(
  //       stdlib::builtin::string_literal::StringLiteral)
  //  ... in the stringified form of the parameter.
  if (auto typeCst = dyn_cast<TypeConstantAttr>(value)) {
    VTableAttr vtable = typeCst.getVTable();

    // We need to mangle in the metatype of the TypeConstantAttr into the name
    // of the type, not just the type itself.  Consider something like:
    //   struct Thing[element_trait: _AnyTypeMetaType, type: element_trait]: ...
    // we need Thing[AnyType, Int] and Thing[Printable, Int] to mangle
    // differently.  We also need to mangle different named traits with the same
    // bodies, e.g. consider "trait MyAnyType: pass" and "Thing[AnyType, Int]"
    // vs "Thing[MyAnyType, Int]".
    //
    // Unfortunately, we currently lose the metatype completely because of
    // lower-lit, so we mangle in the list of requirements as a "incorrect but
    // close" way to unique them.
    // TODO(MOCO-945): We should have the metatype name directly in the vtable.
    std::string requirementsList;

    // Find the __del__ member if present.
    SymbolConstantAttr delMember;
    for (VTableEntryAttr entry : vtable.getEntries()) {
      if (entry.getName() == "__del__") {
        delMember = dyn_cast<SymbolConstantAttr>(entry.getMethod());
      } else {
        // Build a list of the non-del requirements as a gross approximation
        // for the metatype of the trait.
        if (!requirementsList.empty())
          requirementsList += ",";
        requirementsList += entry.getName().strref();
      }
    }

    // If we have a __del__ member, process it.
    if (delMember) {
      auto delMangledName = delMember.getSymbol().getLeafReference().strref();

      // Drop the stuff up to the argument name in the parens.
      size_t pos = delMangledName.find('(');
      assert(pos != StringRef::npos && "Mojo has predictable symbol mangling");
      delMangledName = delMangledName.drop_front(pos + 1);

      // The REPL will cons up names like:
      // "...__del__(Expression [6] wrapper::Sphere)_thunk", and we don't want
      // to treat this as something named Expression, so drop this.
      if (delMangledName.starts_with("Expression [")) {
        delMangledName = delMangledName.drop_front(strlen("Expression ["));
        pos = delMangledName.find(']');
        assert(pos != StringRef::npos && delMangledName[pos + 1] == ' ' &&
               "Mojo has predictable symbol mangling");
        delMangledName = delMangledName.drop_front(pos + 2);
      }

      pos = delMangledName.find(')');
      assert(pos != StringRef::npos && "Mojo has predictable symbol mangling");
      delMangledName = delMangledName.take_front(pos);

      // If the type had parameters, it will have them erased to something
      // like 'simd::SIMD[$0, $1]'.  These are pointless, so drop them.
      pos = delMangledName.find('[');
      if (pos != StringRef::npos)
        delMangledName = delMangledName.take_front(pos);

      assert(!delMangledName.empty() && "Should have something!");
      os << delMangledName;

      // Encode any parameter values.
      if (!delMember.getParamValues().empty()) {
        os << '[';
        llvm::interleaveComma(delMember.getParamValues(), os,
                              [&](TypedAttr paramValue) {
                                printParameterMangling(paramValue, os);
                              });
        os << ']';
      }

      // Include the requirement list (if not AnyType) to include a semblance
      // of metatype into this.
      // TODO(MOCO-945): Use the actual named metatype.
      if (!requirementsList.empty())
        os << '.' << requirementsList;
      return;
    }
    // TODO: We're also getting TypeConstantAttrs for signature types, like:
    //    (!kgen.pointer<none>, !kgen.pointer<none>, !kgen.pointer<none>, index,
    //       index, index) capturing -> !kgen.none
    // Which could surely be compressed somehow as well.
  }

  // Handle VariadicAttr of types as well, which are common for variadic packs.
  if (auto variadic = dyn_cast<VariadicAttr>(value)) {
    os << '[';
    llvm::interleaveComma(variadic.getValues(), os, [&](TypedAttr paramValue) {
      printParameterMangling(paramValue, os);
    });
    os << ']';
    return;
  }

  // The kgen representation will always be a valid choice.
  os << getParamAsString(value);
}

std::string KGEN::mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<TypedAttr> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return generator.getName().str();

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  // Mangle in things like "size=42" for each of the parameters to make it easy
  // to read the resultant symbol and also to make things unique when
  // instantiated with different values.
  auto inputParamDecls = generator.getInputParamsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterMangling(value, os);
  }

  // Having "@" in mangled names is invalid for ELF files and triggers error at
  // linking stage, so replace them.
  std::replace(result.begin(), result.end(), '@', '_');
  return result;
}

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

ImplNode::ImplNode(ParamNode *parent)
    : parent(parent), paramGraph(parent->gen.getBodyRegion()) {}

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
  AsyncRT::await(quiesce());
}

ParamNode *ExpansionGraph::getOrCreate(AsyncRT::Runtime &runtime,
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
  if (!impl)
    return ErrorTree(gen.getLoc(), "function instantiation failed");
  if (!impl->error)
    return impl.get();
  // Propagate the error trivially if the current generator has no parameters.
  if (inputParams.empty())
    return impl->error->copy();
  return ErrorTree(gen.getLoc(), "function instantiation failed",
                   impl->error->copy());
}

ErrorTreeOr<FuncOp> ParamNode::getFirstConcreteFunc() {
  ErrorTreeOr<ImplNode *> impl = getFirstConcreteNode();
  if (impl.isError())
    return impl.takeError();
  return impl.takeValue()->func;
}

ErrorTreeOrSuccess ParamNode::collectErrorsOrSuccess() {
  if (!impl->error)
    return success();
  // Propagate the error trivially if the current generator has no parameters.
  if (inputParams.empty())
    return impl->error->copy();
  return ErrorTree(gen.getLoc(), "function instantiation failed",
                   impl->error->copy());
}

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
  inode->getEvaluator().setParameterValue(op.getParamDecl(), value);

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
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

static void collectOpsToProcessInside(Region &toProcess, ImplNode *parent,
                                      std::vector<Operation *> &opsToRewrite) {
  auto &nestedScopes = parent->paramGraph.nestedScopes;
  auto it = nestedScopes.find(&toProcess);
  assert(it != nestedScopes.end());
  const ParameterUseDefGraph &uses = it->second;

  // Only process the ops in the branch that we ended up taking.
  for (Operation *paramOp : llvm::reverse(uses.paramOps)) {
    // Check if this op is in a region that is a child of the region we care
    // about. If not, don't process it.
    if (!toProcess.isAncestor(paramOp->getParentRegion()))
      continue;

    opsToRewrite.push_back(paramOp);
  }
  collectOpsToProcess(&toProcess, uses, opsToRewrite);
}

//===----------------------------------------------------------------------===//
// Elaborator Implementation
//===----------------------------------------------------------------------===//

Elaborator::Elaborator(SymbolTable &symtab,
                       ParameterCollector::Analysis &paramCache,
                       TargetInfoAttr target, ElaboratorCallbacks callbacks,
                       const ElaborateGeneratorsOptions &config)
    : target(target), config(config), oldSymTab(symtab),
      env(symtab.getOp()->getAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName())),
      runtime(*loadContext(target.getContext())->get<AsyncRT::Runtime>()),
      g(this->runtime),
      paramCache(paramCache, runtime.getWorkQueue()->getParallelismLevel()),
      callbacks(std::move(callbacks)) {}

//===----------------------------------------------------------------------===//
// Elaborator::finalizeFunction
//===----------------------------------------------------------------------===//

void Elaborator::finalizeFunction(ImplNode *node) {
  VerboseCompilerTimeTraceScope traceScope("finalizeFunction");
  // Erase everything but the entry blocks of each region.
  FuncOp func = node->func;
  func.walk<mlir::WalkOrder::PreOrder>([](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(region)))
        block.erase();
  });
}

//===----------------------------------------------------------------------===//
// Elaborator::getConcreteFunction
//===----------------------------------------------------------------------===//

ErrorTreeOr<FuncOp> Elaborator::getConcreteFunction(ImplNode *parent,
                                                    Location loc,
                                                    SymbolConstantAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr();
  auto gen = oldSymTab.lookup<GeneratorOp>(name);
  // If this doesn't reference anything in the existing module, then it must
  // refer to a concrete function in the new module.
  if (!gen)
    return concreteFuncs.read([name](auto &funcs) { return funcs.at(name); });

  auto vals =
      ParameterExprArrayAttr::get(loc.getContext(), symbol.getParamValues());

  // Lookup the node if it already exists.
  ParamNode *node = g.getOrCreate(runtime, vals, gen, /*depth=*/0);
  // If the node has already been elaborated, just use that result.
  ElaborationState result =
      specializeGenerator(parent, node, /*from=*/nullptr, /*addWaiter=*/true);
  if (result.shouldSkipNode())
    return FuncOp();
  return node->getFirstConcreteFunc();
}

ErrorTreeOr<Attribute> Elaborator::concretizeSymbolsWithin(Attribute value,
                                                           ImplNode *parent,
                                                           Location loc) {
  mlir::AttrTypeReplacer replacer;
  std::optional<ErrorTree> error;
  replacer.addReplacement(
      [&](SymbolConstantAttr cst) -> std::pair<Attribute, WalkResult> {
        // Ignore parametric constants.
        if (!cst.getType().getInputParamTypes().empty())
          return {cst, WalkResult::advance()};
        ErrorTreeOr<FuncOp> func = getConcreteFunction(parent, loc, cst);
        if (func.isError()) {
          error = func.takeError();
          return {cst, WalkResult::interrupt()};
        }
        if (!*func)
          return {cst, WalkResult::interrupt()};

        return {SymbolConstantAttr::get(
                    FlatSymbolRefAttr::get(func.takeValue().getSymNameAttr()),
                    cst.getType()),
                WalkResult::skip()};
      });
  replacer.addReplacement([](VTableAttr vtable) {
    return std::make_pair(vtable, WalkResult::skip());
  });
  if (Attribute result = replacer.replace(value))
    return result;
  if (error)
    return std::move(*error);
  return Attribute();
}

//===----------------------------------------------------------------------===//
// Elaborator::addDeferredFunction
//===----------------------------------------------------------------------===//

void Elaborator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  FuncOp op = func.release();
  if (concreteFuncs.modify(
          [this, op, name = op.getSymNameAttr()](auto &funcs) mutable {
            if (funcs.try_emplace(name, op).second) {
              deferredSymbols.push_back(op);
              return true;
            }
            op.erase();
            return false;
          }))
    addConcreteFunc(op);
}

//===----------------------------------------------------------------------===//
// Elaborator::processParamConstantOp
//===----------------------------------------------------------------------===//

template <typename OpT>
ElaborationState Elaborator::processParamConstantOp(ImplNode *parent, OpT op) {
  Attribute attr;
  HANDLE_EVALUATOR_CONC(attr, parent, op->getLoc(), op.getValue());
  auto value = cast<TypedAttr>(attr);

  // Root elaboration at the constant value and concretize any generator
  // references inside it. Multi-versioning is disallowed.
  ErrorTreeOr<Attribute> concrete =
      concretizeSymbolsWithin(value, parent, op.getLoc());
  if (concrete.isError()) {
    parent->setToError(concrete.takeError());
    return ElaborationState::error();
  }
  value = cast_or_null<TypedAttr>(concrete.takeValue());
  if (!value)
    return ElaborationState::skipNode();

  op.getResult().setType(value.getType());
  op.setValueAttr(value);
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// Elaborator::instantiateGeneratorReference
//===----------------------------------------------------------------------===//

std::pair<ElaborationState, ImplNode *>
Elaborator::instantiateGeneratorReference(
    ImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
    ParameterExprArrayAttr &inputParamKey, GeneratorOp &gen,
    function_ref<bool(ParamNode *)> shouldWait) {
  // Lookup the callee.
  StringAttr name = cast<FlatSymbolRefAttr>(calleeSymbol.getSymbol()).getAttr();
  Operation *calleeOp = oldSymTab.lookup(name);

  if (!calleeOp) {
    FuncOp func =
        concreteFuncs.read([name](auto &map) { return map.at(name); });
    ImplNode *node =
        g.concreteNodes.read([func](auto &map) { return map.at(func); });
    return {ElaborationState::advance(), node};
  }

  // Add in the mapping for parameters in the calls.
  inputParamKey = ParameterExprArrayAttr::get(user->getContext(),
                                              calleeSymbol.getParamValues());

  // If we already have a binding for this, we're done.
  gen = cast<GeneratorOp>(calleeOp);

  // Check for excessive instantiation depth.
  if (parent->parent->depth > config.maxDepth) {
    parent->setToError(ErrorTree(parent->parent->gen.getLoc(),
                                 "elaborator expansion is " +
                                     Twine(config.maxDepth + 1) +
                                     " levels deep - infinite recursion?"));
    return {ElaborationState::error(), nullptr};
  }

  // Find the tree node that corresponds to the thing we're calling.
  ParamNode *calleeNode =
      g.getOrCreate(runtime, inputParamKey, gen, parent->parent->depth + 1);
  ElaborationState result = specializeGenerator(
      parent, calleeNode, parent->parent, shouldWait(calleeNode));
  if (result.shouldSkipNode())
    return {ElaborationState::skipNode(), nullptr};

  FailureOr<ImplNode *> concrete =
      collectConcreteImplementations(user, parent, calleeNode);
  if (failed(concrete))
    return {failure(), nullptr};
  return {ElaborationState(success()), *concrete};
}

//===----------------------------------------------------------------------===//
// Elaborator::collectConcreteImplementations
//===----------------------------------------------------------------------===//

FailureOr<ImplNode *>
Elaborator::collectConcreteImplementations(Operation *user, ImplNode *parent,
                                           ParamNode *calleeNode) {
  // Get all valid implementations of the callee node.
  ErrorTreeOr<ImplNode *> concrete = calleeNode->getFirstConcreteNode();
  if (concrete.isError()) {
    // If the callee has no parameters, don't build another error.
    if (calleeNode->inputParams.empty()) {
      parent->setToError(concrete.takeError());
    } else {
      parent->setToError(ErrorTree(user->getLoc(), "call expansion failed",
                                   concrete.takeError()));
    }
    return failure();
  }

  return concrete.takeValue();
}

//===----------------------------------------------------------------------===//
// Elaborator::processGeneratorUser
//===----------------------------------------------------------------------===//

ElaborationState
Elaborator::processGeneratorUser(GeneratorUserOpInterface user,
                                 SymbolConstantAttr calleeSymbol,
                                 ImplNode *parent) {
  // Not all operations can verify their callee type, if for instance, it is a
  // generic type. Verify here as a fallback.
  if (!calleeSymbol.getType().getInputParamTypes().empty()) {
    parent->setToError(
        ErrorTree(user.getLoc(), "cannot reference parametric function"));
    return ElaborationState::error();
  }

  ParameterExprArrayAttr inputParamKey;
  GeneratorOp gen;
  bool wasSkipped = false;
  ParamNode *calleeNode;
  auto [result, concrete] = instantiateGeneratorReference(
      parent, user, calleeSymbol, inputParamKey, gen, [&](ParamNode *genNode) {
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
  return completeCallProcessing(user, concrete, parent);
}

//===----------------------------------------------------------------------===//
// Elaborator::completeCallProcessing
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
ElaborationState Elaborator::processParamApplyOp(ImplNode *inode,
                                                 ParamApplyOp op, FuncOp func) {
  // First concretize the operands.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getOperandsAttr());

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = cast<ParameterExprArrayAttr>(value);
  TypedAttr &cached = lookupCachedInterpretation(func, operandsAttr);
  if (!cached) {
    ErrorTreeOr<Attribute> operandsOr =
        concretizeSymbolsWithin(operandsAttr, inode, op.getLoc());
    if (operandsOr.isError()) {
      inode->setToError(operandsOr.takeError());
      return failure();
    }
    operandsAttr = cast_or_null<ParameterExprArrayAttr>(operandsOr.takeValue());
    if (!operandsAttr)
      return ElaborationState::skipNode();

    ErrorTreeOr<TypedAttr> result =
        inode->getEvaluator().evaluateFunction(func, operandsAttr);
    if (result.isError()) {
      inode->setToError(result.takeError());
      return failure();
    }
    cached = result.takeValue();
    writeGlobalCachedInterpretation(func, operandsAttr, cached);
  }

  // Bind the result and erase the operation.
  inode->getEvaluator().setParameterValue(op.getParamDecl(), cached);
  op.erase();
  return ElaborationState::advance();
}

ElaborationState
Elaborator::completeCallProcessing(GeneratorUserOpInterface user,
                                   ImplNode *thisNode, ImplNode *node) {
  if (thisNode->error) {
    if (thisNode->parent->inputParams.empty()) {
      node->setToError(thisNode->error->copy());
    } else {
      node->setToError(ErrorTree(user.getLoc(), "call expansion failed",
                                 thisNode->error->copy()));
    }
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

  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// Elaborator::processCallOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
ElaborationState Elaborator::processCallOp(ImplNode *parent,
                                           GeneratorUserOpInterface call) {
  Attribute symbol;
  HANDLE_EVALUATOR_CONC(symbol, parent, call.getLoc(), call.getCallee());
  return processGeneratorUser(call, cast<SymbolConstantAttr>(symbol), parent);
}

//===----------------------------------------------------------------------===//
// Locations and DebugInfo
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
  LocationAttr loc = argOrOp.getLoc();
  if (LocationAttr newLocAttr = concretizeAttr<LocationAttr>(loc, loc, inode)) {
    argOrOp.setLoc(newLocAttr);
    return success();
  }
  return failure();
}

static LogicalResult
concretizeLocsInScope(iterator_range<Block::iterator> scope, ImplNode *inode) {
  for (Operation &op : scope) {
    op.walk([&](Operation *op) {
      if (failed(concretizeLocOf(*op, inode)))
        return WalkResult::interrupt();

      // Update the ValueInfo attr since they contain types.
      if (isa<DebugInfo::ValueOp, DebugInfo::KillOp>(op)) {
        op->setAttrs(
            concretizeAttr(op->getAttrDictionary(), op->getLoc(), inode));
        return WalkResult::advance();
      }

      // To be defensive, we only concretize location attributes if we know
      // what we are dealing with.
      if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op)) {
        if (LocationAttr callLoc = inlined.getCallLocAttr()) {
          inlined.setCallLocAttr(
              concretizeAttr<LocationAttr>(callLoc, op->getLoc(), inode));
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

      // Walk over nested scopes.
      if (isa<DeclInterface>(op))
        return WalkResult::skip();

      return WalkResult::advance();
    });
  }
  return success(!inode->error);
}

/// Concretizes the locations of all operations within scope bound by the
/// specified block.
static LogicalResult concretizeLocsInScope(Block &scope, ImplNode *inode) {
  return concretizeLocsInScope({scope.begin(), scope.end()}, inode);
}

//===----------------------------------------------------------------------===//
// Elaborator::processParamIfOp
//===----------------------------------------------------------------------===//

/// We always erase this op and its nested scopes from the parameter graph -
/// it's been handled, and we don't want anyone else touching it later
/// considering we're about to delete the op itself.
static void recursivelyEraseFromNestedScopes(ImplNode *node, Operation *op) {
  ParameterUseDefGraph &paramGraph = node->paramGraph;
  auto eraseScopes = [op](ParameterUseDefGraph &graph) mutable {
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
  eraseScopes(paramGraph);
  for (auto &[scope, graph] : paramGraph.nestedScopes)
    eraseScopes(graph);
}

ElaborationState Elaborator::processParamIfOp(ImplNode *parent, ParamIfOp op) {
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

  // Push a new node and skip over the current frame until it completes.
  ImplNode::WorkItem item{{}, nullptr, parent->getEvaluator()};
  collectOpsToProcessInside(toProcess, parent, item.ops);

  // When the nested scope completes processing, finish processing the current
  // parameter if.
  item.onComplete = [resultBool, debug = config.elaborateDebugInfo](
                        ImplNode *node) -> LogicalResult {
    assert(node->stack.size() >= 2 && "expected at least two work items");
    // Retrieve the current state.
    ImplNode::WorkItem &parentFrame = *std::next(node->stack.rbegin());
    auto op = cast<ParamIfOp>(parentFrame.ops.back());

    // Splice the ops into the parent. Grab the terminator before the iterators
    // invalidate.
    Block::iterator iter = op->getIterator();
    Block &block = op->getRegion(!resultBool).front();

    // First update the locations if necessary
    if (debug && failed(concretizeLocsInScope(block, node)))
      return failure();

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

    // The callback to the current frame finishes processing the current
    // operation, so take it off the parent frame's worklist.
    recursivelyEraseFromNestedScopes(node, op);
    op->erase();
    parentFrame.ops.pop_back();
    return success();
  };

  parent->stack.push_back(std::move(item));
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// Elaborator::processParamForOp
//===----------------------------------------------------------------------===//

ElaborationState Elaborator::processParamForOp(ImplNode *parent,
                                               ParamForOp op) {
  // First, concretize the initializer and sequence generator expressions and
  // result types.
  Attribute initial, iterate;
  SmallVector<Type> resultTypes;
  HANDLE_EVALUATOR_CONC(initial, parent, op.getLoc(), op.getInitial());
  HANDLE_EVALUATOR_CONC(iterate, parent, op.getLoc(), op.getIterate());
  for (Type type : op.getResultTypes()) {
    HANDLE_EVALUATOR_CONC(resultTypes.emplace_back(), parent, op.getLoc(),
                          type);
  }

  // Concretize the sequence generator function.
  ErrorTreeOr<FuncOp> func = getConcreteFunction(
      parent, op.getLoc(), cast<SymbolConstantAttr>(iterate));
  if (func.isError()) {
    parent->setToError(func.takeError());
    return failure();
  }
  if (!*func)
    return ElaborationState::skipNode();

  if (LLVM_UNLIKELY(!FuncOp(*func).getSignature().hasMemoryOnlyResult())) {
    parent->setToError(
        ErrorTree(op.getLoc(),
                  "INTERNAL ERROR: iterator should have memory-only result"));
    return failure();
  }

  // Generate the series of values.
  auto iterator = cast<TypedAttr>(initial);
  SmallVector<TypedAttr> values;
  while (true) {
    iterator =
        StoreToMemAttr::get(iterator, PointerType::get(iterator.getType()));
    ErrorTreeOr<TypedAttr> result =
        parent->getEvaluator().evaluateFunctionWithResultSlot(*func, iterator);
    if (result.isError()) {
      parent->setToError(result.takeError());
      return failure();
    }
    auto structAttr = dyn_cast<StructAttr>(*result);
    if (LLVM_UNLIKELY(!structAttr || structAttr.getValues().size() != 3)) {
      parent->setToError(ErrorTree(
          op.getLoc(), "INTERNAL ERROR: expected a struct of 3 elements"));
      return failure();
    }
    if (cast<BoolAttr>(structAttr.getValues()[2]).getValue())
      break;
    values.push_back(structAttr.getValues()[1]);
    iterator = structAttr.getValues()[0];
  }

  // Now generate the loop bodies and set up their elaboration at the same time.
  // Start by taking the current op off the worklist. It will be deleted by the
  // end of this function.
  parent->stack.back().ops.pop_back();

  // Schedule the ops in the else region, which are always generated. They are
  // processed in the same scope as the parent. The else job is responsible for
  // cleaning up dead IR because it runs last.
  Block *elseBlock = &op.getElseRegion().front();
  auto yield = cast<ParamYieldOp>(elseBlock->getTerminator());
  auto onElseComplete = [debug = config.elaborateDebugInfo,
                         begin = &*elseBlock->begin(), yield, op,
                         parent](ImplNode *node) mutable -> LogicalResult {
    if (debug && failed(concretizeLocsInScope(
                     {begin->getIterator(), yield->getIterator()}, node)))
      return failure();
    // Erase the terminator when elaboration of the else region is done.
    yield.erase();
    recursivelyEraseFromNestedScopes(parent, op);
    op.erase();
    return success();
  };
  ImplNode::WorkItem elseItem{
      {}, std::move(onElseComplete), parent->getEvaluator()};
  collectOpsToProcessInside(op.getElseRegion(), parent, elseItem.ops);
  parent->stack.push_back(std::move(elseItem));

  // Lower the `kgen.param.for` into an outer loop and wrapper loops for each
  // generated iteration. This way, we can lower `continue` to a break to the
  // wrapper loop to model exiting a single iteration and lower `break` to a
  // break to the outer loop to model exiting the whole loop.
  mlir::IRRewriter b{OpBuilder(op)};
  StringAttr outerLabel = b.getStringAttr("param_for_outer");
  auto outerLoop = b.create<HLCF::LoopOp>(op.getLoc(), resultTypes, outerLabel);
  b.createBlock(&outerLoop.getBody());

  // Upon completion of elaboration of each such generated loop, replace the
  // `kgen.param.for` terminators with the appropriate HLCF ones.
  auto makeCompletion =
      [debug = config.elaborateDebugInfo,
       outerLabel](Region &region) -> std::function<LogicalResult(ImplNode *)> {
    return [debug, &region, outerLabel](ImplNode *node) -> LogicalResult {
      if (debug && failed(concretizeLocsInScope(region.front(), node)))
        return failure();

      // Replace the `kgen.param.for` terminators with the HLCF equivalent.
      region.walk([&](Operation *op) {
        if (isa<ParamForOp>(op))
          return WalkResult::skip();
        if (isa<ParamForBreakOp>(op)) {
          mlir::IRRewriter b{OpBuilder(op)};
          b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(),
                                              outerLabel);
          return WalkResult::advance();
        }
        if (isa<ParamForContinueOp>(op)) {
          mlir::IRRewriter b{OpBuilder(op)};
          b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands());
          return WalkResult::advance();
        }
        return WalkResult::advance();
      });
      return success();
    };
  };

  // Compute the ops that need to be processed in the body.
  std::vector<Operation *> opsToRewrite;
  collectOpsToProcessInside(op.getBody(), parent, opsToRewrite);
  ParamDeclAttr decl = op.getParamDecl();

  auto replaceArgs = [](Region &body, ValueRange values) {
    // Replace the arguments with the results of the previous loop. Then erase
    // the arguments.
    for (auto [arg, res] : llvm::zip(body.getArguments(), values))
      arg.replaceAllUsesWith(res);
    body.front().eraseArguments(0, body.getNumArguments());
  };

  IRMapping mapping;
  auto &nestedScopes = parent->paramGraph.nestedScopes;
  SmallVector<DeclInterface> nestedDecls;
  op.getBody().walk([&](DeclInterface decl) { nestedDecls.push_back(decl); });
  IREvaluator evaluator = parent->getEvaluator();

  // Forward the result of one iteration into the next.
  ValueRange nextOperands = op.getOperands();
  for (TypedAttr value : values) {
    // Create the loop op for this iteration and clone the body into it.
    auto loop = b.create<HLCF::LoopOp>(op.getLoc(), resultTypes);
    mapping.clear();
    op.getBody().cloneInto(&loop.getBody(), mapping);
    replaceArgs(loop.getBody(), nextOperands);
    nextOperands = loop.getResults();

    // Map the ops to rewrite from the original body into the clone one.
    ImplNode::WorkItem nextItem{{}, makeCompletion(loop.getBody()), evaluator};
    for (Operation *op : opsToRewrite)
      nextItem.ops.push_back(mapping.lookup(op));
    // If any DeclInterface got cloned, we also have to make sure to clone its
    // parameter use-def list.
    for (DeclInterface decl : nestedDecls) {
      Operation *cloned = mapping.lookup(decl);
      for (auto [declRegion, clonedRegion] :
           llvm::zip(decl->getRegions(), cloned->getRegions()))
        nestedScopes.try_emplace(&clonedRegion,
                                 nestedScopes.at(&declRegion).copy(mapping));
    }

    // Now schedule the work item for this body.
    nextItem.evaluator.setParameterValue(decl, value);
    parent->stack.push_back(std::move(nextItem));
  }

  b.inlineBlockBefore(elseBlock, b.getInsertionBlock(), b.getInsertionPoint(),
                      nextOperands);
  b.create<HLCF::BreakOp>(op.getLoc(), yield.getOperands(), outerLabel);
  op.replaceAllUsesWith(outerLoop.getResults());
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// Elaborator::processScope
//===----------------------------------------------------------------------===//

void Elaborator::completeImplNodeProcessing(ImplNode *inode) {
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
      // Otherwise, get all bound nodes.
      FailureOr<ImplNode *> concrete =
          collectConcreteImplementations(call, inode, genNode);
      if (failed(concrete))
        break;
      // Process the multiple concrete nodes. If this causes multi-versioning,
      // the forks will correctly get rescheduled on the worklists with no
      // stacks, and then immediately fallthrough to this function.
      ElaborationState result = completeCallProcessing(call, *concrete, inode);
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

void Elaborator::scheduleImplNode(ImplNode *inode) {
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

LogicalResult Elaborator::processImplNode(ImplNode *inode) {
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

  VerboseCompilerTimeTraceScope traceScope(
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
      assert(inode->stack.size() > size && "skip with no new frame");
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

ElaborationState Elaborator::processScope(ImplNode *node,
                                          ImplNode::WorkItem &item) {
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
  return ElaborationState::advance();
}

ElaborationState Elaborator::processOp(ImplNode *node, Operation *op) {
  if (Block *block = op->getBlock())
    if (!block->isEntryBlock())
      return ElaborationState::advance();

  if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
    return processParamDeclareOp(node, declare);
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
  } else if (auto forOp = dyn_cast<ParamForOp>(op)) {
    return processParamForOp(node, forOp);
  } else if (auto call = dyn_cast<GeneratorUserOpInterface>(op)) {
    return processCallOp(node, call);
  } else if (isa<DebugInfo::ValueOp, DebugInfo::KillOp>(op)) {
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
// Elaborator::specializeGenerator
//===----------------------------------------------------------------------===//

ElaborationState Elaborator::specializeGenerator(ImplNode *inode,
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

  GeneratorOp gen = genNode->gen;

  // Bind all parameter values in this scope.
  ArrayRef<TypedAttr> inputParamValues = genNode->inputParams.getValue();
  ArrayRef<ParamDeclAttr> inputParamDecls = gen.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  VerboseCompilerTimeTraceScope traceScope("specializeGenerator: " +
                                           gen.getSymName().str());

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
  concreteFuncs.modify([newFunc, mangledName](auto &map) {
    map.try_emplace(mangledName, newFunc);
  });

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
  auto childNode = std::make_unique<ImplNode>(
      newFunc, genNode, std::move(childGraph), std::move(baseName));
  g.concreteNodes.modify(
      [newFunc, node = childNode.get()](DenseMap<FuncOp, ImplNode *> &map) {
        map.try_emplace(newFunc, node);
      });
  ImplNode *newFuncNode = childNode.get();
  genNode->impl = std::move(childNode);
  ParameterUseDefGraph &uses = newFuncNode->paramGraph;

  // Kick off the expansion for the new function.
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
    name = DebugInfo::SourceNameAttr::get(
        name.getName(), name.getParamTypes(), name.getArgTypes(), paramValues,
        name.getParent(), name.getKind(), name.getDecorators());
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
    onComplete = [](ImplNode *inode) -> LogicalResult {
      if (failed(concretizeLocOf(*inode->func, inode)))
        return failure();
      if (failed(concretizeLocsInScope(*inode->func.getBody(), inode)))
        return failure();
      return success();
    };
  } else {
    onComplete = [](ImplNode *) { return success(); };
  }

  IREvaluator evaluator(*this, newFuncNode);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setParameterValue(decl, val);

  ImplNode::WorkItem item{std::move(opsToRewrite), std::move(onComplete),
                          std::move(evaluator)};
  newFuncNode->stack.push_back(std::move(item));

  if (addWaiter) {
    [[maybe_unused]] bool added = genNode->state.addWaiter();
    assert(added);
    genNode->andThenSync([inode, this] { scheduleImplNode(inode); });
  }
  assert(genNode->numActive == 0 && "expected first implementation");
  initialScheduleImplNode(newFuncNode);
  return ElaborationState::skipNode();
}

//===----------------------------------------------------------------------===//
// Elaborator::diagnoseAndBreakRecursion
//===----------------------------------------------------------------------===//

bool Elaborator::diagnoseAndBreakRecursion(unsigned generation,
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
      // This `genNode` is cyclic. Handle the cycle. Break the cycle.
      (void)completeCallProcessing(call, genNode->impl.get(), inode);
      completed.set(idx);
      anyBroken = true;
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

    visitImplNode(pnode->impl.get());
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
// Elaborator::run
//===----------------------------------------------------------------------===//

LogicalResult Elaborator::run(ModuleOp theModule,
                              ArrayRef<GeneratorOp> primaryGenerators) {
  MLIRContext *ctx = theModule.getContext();

  // Find any kgen.func we have already - they're already elaborated, and we do
  // not want to re-process them. Add concrete ImplNodes for each one.
  for (FuncOp func : theModule.getOps<FuncOp>()) {
    addConcreteFunc(func);
    concreteFuncs.get().try_emplace(func.getSymNameAttr(), func);
  }

  auto emptyInputParamKey = ParameterExprArrayAttr::get(ctx, {});
  std::vector<AnyAsyncValueRef> primaryChs;
  std::vector<std::unique_ptr<ImplNode>> rootNodes;
  std::vector<ParamNode *> primaryNodes;
  primaryChs.reserve(primaryGenerators.size());
  primaryNodes.reserve(primaryGenerators.size());
  for (GeneratorOp gen : primaryGenerators) {
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
    VerboseCompilerTimeTraceScope traceScope("doElaboration");
    unsigned cycleGeneration = 0;
    while (true) {
      signalWorklist();
      AsyncRT::await(g.worklistCh);
      assert(g.numWorkItems == 0);

      // Check if all primary generators are done. If so, break.
      if (llvm::all_of(primaryChs, [](auto &ch) { return ch.isReady(); }))
        break;
      g.numWorkItems = 1;

      // Re-initialize the worklist chain.
      g.worklistCh = AsyncValueRef<Chain>::allocate(runtime);

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
      err.takeError().emit([](Location loc) { return mlir::emitError(loc); },
                           "call expansion failed");
    }
  }
  if (failed) {
    for (FuncOp func : llvm::make_second_range(concreteFuncs.get()))
      func.erase();

    return failure();
  }

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  // Sort instantiations of each generator to ensure we have a deterministic
  // output in multithreaded execution.
  struct SuccessfulFuncs {
    std::string paramStr;
    FuncOp func;
  };
  auto *newBlock = new Block;
  llvm::MapVector<GeneratorOp, std::vector<SuccessfulFuncs>> genInstantiations;
  for (Operation &op : llvm::make_early_inc_range(*theModule.getBody())) {
    if (auto gen = dyn_cast<GeneratorOp>(op)) {
      genInstantiations[gen];
    } else {
      op.remove();
      newBlock->push_back(&op);
    }
  }
  for (ParamNode &node :
       llvm::make_pointee_range(llvm::make_second_range(g.nodes.get()))) {
    VerboseCompilerTimeTraceScope traceScope(
        "processGen", [name = node.gen.getSymName()] { return name.str(); });
    // Erase all erroneous functions.
    if (node.impl->error) {
      node.impl->func.erase();
      continue;
    }

    genInstantiations[node.gen].push_back(
        SuccessfulFuncs{mlir::debugString(node.inputParams), node.impl->func});
  }

  // Now reorder all instantiations of each generator to be deterministic.
  for (auto &[gen, instantiations] : genInstantiations) {
    llvm::sort(instantiations, [](auto &lhs, auto &rhs) {
      return lhs.paramStr < rhs.paramStr;
    });
    for (auto &[_, func] : instantiations)
      newBlock->push_back(func);
  }

  // Sort and then push on all the deferred functions.
  llvm::sort(deferredSymbols, [](FuncOp lhs, FuncOp rhs) {
    return lhs.getSymName() < rhs.getSymName();
  });
  for (FuncOp func : deferredSymbols)
    newBlock->push_back(func);

  // Update the symbol table with the new one.
  theModule.getBody()->erase();
  theModule.getBodyRegion().push_back(newBlock);
  // Recompute the new symbol table.
  oldSymTab = SymbolTable(theModule);
  return success();
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
                          TargetInfoAttr target = nullptr,
                          ElaboratorCompileAsmFn compileAsmFn = {})
      : ElaborateGeneratorsBase(options), target(target),
        compileAsmFn(std::move(compileAsmFn)) {}

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

    // Default compile assembly hook will just error.
    if (!compileAsmFn) {
      compileAsmFn = +[](GeneratorOp, SymbolConstantAttr, StringAttr,
                         const SymbolTable &, TargetInfoAttr, EmissionKind) {
        return Error("internal error: cannot compile assembly without a JIT");
      };
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp theModule = getOperation();

    auto &symtab = getAnalysis<mlir::SymbolTableAnalysis>();
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

    // Root elaboration on exports and global variables. These are the
    // generators that elaboration will start from. If there are no such
    // generators, then elaborate anything with no input parameters.
    DenseSet<GeneratorOp> roots;
    auto addAsRoot = [&](SymbolRefAttr ref) {
      roots.insert(symtab.getTopLevelSymbolTable().lookup<GeneratorOp>(
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

    ElaboratorCallbacks callbacks{compileAsmFn};
    ElaborateGeneratorsOptions config{enableSearch, allowMultiplePrimaryImpls,
                                      maxDepth, elaborateDebugInfo,
                                      diagAllFailures};

    VerboseCompilerTimeTraceScope traceScope("elaborate-generators");

    // Now, construct and run the elaborator.
    Elaborator impl(symtab.getTopLevelSymbolTable(), paramCache, target,
                    std::move(callbacks), config);
    if (failed(impl.run(theModule, primaryGenerators)))
      return signalPassFailure();
  }

private:
  /// The compilation target.
  TargetInfoAttr target;
  /// The functor used to compile a module to assembly.
  ElaboratorCompileAsmFn compileAsmFn;
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(TargetInfoAttr target,
                                const ElaborateGeneratorsOptions &options,
                                ElaboratorCompileAsmFn compileAsmFn) {
  return std::make_unique<ElaborateGeneratorsPass>(options, target,
                                                   std::move(compileAsmFn));
}
