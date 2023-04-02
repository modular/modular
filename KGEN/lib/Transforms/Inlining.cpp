//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Inlining.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITOps.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "Support/TimeProfiler.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RWMutex.h"

#define DEBUG_TYPE "kgen-inlining"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// AttrTypeMangler
//===----------------------------------------------------------------------===//

namespace {
/// Signature types define a nested parameter scope inside a parameter
/// expression. Manually walk and mangle parameter references in attributes and
/// types in an expression tree while accounting for name shadowing in a
/// signature type.
class AttrTypeMangler {
public:
  explicit AttrTypeMangler(DenseSet<const void *> &manglerCache)
      : manglerCache(manglerCache) {}

  /// Mangle references within a type.
  Type mangleRefsIn(Type type, bool &hasRefs) {
    return mangleRefsInImpl(type, hasRefs);
  }
  Type mangleRefsIn(Type type) {
    bool unused;
    return mangleRefsIn(type, unused);
  }

  /// Mangle references within an attribute.
  Attribute mangleRefsIn(Attribute attr, bool &hasRefs);
  Attribute mangleRefsIn(Attribute type) {
    bool unused;
    return mangleRefsIn(type, unused);
  }

  /// Populate the mangler using the decls in two potentially conflicting
  /// scopes. Returns false if there is nothing to mangle.
  bool populate(Builder &b, const ParameterUseDefGraph &curScope,
                const llvm::SetVector<StringAttr> &calleeDecls);

  /// Optionally mangle a declaration.
  ParamDeclAttr mangleDecl(ParamDeclAttr decl, bool needsMangling);

  /// Mangle attributes, types, and locations in an operation.
  void mangleElementsIn(Operation *op);

  /// Recursively mangle declarations in the nested scope.
  void recursivelyMangle(Region *scope, const ParameterUseDefGraph &graph);

private:
  template <typename T, typename U = std::conditional_t<
                            std::is_base_of_v<Type, T>, Type, Attribute>>
  U mangleRefsInImpl(T value, bool &hasRefs) {
    if (manglerCache.contains(value.getAsOpaquePointer()))
      return value;

    SmallVector<Attribute, 16> replAttrs;
    SmallVector<Type, 16> replTypes;
    bool changed = false;
    bool hasNestedRefs = false;
    value.walkImmediateSubElements(
        [&](Attribute attr) {
          Attribute result = mangleRefsIn(attr, hasNestedRefs);
          replAttrs.push_back(result);
          changed |= result != attr;
        },
        [&](Type type) {
          Type result = mangleRefsIn(type, hasNestedRefs);
          replTypes.push_back(result);
          changed |= result != type;
        });

    hasRefs |= hasNestedRefs;
    if (!hasNestedRefs)
      manglerCache.insert(value.getAsOpaquePointer());
    return changed ? value.replaceImmediateSubElements(replAttrs, replTypes)
                   : value;
  }

  /// The map of mangled declarations.
  DenseMap<StringAttr, StringAttr> mangledDecls;
  /// A cache of attributes and types known to have no parameter references.
  DenseSet<const void *> manglerCache;
};
} // namespace

Attribute AttrTypeMangler::mangleRefsIn(Attribute attr, bool &hasRefs) {
  if (auto ref = dyn_cast<ParamDeclRefAttr>(attr)) {
    hasRefs = true;
    if (StringAttr mangled = mangledDecls.lookup(ref.getName()))
      return ParamDeclRefAttr::get(mangled,
                                   mangleRefsIn(ref.getType(), hasRefs));
  }
  return mangleRefsInImpl(attr, hasRefs);
}

bool AttrTypeMangler::populate(Builder &b, const ParameterUseDefGraph &curScope,
                               const llvm::SetVector<StringAttr> &calleeDecls) {
  TimeTraceScope<> traceScope("AttrTypeMangler::populate");

  // This uniquing scheme involves splitting each decl name into a key string
  // and a substring of trailing digits. We track the max of such digits of the
  // same key string and use that to generate the next unique ID.
  llvm::StringMap<ssize_t> maxIds;
  auto getId = [&](StringRef name) {
    size_t splitIdx = llvm::count_if(llvm::reverse(name),
                                     [](char c) { return std::isdigit(c); });
    splitIdx = name.size() - splitIdx;
    // -1 means no number suffix.
    ssize_t id = -1;
    name.substr(splitIdx).getAsInteger(/*Radix=*/10, id);
    return std::make_pair(name.substr(0, splitIdx), id);
  };

  bool needsMangling = false;
  // `curScope` contains all declarations visible in the scope of the call,
  // including those defined in higher scopes. When the function is inlined,
  // these are the declarations that will project into the inlined body. We need
  // to mangle parameters in the inlined body such that they do not collide with
  // any declarations visible in the call scope.
  for (StringAttr decl : calleeDecls) {
    if (curScope.decls.find(decl) == curScope.decls.end()) {
      // This declaration will not collide.
      continue;
    }
    if (!needsMangling) {
      // Lazily populate the IDs;
      auto updateMaxId = [&](StringRef name) {
        auto [key, id] = getId(name);
        ssize_t &max = maxIds.try_emplace(key, -1).first->second;
        max = std::max(max, id);
      };
      for (StringAttr name : calleeDecls)
        updateMaxId(name);
      for (auto &[decl, _] : curScope.decls)
        updateMaxId(decl);
    }
    // Generate a new ID by taking the max number.
    auto [key, _] = getId(decl);
    ssize_t newId = ++maxIds[key];
    mangledDecls.try_emplace(decl, b.getStringAttr(key + Twine(newId)));
    needsMangling = true;
  }
  return needsMangling;
}

ParamDeclAttr AttrTypeMangler::mangleDecl(ParamDeclAttr decl,
                                          bool needsMangling) {
  if (!needsMangling)
    return decl;
  Type type = mangleRefsIn(decl.getType());
  if (StringAttr mangled = mangledDecls.lookup(decl.getName()))
    return ParamDeclAttr::get(mangled, type);
  if (type == decl.getType())
    return decl;
  return ParamDeclAttr::get(decl.getName(), type);
}

void AttrTypeMangler::mangleElementsIn(Operation *op) {
  TimeTraceScope<> traceScope("AttrTypeMangler::mangleElementsIn");

  op->setAttrs(cast<DictionaryAttr>(mangleRefsIn(op->getAttrDictionary())));
  op->setLoc(
      cast<mlir::LocationAttr>(mangleRefsIn(mlir::LocationAttr(op->getLoc()))));

  for (OpResult result : op->getResults())
    result.setType(mangleRefsIn(result.getType()));

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments()) {
        arg.setLoc(cast<mlir::LocationAttr>(
            mangleRefsIn(mlir::LocationAttr(arg.getLoc()))));
        arg.setType(mangleRefsIn(arg.getType()));
      }
    }
  }
}

void AttrTypeMangler::recursivelyMangle(Region *scope,
                                        const ParameterUseDefGraph &graph) {
  TimeTraceScope</*Enabled=*/false> traceScope(
      "AttrTypeMangler::recursivelyMangle");

  // Exit early if the scope is parametrically isolated.
  if (cast<DeclInterface>(scope->getParentOp())
          .isIsolatedFromAbove(scope->getRegionNumber()))
    return;

  const ParameterUseDefGraph &uses = graph.nestedScopes.find(scope)->second;
  AttrTypeMangler mangler(manglerCache);
  bool empty = true;
  for (ParamDeclRefAttr ref : uses.usesFromAbove) {
    if (StringAttr mangled = mangledDecls.lookup(ref.getName())) {
      mangler.mangledDecls.try_emplace(ref.getName(), mangled);
      empty = false;
    }
  }
  // Exit early if there is nothing to mangle.
  if (empty)
    return;

  for (Operation *op : uses.paramOps) {
    if (op == scope->getParentOp())
      continue;
    mangler.mangleElementsIn(op);
  }
  for (auto &[_, decl] : uses.decls) {
    if (!scope->getParentOp()->isProperAncestor(decl.declOp))
      continue;
    mangler.mangleElementsIn(decl.declOp);
  }
  for (Region *nestedScope : uses.nestedDecls)
    mangler.recursivelyMangle(nestedScope, graph);
}

/// Insert a new parameter declaration into all nested declaration scopes.
static void propagateNewDecls(ArrayRef<ParamDeclAttr> newDecls,
                              ParameterUseDefGraph &topLevelGraph,
                              ParameterUseDefGraph &graph, Operation *declOp,
                              Region *declScope) {
  // Populate the new declarations into the call scope graph.
  for (ParamDeclAttr decl : newDecls) {
    graph.decls.try_emplace(
        decl.getName(), ParamDeclaration{decl.getType(), declOp, declScope});
  }
  // Recurse on nested scopes.
  for (Region *nestedDecl : graph.nestedDecls) {
    propagateNewDecls(newDecls, topLevelGraph,
                      topLevelGraph.nestedScopes.find(nestedDecl)->second,
                      declOp, declScope);
  }
}

//===----------------------------------------------------------------------===//
// inlineGeneratorCall
//===----------------------------------------------------------------------===//

/// Get the nearest declaration from the operation and the region of the
/// declaration that contains the operation.
static Region *getNearestDeclRegion(Operation *op) {
  Region *region = op->getParentRegion();
  auto decl = dyn_cast<DeclInterface>(region->getParentOp());
  while (!decl) {
    region = region->getParentRegion();
    decl = dyn_cast<DeclInterface>(region->getParentOp());
  }
  return region;
}

/// Generator inputs and results cross parameter domains. Make sure to rebind
/// them if necessary.
static SmallVector<Value> rebindValues(OpBuilder &b, Location loc,
                                       ValueRange inputs, TypeRange outputs) {
  SmallVector<Value> newValues;
  for (auto [input, output] : llvm::zip(inputs, outputs))
    if (input.getType() != output)
      newValues.push_back(b.create<RebindOp>(loc, output, input));
    else
      newValues.push_back(input);
  return newValues;
}

/// The operands of returns cross parameter domains. Make sure to rebind them if
/// necessary.
static SmallVector<Value>
rebindReturnOperands(OpBuilder &b, Operation *newReturn, Operation *call) {
  return rebindValues(b, newReturn->getLoc(), newReturn->getOperands(),
                      call->getResultTypes());
}

void KGEN::inlineGeneratorCall(CallOp call, GeneratorOp callee,
                               ParameterUseDefGraph &topLevelGraph,
                               const ParameterUseDefGraph &calleeParams,
                               const llvm::SetVector<StringAttr> &calleeDecls,
                               DenseSet<const void *> &manglerCache) {
  assert(callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled);
  TimeTraceScope<> traceScope("callee",
                              [&] { return callee.getSymName().str(); });

  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  // Get the parameters in-scope at the callsite.
  Region *scopeRegion = getNearestDeclRegion(call);
  ParameterUseDefGraph *callScope =
      scopeRegion == topLevelGraph.scope
          ? &topLevelGraph
          : &topLevelGraph.nestedScopes.find(scopeRegion)->second;

  mlir::IRRewriter b{OpBuilder(call)};
  // Use a LoopOp to be able to break to a label - any returns inlined from
  // callee must only exit the inlined block.
  auto scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                      ValueRange(), label);
  b.createBlock(&scope.getBody());

  AttrTypeMangler mangler(manglerCache);
  bool needsMangling = mangler.populate(b, *callScope, calleeDecls);

  // Make sure to rebind the call operands based on the mangled types of the
  // callee's argument types.
  SmallVector<Type> argTypes = llvm::to_vector(callee.getArgumentTypes());
  if (needsMangling)
    for (Type &type : argTypes)
      type = mangler.mangleRefsIn(type);

  SmallVector<Value> argVals =
      rebindValues(b, call.getLoc(), call->getOperands(), argTypes);

  // Materialize any constraints on the callee as asserts.
  for (ConstraintAttr constraint : callee.getConstraints()) {
    auto assertOp = b.create<ParamAssertOp>(
        constraint.getLoc(), constraint.getExpr(),
        StringAttr::get(constraint.getMessage().getValue(),
                        StringType::get(b.getContext())));
    if (needsMangling)
      mangler.mangleElementsIn(assertOp);
  }

  // Map the callee inputs.
  IRMapping map;
  for (auto [value, arg] : llvm::zip(argVals, callee.getArguments()))
    map.map(arg, value);
  for (Operation &op : *callee.getBody())
    b.clone(op, map);

  // Clone the nested parameter use-def graphs into the current set of
  // nested graphs.
  callee.walk([&](DeclInterface containedScope) {
    if (containedScope == callee)
      return;
    Operation *clonedScope = map.lookup(&*containedScope);
    for (auto [region, clonedRegion] :
         llvm::zip(containedScope->getRegions(), clonedScope->getRegions())) {
      const ParameterUseDefGraph &nestedGraph =
          calleeParams.nestedScopes.find(&region)->second;
      bool inserted = topLevelGraph.nestedScopes
                          .try_emplace(&clonedRegion, nestedGraph.copy(map))
                          .second;
      assert(inserted);
    }
  });
  // Re-acquire `callScope` since the reference could have been invalidated
  // by the insertions into `calleeParams.nestedScopes`.
  callScope = scopeRegion == topLevelGraph.scope
                  ? &topLevelGraph
                  : &topLevelGraph.nestedScopes.find(scopeRegion)->second;
  // Decl scopes that were nested under the callee are now nested under the
  // current call scope.
  for (Region *nestedDecl : calleeParams.nestedDecls) {
    callScope->nestedDecls.push_back(
        &map.lookup(nestedDecl->getParentOp())
             ->getRegion(nestedDecl->getRegionNumber()));
  }

  // Do name mangling.
  if (needsMangling) {
    for (Operation *user : calleeParams.paramOps) {
      // Skip the parent decl. It's handled after.
      if (user == callee)
        continue;
      Operation *cloned = map.lookup(user);
      mangler.mangleElementsIn(cloned);
    }
  }
  for (auto &[name, def] : calleeParams.defs) {
    // Skip the parent decl. It's handled after.
    if (def.defOp == callee)
      continue;
    Operation *cloned = map.lookup(def.defOp);
    mangler.mangleElementsIn(cloned);
    // Rename declarations.
    auto itf = cast<ParamOpInterface>(def.defOp);
    SmallVector<ParamDeclAttr> newDecls;
    itf.walkDeclarations([&](ParamDeclAttr decl) {
      newDecls.push_back(mangler.mangleDecl(decl, needsMangling));
    });
    cast<ParamOpInterface>(cloned).renameDeclarations(newDecls);
    // Populate the new declarations into the call scope graph.
    propagateNewDecls(newDecls, topLevelGraph, *callScope, cloned, scopeRegion);
  }
  if (needsMangling) {
    for (Region *nestedScope : calleeParams.nestedDecls) {
      mangler.recursivelyMangle(
          &map.lookup(nestedScope->getParentOp())
               ->getRegion(nestedScope->getRegionNumber()),
          topLevelGraph);
    }
  }

  // Mangle the DeclInterface declarations.
  b.setInsertionPointToStart(&scope.getBody().front());
  for (auto [origDecl, value] :
       llvm::zip(callee.getInputParams(), call.getParamValues())) {
    ParamDeclAttr decl = mangler.mangleDecl(origDecl, needsMangling);
    auto declOp = b.create<ParamDeclareOp>(
        callee.getLoc(), decl,
        ParamOperatorAttr::get(b.getContext(), POC::Rebind, value,
                               decl.getType()));
    // Register the new declaration.
    propagateNewDecls(decl, topLevelGraph, *callScope, declOp, scopeRegion);
  }

  bool stripDebugInfo =
      callee.getAlwaysInlineLevel() == AlwaysInlineLevel::EnabledNoDebug;
  scope.getBody().walk([&](Operation *op) {
    // If this is an `always_inline(nodebug)`, erase the location of the
    // inlined operations by replacing them with the location of the call.
    // Otherwise, propagate the inlined location via a `CallSiteLoc`.
    if (stripDebugInfo)
      op->setLoc(call.getLoc());
    else
      op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), call.getLoc()));

    // Erase `debuginfo.value` operations when inlining without debug info.
    if (stripDebugInfo) {
      if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
        value.erase();
        return;
      }
    }
  });

  // Handle all terminators.
  unsigned numReturns = 0;
  callee.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk over nested functions. Control-flow does not cross them.
    if (op != callee && isa<FuncInterface>(op))
      return WalkResult::skip();
    if (!isa<ReturnOp, ParamResultBindOp>(op))
      return WalkResult::advance();

    Operation *cloned = map.lookup(op);
    b.setInsertionPoint(cloned);
    if (auto bind = dyn_cast<ParamResultBindOp>(cloned)) {
      for (auto [decl, value] :
           llvm::zip(call.getParamDecls(), bind.getParameters())) {
        auto rebound = ParamOperatorAttr::get(b.getContext(), POC::Rebind,
                                              value, decl.getType());
        b.create<ParamDeclareOp>(bind.getLoc(), decl, rebound);
      }
    } else {
      ++numReturns;
      b.create<HLCF::BreakOp>(cloned->getLoc(),
                              rebindReturnOperands(b, cloned, call), label);
    }
    cloned->erase();
    return WalkResult::advance();
  });
  b.replaceOp(call, scope.getResults());

  // If the scope was trivial (one return), fold it away.
  assert(numReturns > 0);
  if (numReturns == 1) {
    for (Operation &op : llvm::make_early_inc_range(
             scope.getBody().front().without_terminator()))
      op.moveBefore(scope);
    b.replaceOp(scope, scope.getBody().front().getTerminator()->getOperands());
  }
}

//===----------------------------------------------------------------------===//
// InlineGraph
//===----------------------------------------------------------------------===//

namespace {
/// A node in the inlining graph contains a function, edges to its callers, and
/// edges to its callees. A node is ready to inline its callees when all of
/// its callees have been processed.
struct InliningGraphNode {
  /// Create the node for the given function.
  explicit InliningGraphNode(GeneratorOp func)
      : func(func), calleeParamGraph(func.getBodyRegion()) {}

  /// This class is only move-constructed when the node map in
  /// `InliningGraphNode` is resized. That occurs before any references are
  /// taken to instances of this object, so just default-construct all other
  /// members of this class.
  InliningGraphNode(InliningGraphNode &&other)
      : func(other.func), calleeParamGraph(func.getBodyRegion()) {}

  /// Compute the caller parameter graph and declarations.
  void calculateParams(ParameterCollector::Analysis &paramCache);

  /// The function represented by the node.
  GeneratorOp func;

  /// Nodes of functions that inline call this function. These are the child
  /// edges.
  std::vector<InliningGraphNode *> callers;
  /// Calls and callees to inline inside this function. These are the parent
  /// edges.
  std::vector<std::pair<CallOp, InliningGraphNode *>> callsites;
  /// This mutex guards `callsites` and `callers` during parallel graph
  /// construction.
  llvm::sys::SmartRWMutex<true> mutex;

  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;

  /// In parametric inlining, each function has its parameter use-def graph
  /// computed twice: once as a caller, computed when the node is being
  /// processed, and once as a callee, when the fully processed node is called
  /// from somewhere else. Stash the callee graph on the node itself.
  ParameterUseDefGraph calleeParamGraph;
  /// A set of all declarations, regardless of type, in the callee.
  llvm::SetVector<StringAttr> allDecls;
};

void InliningGraphNode::calculateParams(
    ParameterCollector::Analysis &paramCache) {
  calleeParamGraph.calculate(paramCache);
  func.walk([&](Operation *op) {
    if (auto decl = dyn_cast<DeclInterface>(op)) {
      for (ParamDeclAttr decl : llvm::concat<const ParamDeclAttr>(
               decl.getInputParams(), decl.getResultParams()))
        allDecls.insert(decl.getName());
    }
    if (auto paramOp = dyn_cast<ParamOpInterface>(op)) {
      paramOp.walkDeclarations(
          [&](ParamDeclAttr decl) { allDecls.insert(decl.getName()); });
    }
  });
}

/// An inlining graph is a call graph between functions of concrete calls to
/// functions that must be inlined. The root nodes of the graph are
/// `always_inline` functions with no calls to other such functions, and the
/// leaf nodes are non-inlined functions.
///
/// This data structure is used to inline functions starting from the leaves of
/// callgraphs. This is more efficient because inlining from the roots of the
/// callgraph leads to duplicate work (splats callgraph into a tree). It also
/// enables inlined functions to be optimized and pruned as they are processed.
struct InliningGraph {
  /// Create an inlining graph with the specified inlining level.
  explicit InliningGraph(AlwaysInlineLevel level, LLCL::Runtime &runtime,
                         ParameterCollector::Analysis &paramCache)
      : level(level), runtime(runtime), paramCache(paramCache) {}

  /// Build the inlining graph for a module.
  void build(ModuleOp module, const SymbolTable &symtab);

  /// Process the graph by performing all requested inlining from the root
  /// nodes.
  void process();

  // Complete processing of a node by incrementing the number of processed calls
  // of all its callers. Note that the same function can appear in the caller
  // list N, indicating that it calls this function N times. This loop will
  // increment the `numProcessedCalls` counters N times as appropriate.
  void complete(InliningGraphNode *node);

  /// Inline the requested call.
  void inlineCall(ParameterUseDefGraph &callerParams, CallOp call,
                  InliningGraphNode *callee);

  /// Get the parameter cache copy belonging to the thread.
  ParameterCollector::Analysis &getThreadLocalCache(uint64_t threadId);

  /// The nodes in the graph. The map does not resize after it is constructed,
  /// so references always remain valid.
  DenseMap<GeneratorOp, InliningGraphNode> nodes;
  /// Calls to functions with at least this inline level are considered edges in
  /// the inlining graph.
  AlwaysInlineLevel level;
  /// The runtime to use.
  LLCL::Runtime &runtime;

  /// A parameter collector cache to use.
  ParameterCollector::Analysis &paramCache;
  /// The parameter mangler cache.
  DenseSet<const void *> manglerCache;
  /// Thread-local copies of the parameter cache.
  DenseMap<uint64_t, ParameterCollector::Analysis> threadCaches;
  /// This mutex guards the map of thread-local caches.
  llvm::sys::SmartRWMutex<true> cacheMutex;

  /// This chain is set when all in-flight work items are processed.
  LLCL::AsyncValueRef<LLCL::Chain> done;
  /// This is the number of in-flight work items.
  std::atomic<size_t> numWorkItems = 1;
};
} // namespace

void InliningGraph::build(ModuleOp module, const SymbolTable &symtab) {
  // Instantiate the nodes for each generator first.
  for (auto func : module.getOps<GeneratorOp>())
    nodes.try_emplace(func, InliningGraphNode(func));

  // Build the graph by walking all the calls in each function and adding edges
  // as appropriate.
  auto workFn = [this,
                 &symtab](std::pair<GeneratorOp, InliningGraphNode> &value) {
    auto &[func, node] = value;
    InliningGraphNode *callerNode = &node;
    func.walk([&](CallOp call) {
      auto callee = symtab.lookup<GeneratorOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
      // Filter calls that do not satisfy the inlining level.
      if (callee.getAlwaysInlineLevel() < level)
        return;
      InliningGraphNode *calleeNode = &nodes.find(callee)->second;
      {
        llvm::sys::SmartScopedWriter<true> lock(callerNode->mutex);
        callerNode->callsites.emplace_back(call, calleeNode);
      }
      {
        llvm::sys::SmartScopedWriter<true> lock(calleeNode->mutex);
        calleeNode->callers.push_back(callerNode);
      }
    });
  };
  mlir::parallelForEach(module.getContext(), nodes, workFn);
}

ParameterCollector::Analysis &
InliningGraph::getThreadLocalCache(uint64_t threadId) {
  ParameterCollector::Analysis *cache = nullptr;
  {
    llvm::sys::SmartScopedReader<true> lock(cacheMutex);
    auto it = threadCaches.find(threadId);
    if (it != threadCaches.end())
      cache = &it->second;
  }
  if (!cache) {
    llvm::sys::SmartScopedWriter<true> lock(cacheMutex);
    // Each thread gets a copy of the saved cache.
    cache = &threadCaches.try_emplace(threadId, paramCache).first->second;
  }
  return *cache;
}

void InliningGraph::complete(InliningGraphNode *node) {
  // Since the function is complete, compute its callee graph, if it has
  // any callers.
  if (node->callers.empty())
    return;
  node->calculateParams(getThreadLocalCache(llvm::get_threadid()));

  // Indicate it as complete to its callers by incrementing the ready counter on
  // the caller nodes. Schedule any ready callers.
  for (InliningGraphNode *caller : node->callers) {
    if (caller->numProcessedCalls.fetch_add(1) + 1 != caller->callsites.size())
      continue;
    // This caller is ready. Increment the number of active work items.
    numWorkItems.fetch_add(1);
    runtime.getWorkQueue()->addTask([caller, this] {
      // Compute the parameter use-def graph of the function as a caller.
      ParameterUseDefGraph callerParams(caller->func.getBodyRegion());
      callerParams.calculate(getThreadLocalCache(llvm::get_threadid()));
      // Inline all callees.
      for (auto [call, callee] : caller->callsites)
        inlineCall(callerParams, call, callee);
      complete(caller);
      // Complete this task. Check if all tasks are done.
      if (numWorkItems.fetch_sub(1) == 1)
        done.copy().emplace();
    });
  }
}

void InliningGraph::process() {
  // Reserve the thread-local cache map so that it never resizes.
  threadCaches.reserve(runtime.getWorkQueue()->getParallelismLevel());

  // Allocate the completion chain.
  done = LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime);

  // Populate the worklist with root nodes.
  for (auto &[func, node] : nodes) {
    // Root nodes are already complete.
    if (!node.callsites.empty())
      continue;
    InliningGraphNode *caller = &node;
    // Increment the number of in-flight tasks.
    numWorkItems.fetch_add(1);
    runtime.getWorkQueue()->addTask([caller, this] {
      complete(caller);
      // Check if all tasks are done.
      if (numWorkItems.fetch_sub(1) == 1)
        done.copy().emplace();
    });
  }
  // Check if all tasks are done.
  if (numWorkItems.fetch_sub(1) == 1)
    done.copy().emplace();

  // Wait on all active work items.
  LLCL::await(done);
}

void InliningGraph::inlineCall(ParameterUseDefGraph &callerParams, CallOp call,
                               InliningGraphNode *callee) {
  inlineGeneratorCall(call, callee->func, callerParams,
                      callee->calleeParamGraph, callee->allDecls, manglerCache);
}

//===----------------------------------------------------------------------===//
// AlwaysInlineParametricPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ALWAYSINLINEPARAMETRIC
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct AlwaysInlineParametricPass
    : impl::AlwaysInlineParametricBase<AlwaysInlineParametricPass> {
  explicit AlwaysInlineParametricPass(LLCL::Runtime *runtime = nullptr)
      : runtime(runtime) {}

  void runOnOperation() override;

  LLCL::Runtime *runtime;
};
} // namespace

void AlwaysInlineParametricPass::runOnOperation() {
  // Create a runtime instance if needed.
  auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
      runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  InliningGraph graph(AlwaysInlineLevel::Enabled, *rt, paramCache);
  graph.build(getOperation(), symtab);
  graph.process();
}

std::unique_ptr<mlir::Pass>
KGEN::createAlwaysInlineParametric(LLCL::Runtime &runtime) {
  return std::make_unique<AlwaysInlineParametricPass>(&runtime);
}

//===----------------------------------------------------------------------===//
// inlineFunctionCall
//===----------------------------------------------------------------------===//

static StringAttr getCalleeSymbol(KGENCallOpInterface call) {
  return cast<FlatSymbolRefAttr>(
             cast<SymbolConstantAttr>(call.getCallee()).getSymbol())
      .getAttr();
}

/// Replace the call operation with the given region using values from args for
/// the region inputs.
///
/// The region is inserted into its own scope - either a loop or async execute
/// op (depending on the type of the call). This scope is returned from the
/// function.
static std::pair<Operation *, int> inlineRegion(KGENCallOpInterface call,
                                                Region &region,
                                                ArrayRef<BlockArgument> args) {
  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  mlir::IRRewriter b{OpBuilder(call)};
  Operation *scope;
  if (isa<CallOp>(call.getOperation())) {
    scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                   ValueRange(), label);
  } else {
    auto asyncCall = cast<LIT::AsyncCallOp>(call.getOperation());
    scope = b.create<LIT::AsyncExecuteOp>(call.getLoc(), asyncCall.getType());
  }
  b.createBlock(&scope->getRegions().front());

  IRMapping map;
  for (auto [value, arg] : llvm::zip(call->getOperands(), args))
    map.map(arg, value);
  for (Operation &op : region.getOps())
    b.clone(op, map);
  unsigned numReturns = 0;
  scope->walk([&](Operation *op) {
    // Replace all returns with breaks to the control flow scope.
    if (!isa<ReturnOp>(op))
      return;
    b.setInsertionPoint(op);
    if (isa<CallOp>(call.getOperation()))
      b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
    else
      b.replaceOpWithNewOp<LIT::AsyncReturnOp>(op, op->getOperands());

    ++numReturns;
  });
  b.replaceOp(call, scope->getResults());
  assert(numReturns > 0);
  return std::make_pair(scope, numReturns);
}

/// Replace the call operation with the body of the callee function.
///
/// The function body is inserted into its own scope - either a loop or async
/// execute op (depending on the type of the call). This scope is returned from
/// the function.
static std::pair<Operation *, int> inlineFunctionBody(KGENCallOpInterface call,
                                                      FuncOp callee) {
  return inlineRegion(call, callee.getBodyRegion(), callee.getArguments());
}

/// Inlining might create trivial loops with a single break at the end. This
/// function cleans it up.
static void foldTrivialLoop(Operation *op) {
  auto loop = dyn_cast<HLCF::LoopOp>(op);
  if (!loop)
    return;

  mlir::IRRewriter b{OpBuilder(op)};

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();
  b.inlineBlockBefore(&body, loop);
  b.replaceOp(loop, term->getOperands());
  b.eraseOp(term);
}

/// Iteratively inline calls into the given function.
///
/// The decision whether to inline a specific call is performed by
/// 'shouldInline' function, which takes two arguments: a function and a call
/// considered to be inlined into it.
/// The function body is inserted into a special scope which is passed to the
/// 'postprocess' callback. The callback takes three arguments: the function
/// being inlined, the scope containing its body, and location of the original
/// call operation.
static LogicalResult iterativelyInlineFunctionCalls(
    FuncOp func, const SymbolTable &symtab,
    function_ref<bool(FuncOp, KGENCallOpInterface)> shouldInline,
    function_ref<void(FuncOp, Operation *, Location loc)> postprocess) {
  struct EndStack {};

  // Collect all calls that inline in this function.
  SmallVector<SmartVariant<Operation *, EndStack>> calls;
  func.walk([&](KGENCallOpInterface call) {
    if (shouldInline(func, call))
      calls.emplace_back(call);
  });

  // Keep a callstack for a nice error when cycles are detected.
  SmallVector<Location, 16> callstack;
  llvm::SetVector<Operation *, SmallVector<Operation *, 16>,
                  SmallPtrSet<Operation *, 16>>
      seenFuncs;
  while (!calls.empty()) {
    SmartVariant<Operation *, EndStack> next = calls.pop_back_val();
    if (isa<EndStack>(next)) {
      callstack.pop_back();
      seenFuncs.pop_back();
      continue;
    }
    auto call = cast<KGENCallOpInterface>(cast<Operation *>(next));
    if (!isa<CallOp>(call) && !isa<LIT::AsyncCallOp>(call))
      continue;
    auto callee = symtab.lookup<FuncOp>(getCalleeSymbol(call));

    // If we recursed onto the same function, give up and emit an error.
    if (!seenFuncs.insert(callee)) {
      InFlightDiagnostic diag = mlir::emitError(
          func.getLoc(),
          "function has recursive call to 'always_inline' function");
      assert(callstack.size() == seenFuncs.size());
      for (auto [callLoc, func] : llvm::zip(callstack, seenFuncs)) {
        diag.attachNote(callLoc) << "through call here";
        diag.attachNote(func->getLoc())
            << "to function marked 'always_inline' here";
      }
      diag.attachNote(call.getLoc()) << "function call here recurses";
      diag.attachNote(callee.getLoc()) << "back to function here";
      return failure();
    }
    callstack.push_back(call.getLoc());
    calls.emplace_back(EndStack{});
    auto loc = call.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "Inlining\n    " << callee.getSymName()
                            << "  into\n    " << func.getSymName() << "\n");
    auto [scope, numReturns] = inlineFunctionBody(call, callee);

    postprocess(callee, scope, loc);

    // Scan the inlined body for calls that we might want to also inline
    scope->walk([&](Operation *op) {
      // Check for a call to recursively inline.
      if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
        if (shouldInline(func, call))
          calls.emplace_back(call);
      }
    });
    // If the loop scope was trivial (one return), fold it away.
    if (numReturns == 1)
      foldTrivialLoop(scope);
  }
  assert(callstack.empty() && seenFuncs.empty());
  return success();
}

//===----------------------------------------------------------------------===//
// ForceInlinePass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_FORCEINLINE
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ForceInlinePass : impl::ForceInlineBase<ForceInlinePass> {
  using ForceInlineBase::ForceInlineBase;

  void runOnOperation() override;
};
} // namespace

void ForceInlinePass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "==== ForceInline Pass ====\n");
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  std::vector<FuncOp> rootFuncs;
  for (auto func : getOperation().getOps<FuncOp>()) {
    // Skip over functions that are force inlined. Start inlining from the
    // tips.
    if (func.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
      continue;
    rootFuncs.push_back(func);
  }
  auto shouldInline = [&](FuncOp func, KGENCallOpInterface call) {
    auto callee = symtab.lookup<FuncOp>(getCalleeSymbol(call));
    if (callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
      return true;

    return false;
  };
  auto updateDebugInfo = [&](FuncOp func, Operation *scope, Location callLoc) {
    AlwaysInlineLevel level = func.getAlwaysInlineLevel();

    scope->walk([&](Operation *op) {
      if (op != scope) {
        // If this is an `always_inline(nodebug)`, erase the location of the
        // inlined operations by replacing them with the location of the call.
        // Otherwise, propagate the inlined location via a `CallSiteLoc`.
        if (level == AlwaysInlineLevel::EnabledNoDebug)
          op->setLoc(callLoc);
        else
          op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), callLoc));
      }
      // Erase `debuginfo.value` operations when inlining without debug info.
      if (level == AlwaysInlineLevel::EnabledNoDebug) {
        if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
          value.erase();
          return;
        }
      }
    });
  };
  if (failed(mlir::failableParallelForEach(
          &getContext(), rootFuncs, [&](FuncOp func) {
            return iterativelyInlineFunctionCalls(func, symtab, shouldInline,
                                                  updateDebugInfo);
          })))
    return signalPassFailure();
}
