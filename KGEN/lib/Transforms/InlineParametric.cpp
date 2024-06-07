//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "KGEN/TransformUtils/InliningUtils.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "LLCL/Support/ForkJoin.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/STLExtras.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

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
  using Cache = DenseSet<const void *>;

  explicit AttrTypeMangler(Cache &manglerCache) : manglerCache(manglerCache) {}

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
                const llvm::SetVector<StringAttr> &calleeDecls,
                const ParameterUseDefGraph &topLevelGraph);

  /// Optionally mangle a declaration.
  ParamDeclAttr mangleDecl(ParamDeclAttr decl, bool needsMangling);

  /// Mangle attributes and types.
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
  Cache &manglerCache;
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

/// This uniquing scheme involves splitting each decl name into a key string
/// and a substring of trailing digits. We track the max of such digits of the
/// same key string and use that to generate the next unique ID.
class NameUniquer {
public:
  NameUniquer(const ParameterUseDefGraph &scope,
              const ParameterUseDefGraph &topLevelGraph)
      : topLevelGraph(topLevelGraph) {
    updateMaxIds(scope);
  }

  /// Check if the name needs mangling.
  bool needsMangling(StringAttr name) {
    auto [key, id] = split(name);
    if (auto it = maxIds.find(key); it != maxIds.end())
      return id <= it->second;
    return false;
  }

  /// Uniquely mangle a parameter name. Returns the original name if mangling is
  /// not needed.
  StringAttr mangle(StringAttr name) {
    if (!needsMangling(name))
      return name;
    auto [key, _] = split(name);
    ssize_t newId = ++maxIds[key];
    return StringAttr::get(name.getContext(), key + Twine(newId));
  }

  /// Update the uniquer with a new name.
  void updateWith(StringRef name) {
    auto [key, id] = split(name);
    ssize_t &max = maxIds.try_emplace(key, -1).first->second;
    max = std::max(max, id);
  }

private:
  /// Split the name into the base name and a trailing id. If there is not
  /// trailing number, -1 is returned.
  static std::pair<StringRef, ssize_t> split(StringRef name) {
    // We first
    StringRef key = name.rtrim("0123456789");
    size_t splitIdx = key.size();

    // -1 means no number suffix.
    ssize_t id = -1;
    name.substr(splitIdx).getAsInteger(/*Radix=*/10, id);

    return std::make_pair(key, id);
  };

  /// Update the ids we are tracking with the declarations (including those
  /// nested) in the given scope.
  void updateMaxIds(const ParameterUseDefGraph &scope) {
    for (auto [declName, _] : scope.decls)
      updateWith(declName);
    for (Region *nestedRegion : scope.nestedDecls)
      updateMaxIds(topLevelGraph.nestedScopes.at(nestedRegion));
  }

  /// Map to store the maximum id for each base name we are tracking.
  llvm::StringMap<ssize_t> maxIds;

  /// The top level ParameterUseDefGraph that contains nested scopes that that
  /// carry declarations.
  const ParameterUseDefGraph &topLevelGraph;
};

bool AttrTypeMangler::populate(Builder &b, const ParameterUseDefGraph &curScope,
                               const llvm::SetVector<StringAttr> &calleeDecls,
                               const ParameterUseDefGraph &topLevelGraph) {
  CompilerTimeTraceScope traceScope("AttrTypeMangler::populate");

  // `curScope` contains all declarations visible in the scope of the call,
  // including those defined in higher scopes. When the function is inlined,
  // these are the declarations that will project into the inlined body. We need
  // to mangle parameters in the inlined body such that they do not collide with
  // any declarations visible in the call scope, or in any nested scopes.
  NameUniquer uniquer(curScope, topLevelGraph);
  bool needsMangling = false;
  for (StringAttr decl : calleeDecls) {
    if (!uniquer.needsMangling(decl))
      continue;
    if (!needsMangling) {
      // Lazily populate with the callee decls
      for (StringAttr name : calleeDecls)
        uniquer.updateWith(name);
    }
    auto mangled = uniquer.mangle(decl);
    mangledDecls.try_emplace(decl, mangled);
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
  op->setAttrs(cast<DictionaryAttr>(mangleRefsIn(op->getAttrDictionary())));

  for (OpResult result : op->getResults())
    result.setType(mangleRefsIn(result.getType()));

  for (Region &region : op->getRegions())
    for (BlockArgument arg : region.front().getArguments())
      arg.setType(mangleRefsIn(arg.getType()));
}

void AttrTypeMangler::recursivelyMangle(Region *scope,
                                        const ParameterUseDefGraph &graph) {
  VerboseCompilerTimeTraceScope traceScope(
      "AttrTypeMangler::recursivelyMangle");

  const ParameterUseDefGraph &uses = graph.nestedScopes.find(scope)->second;

  for (Operation *op : uses.paramOps)
    if (op != scope->getParentOp())
      mangleElementsIn(op);
  for (auto &[_, decl] : uses.decls)
    if (scope->getParentOp()->isProperAncestor(decl.declOp))
      mangleElementsIn(decl.declOp);

  for (Region *nestedScope : uses.nestedDecls)
    recursivelyMangle(nestedScope, graph);
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
  newValues.reserve(inputs.size());
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

using MangleDefTy = function_ref<void(const ParamDefinition &, Region *,
                                      ParameterUseDefGraph *, IRMapping &)>;

/// Recursively mangle parameter definitions within the inlined scope
/// corresponding to the callee's use def graph. The mangling callback is
/// more or less arbitrary.
static void recursivelyMangleDefs(IRMapping &map, Region *calleeRegion,
                                  const ParameterUseDefGraph &calleeGraph,
                                  ParameterUseDefGraph &topLevelGraph,
                                  MangleDefTy mangleDef) {
  const ParameterUseDefGraph *calleeNestedGraph =
      &calleeGraph.nestedScopes.find(calleeRegion)->second;
  Region *clonedNestedRegion =
      &map.lookup(calleeRegion->getParentOp())
           ->getRegion(calleeRegion->getRegionNumber());
  for (auto &[_, def] : calleeNestedGraph->defs) {
    mangleDef(def, clonedNestedRegion,
              &topLevelGraph.nestedScopes.find(clonedNestedRegion)->second,
              map);
  }
  for (Region *calleeNestedRegion : calleeNestedGraph->nestedDecls) {
    recursivelyMangleDefs(map, calleeNestedRegion, calleeGraph, topLevelGraph,
                          mangleDef);
  }
}

static void inlineGeneratorCall(GeneratorOp caller, CallOp call,
                                GeneratorOp callee, InlineLevel level,
                                ParameterUseDefGraph &topLevelGraph,
                                const ParameterUseDefGraph &calleeParams,
                                const llvm::SetVector<StringAttr> &calleeDecls,
                                AttrTypeMangler::Cache &manglerCache,
                                bool updateDebugInfo, bool debugCallsite) {
  CompilerTimeTraceScope traceScope("inlineGeneratorCall",
                                    [&] { return callee.getSymName().str(); });

  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  // Get the parameters in-scope at the callsite.
  Region *scopeRegion = getNearestDeclRegion(call);
  ParameterUseDefGraph *callScope =
      scopeRegion == topLevelGraph.scope
          ? &topLevelGraph
          : &topLevelGraph.nestedScopes.find(scopeRegion)->second;

  mlir::IRRewriter b{OpBuilder(call)};
  AttrTypeMangler mangler(manglerCache);
  bool needsMangling =
      mangler.populate(b, *callScope, calleeDecls, topLevelGraph);

  // Make sure to rebind the call operands based on the mangled types of the
  // callee's argument types.
  SmallVector<Type> argTypes = llvm::to_vector(callee.getArgumentTypes());
  if (needsMangling)
    for (Type &type : argTypes)
      type = mangler.mangleRefsIn(type);

  b.setInsertionPointAfter(call);
  if (debugCallsite && callee.getLocScope())
    b.create<DebugInfo::LineTableLocOp>(call->getLoc());

  SmallVector<Value> argVals =
      rebindValues(b, call.getLoc(), call->getOperands(), argTypes);

  // Use a LoopOp to be able to break to a label - any returns inlined from
  // callee must only exit the inlined block.
  auto scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                      ValueRange(), label);
  b.createBlock(&scope.getBody());

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

  /// Since the callee might contain nested parameter scopes (e.g.
  /// `kgen.param.if`), we recursively walk them and mangle parameter
  /// definitions.
  auto mangleDef = [&mangler, &needsMangling, &topLevelGraph](
                       const ParamDefinition &def, Region *scopeRegion,
                       ParameterUseDefGraph *defScope, IRMapping &map) {
    Operation *cloned = map.lookup(def.defOp);
    mangler.mangleElementsIn(cloned);
    // Rename declarations.
    auto itf = cast<ParamOpInterface>(def.defOp);
    SmallVector<ParamDeclAttr> newDecls;
    itf.walkDeclarations([&](ParamDeclAttr decl) {
      newDecls.push_back(mangler.mangleDecl(decl, needsMangling));
    });
    cast<ParamOpInterface>(cloned).renameDeclarations(newDecls);

    // At this point, the only nested ops that declares parameters in their
    // scope are ParamDeclareRegionOp and ParamForOp, whose declarations need
    // special treatment.
    if (needsMangling) {
      if (auto regionDecl = dyn_cast<ParamDeclareRegionOp>(cloned)) {
        SmallVector<ParamDeclAttr> newInputDecls;
        for (ParamDeclAttr decl : regionDecl.getInputParams())
          newInputDecls.emplace_back(mangler.mangleDecl(decl, needsMangling));
        regionDecl.setInputParams(newInputDecls);
        newDecls.append(newInputDecls);

        SmallVector<ParamDeclAttr> newResDecls;
        for (ParamDeclAttr decl : regionDecl.getResultParams())
          newResDecls.emplace_back(mangler.mangleDecl(decl, needsMangling));
        regionDecl.setResultParams(newResDecls);
        newDecls.append(newResDecls);
      } else if (auto paramFor = dyn_cast<ParamForOp>(cloned)) {
        ParamDeclAttr newDecl =
            mangler.mangleDecl(paramFor.getParamDecl(), needsMangling);
        paramFor.setParamDeclAttr(newDecl);
        newDecls.push_back(newDecl);
      }
    }

    // Populate the new declarations into the call scope graph.
    propagateNewDecls(newDecls, topLevelGraph, *defScope, cloned, scopeRegion);
  };
  for (auto &[_, def] : calleeParams.defs) {
    // Skip the parent decl. It's handled after.
    if (def.defOp == callee)
      continue;
    mangleDef(def, scopeRegion, callScope, map);
  }
  for (Region *calleeNestedRegion : calleeParams.nestedDecls) {
    recursivelyMangleDefs(map, calleeNestedRegion, calleeParams, topLevelGraph,
                          mangleDef);
  }

  if (needsMangling) {
    for (Region *nestedScope : calleeParams.nestedDecls) {
      Operation *clonedOp = map.lookup(nestedScope->getParentOp());
      Region &clonedRegion =
          clonedOp->getRegion(nestedScope->getRegionNumber());
      mangler.recursivelyMangle(&clonedRegion, topLevelGraph);
    }
  }

  // Mangle the DeclInterface declarations.
  // TODO: mangle result parameter names as well.
  b.setInsertionPoint(call);
  for (auto [origDecl, value] :
       llvm::zip(callee.getInputParams(), call.getParamValues())) {
    ParamDeclAttr decl = mangler.mangleDecl(origDecl, needsMangling);
    auto declOp = b.create<ParamDeclareOp>(
        call.getLoc(), decl,
        ParamOperatorAttr::get(b.getContext(), POC::Rebind, value,
                               decl.getType()));
    // Register the new declaration.
    propagateNewDecls(decl, topLevelGraph, *callScope, declOp, scopeRegion);
  }

  // When building in debuginfo, mangle parameters in all op locations.
  if (updateDebugInfo) {
    scope.getBody().walk([&](Operation *op) {
      op->setLoc(cast<LocationAttr>(mangler.mangleRefsIn(op->getLoc())));
    });
  }

  // Handle all terminators.
  unsigned numReturns = 0;
  callee.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SourceLocOp>(op)) {
      processSourceLocOp(cast<SourceLocOp>(map.lookup(op)), call.getLoc(), b);
      return WalkResult::advance();
    }

    // Walk over nested functions. Control-flow does not cross them.
    if (isa<FuncInterface>(op))
      return WalkResult::skip();
    if (!isa<ReturnOp>(op))
      return WalkResult::advance();

    Operation *cloned = map.lookup(op);
    b.setInsertionPoint(cloned);
    ++numReturns;
    b.create<HLCF::BreakOp>(cloned->getLoc(),
                            rebindReturnOperands(b, cloned, call), label);
    cloned->erase();
    return WalkResult::advance();
  });
  b.replaceOp(call, scope.getResults());

  std::optional<StringAttr> updateAttrName;
  if (updateDebugInfo)
    updateAttrName = StringAttr();
  bool singleExit =
      numReturns == 1 && isa<ReturnOp>(callee.getBody()->getTerminator());
  maybeUpdateDebugInfo(scope, updateAttrName, singleExit);
}

//===----------------------------------------------------------------------===//
// InlineGraph
//===----------------------------------------------------------------------===//

namespace {
/// An inlining graph is a call graph between functions of concrete calls to
/// functions that must be inlined. The root nodes of the graph are
/// `always_inline` functions with no calls to other such functions, and the
/// leaf nodes are non-inlined functions.
///
/// This data structure is used to inline functions starting from the leaves of
/// callgraphs. This is more efficient because inlining from the roots of the
/// callgraph leads to duplicate work (splats callgraph into a tree). It also
/// enables inlined functions to be optimized and pruned as they are processed.
///
/// This structure is implemented as a CRTP class so that the core algorithm can
/// be shared between both inliners.
template <typename DerivedT, typename NodeT>
struct InliningGraphBase : public CallGraphBase<DerivedT, NodeT> {
  explicit InliningGraphBase(LLCL::Runtime &runtime)
      : runtime(runtime), state(runtime) {}

  using CallGraphBase<DerivedT, NodeT>::getDerived;

  /// Process the graph by performing all requested inlining from the root
  /// nodes.
  void process();

  // Complete processing of a node by incrementing the number of processed calls
  // of all its callers. Note that the same function can appear in the caller
  // list N, indicating that it calls this function N times. This loop will
  // increment the `numProcessedCalls` counters N times as appropriate.
  void complete(NodeT *node);

  /// The runtime to use.
  LLCL::Runtime &runtime;

  /// The inlining task state.
  LLCL::ForkJoin state;
  /// The number of nodes that complete processing. If this is not equal to the
  /// number of nodes, then there are cycles in the graph.
  std::atomic<size_t> numProcessed = 0;
};
} // namespace

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::complete(NodeT *node) {
  // Run the function pipeline after inlining has been performed for a function.
  // Make sure the verifier is off. Note that `Pass::runPipeline` is not thread
  // safe due to analysis manager nesting.
  getDerived().onComplete(node);

  // Since the function is complete, compute its callee graph, if it has
  // any callers.
  numProcessed.fetch_add(1);
  if (!getDerived().prepareForInlining(node))
    return;

  // Indicate it as complete to its callers by incrementing the ready counter on
  // the caller nodes. Schedule any ready callers.
  for (NodeT *caller : node->callers) {
    if (caller->numProcessedCalls.fetch_add(1) + 1 != caller->callsites.size())
      continue;
    // This caller is ready. Increment the number of active work items.
    state.fork([caller, this] {
      // Compute the parameter use-def graph of the function as a caller.
      // Inline all callees.
      getDerived().performInlining(caller);
      complete(caller);
    });
  }
}

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::process() {
  CompilerTimeTraceScope traceScope("InliningGraphBase::process");

  // Populate the worklist with root nodes.
  for (auto &[func, node] : this->nodes) {
    // Root nodes are already complete.
    if (!node.callsites.empty())
      continue;
    NodeT *caller = &node;
    // Increment the number of in-flight tasks.
    state.fork([caller, this] { complete(caller); });
  }
  // Wait on all active work items.
  state.join();
}

//===----------------------------------------------------------------------===//
// ParametricInliningGraph
//===----------------------------------------------------------------------===//

namespace {
struct ParametricInliningGraphNode
    : public CallGraphNodeBase<ParametricInliningGraphNode, GeneratorOp,
                               CallOp> {
  explicit ParametricInliningGraphNode(GeneratorOp func)
      : CallGraphNodeBase(func), level(func.getInlineLevel()),
        calleeParamGraph(func.getBodyRegion()) {}
  ParametricInliningGraphNode(ParametricInliningGraphNode &&other)
      : CallGraphNodeBase(other.func), level(other.level),
        calleeParamGraph(other.func.getBodyRegion()) {}

  /// Compute the caller parameter graph and declarations.
  void calculateParams(ParameterCollector::Analysis &paramCache);

  /// The inlining level of the function.
  InlineLevel level;
  /// In parametric inlining, each function has its parameter use-def graph
  /// computed twice: once as a caller, computed when the node is being
  /// processed, and once as a callee, when the fully processed node is called
  /// from somewhere else. Stash the callee graph on the node itself.
  ParameterUseDefGraph calleeParamGraph;
  /// A set of all declarations, regardless of type, in the callee.
  llvm::SetVector<StringAttr> allDecls;
  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;
};

struct ParametricInliningGraph
    : public InliningGraphBase<ParametricInliningGraph,
                               ParametricInliningGraphNode> {
  explicit ParametricInliningGraph(InlineLevel level, LLCL::Runtime &runtime,
                                   ParameterCollector::Analysis &paramCache,
                                   unsigned optimizationLevel,
                                   bool updateDebugInfo)
      : InliningGraphBase(runtime), level(level),
        paramCaches(paramCache, runtime.getWorkQueue()->getParallelismLevel()),
        manglerCaches(baseManglerCache,
                      runtime.getWorkQueue()->getParallelismLevel()),
        optimizationLevel(optimizationLevel), updateDebugInfo(updateDebugInfo) {
  }

  void onComplete(ParametricInliningGraphNode *node) {}

  /// CallGraphBase interface for whether to add the node to the graph.
  bool shouldAddToGraph(CallOp call, ParametricInliningGraphNode *node) {
    return shouldInline(node);
  }

  /// Only inline functions that satisfy the inlining level.
  bool shouldInline(ParametricInliningGraphNode *node) const {
    assert(node->level == node->func.getInlineLevel());

    bool shouldInlineAutomatically =
        node->level == InlineLevel::Always &&
        getNumOperations(node->func) < getInlineThreshold();

    bool shouldAlwaysInline =
        (node->level >= level && node->level != InlineLevel::Never);
    return shouldInlineAutomatically || shouldAlwaysInline;
  }

  /// When a function is finished processing and will be inlined, compute is
  /// callee parameter graph.
  bool prepareForInlining(ParametricInliningGraphNode *node);
  /// Inline all functions by invoking the parametric inliner.
  void performInlining(ParametricInliningGraphNode *caller);

  /// The inlining level.
  InlineLevel level;
  /// Base mangler cache instance. It is always empty.
  AttrTypeMangler::Cache baseManglerCache;
  /// Thread local parameter collector caches.
  ThreadLocalCache<ParameterCollector::Analysis> paramCaches;
  /// Thread local mangler caches.
  ThreadLocalCache<AttrTypeMangler::Cache> manglerCaches;

  /// Get inlining threshold based optimization level.
  uint64_t getInlineThreshold() const;

  /// Compiler optimization level.
  unsigned optimizationLevel;

  /// Whether DebugInfo should be updated or not.
  bool updateDebugInfo;
};
} // namespace

void ParametricInliningGraphNode::calculateParams(
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

bool ParametricInliningGraph::prepareForInlining(
    ParametricInliningGraphNode *node) {
  // Skip inlining of functions with no callers.
  if (node->callers.empty())
    return false;
  node->calculateParams(paramCaches.getThreadLocalCache());
  return true;
}

uint64_t ParametricInliningGraph::getInlineThreshold() const {
  // TODO: add better heuristics
  switch (optimizationLevel) {
  case 0:
    return 0;
  case 1:
    return 2;
  case 2:
    return 5;
  case 3:
    return 8;
  default:
    return 0;
  }
}

void ParametricInliningGraph::performInlining(
    ParametricInliningGraphNode *caller) {
  ParameterUseDefGraph callerParams(caller->func.getBodyRegion());
  callerParams.calculate(paramCaches.getThreadLocalCache());
  for (auto [call, callee] : caller->callsites) {
    inlineGeneratorCall(caller->func, call, callee->func, callee->level,
                        callerParams, callee->calleeParamGraph,
                        callee->allDecls, manglerCaches.getThreadLocalCache(),
                        updateDebugInfo, !optimizationLevel);
  }
}

//===----------------------------------------------------------------------===//
// InlineParametricPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_INLINEPARAMETRIC
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct InlineParametricPass : impl::InlineParametricBase<InlineParametricPass> {
  explicit InlineParametricPass(const InlineParametricOptions &options = {})
      : InlineParametricBase(options) {}
  void runOnOperation() override;
};
} // namespace

void InlineParametricPass::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  LLCL::Runtime &runtime = *loadContext(&getContext())->get<LLCL::Runtime>();
  ParametricInliningGraph graph(
      nodebugOnly ? InlineLevel::AlwaysNoDebug : InlineLevel::Always, runtime,
      paramCache, optimizationLevel, updateDebugInfo);
  graph.build(getOperation(), symtab);
  graph.process();

  // Do one quick pass to inline any call to a function that is ready to be
  // inlined, in case cycles prevent us from inlining trivial functions.
  //
  // Note: we choose not to iterate because most nodebug inline functions should
  // be trivial and not have calls to recursive functions.
  auto inlineReadyFn = [&graph, updateDebugInfo = updateDebugInfo.getValue(),
                        optimizationLevel = optimizationLevel.getValue()](
                           ParametricInliningGraphNode &caller) {
    // Skip nodes that are completely processed.
    if (caller.numProcessedCalls == caller.callsites.size())
      return;
    ParameterUseDefGraph callerParams(caller.func.getBodyRegion());
    callerParams.calculate(graph.paramCaches.getThreadLocalCache());
    for (auto [call, callee] : caller.callsites) {
      // Skip nodes that are not complete.
      if (callee->numProcessedCalls != callee->callsites.size())
        continue;
      inlineGeneratorCall(caller.func, call, callee->func, callee->level,
                          callerParams, callee->calleeParamGraph,
                          callee->allDecls,
                          graph.manglerCaches.getThreadLocalCache(),
                          updateDebugInfo, !optimizationLevel);
    }
  };

  // Note: use the same threadpool as before, because that's what the thread
  // local caches are initialized for.
  LLCL::ForkJoin state(runtime);
  for (ParametricInliningGraphNode &caller :
       llvm::make_second_range(graph.nodes))
    state.fork([&] { inlineReadyFn(caller); });
  state.join();
}
