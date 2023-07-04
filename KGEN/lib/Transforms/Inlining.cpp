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
#include "Support/Threading/ThreadLocalCache.h"
#include "Support/TimeProfiler.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
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
  TimeTraceScope<> traceScope("AttrTypeMangler::populate");

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

  const ParameterUseDefGraph &uses = graph.nestedScopes.find(scope)->second;

  for (Operation *op : uses.paramOps) {
    if (op == scope->getParentOp())
      continue;
    mangleElementsIn(op);
  }
  for (auto &[_, decl] : uses.decls) {
    if (!scope->getParentOp()->isProperAncestor(decl.declOp))
      continue;
    mangleElementsIn(decl.declOp);
  }

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

void KGEN::inlineGeneratorCall(CallOp call, GeneratorOp callee,
                               AlwaysInlineLevel level,
                               ParameterUseDefGraph &topLevelGraph,
                               const ParameterUseDefGraph &calleeParams,
                               const llvm::SetVector<StringAttr> &calleeDecls,
                               AttrTypeMangler::Cache &manglerCache) {
  TimeTraceScope<> traceScope("inlineGeneratorCall",
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
  auto scope = b.create<HLCF::LoopOp>(
      call.getLoc(), call->getResultTypes(), ValueRange(), label,
      HLCF::LoopUnrollFullAttr::get(call.getContext(),
                                    HLCF::LoopUnrollFull::None));
  b.createBlock(&scope.getBody());

  AttrTypeMangler mangler(manglerCache);
  bool needsMangling =
      mangler.populate(b, *callScope, calleeDecls, topLevelGraph);

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

    // At this point, the only nested op that declares parameters in its scope
    // is ParamDeclareRegionOp, whose declarations need special treatment.
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

  bool stripDebugInfo = level == AlwaysInlineLevel::EnabledNoDebug;
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
  callee.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk over nested functions. Control-flow does not cross them.
    if (isa<FuncInterface>(op))
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

  // If the scope was trivial (one return at the end), fold it away.
  if (numReturns == 1 && isa<ReturnOp>(callee.getBody()->getTerminator())) {
    Operation *term = scope.getBody().front().getTerminator();
    scope->getBlock()->getOperations().splice(
        scope->getIterator(), scope.getBody().front().getOperations());
    b.replaceOp(scope, term->getOperands());
    term->erase();
  }
}

//===----------------------------------------------------------------------===//
// InlineGraph
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains the parallelism state of the graph traversal.
struct ParallelState {
  explicit ParallelState(LLCL::Runtime &runtime)
      : done(LLCL::AsyncValueRef<LLCL::Chain>::allocate(runtime)) {}

  /// Start a new work item.
  void startWork() { numWorkItems.fetch_add(1); }
  /// End a work item. Emplace the chain if everything is done.
  void endWork() {
    if (numWorkItems.fetch_sub(1) == 1)
      done.copy().emplace();
  }
  /// Called by the main thread, this function waits for all work to complete.
  void await() {
    endWork();
    LLCL::await(done);
  }

  /// This chain is set when all in-flight work items are processed.
  LLCL::AsyncValueRef<LLCL::Chain> done;
  /// This is the number of in-flight work items.
  std::atomic<size_t> numWorkItems = 1;
};

/// A node in the inlining graph contains a function, edges to its callers, and
/// edges to its callees. A node is ready to inline its callees when all of
/// its callees have been processed.
template <typename DerivedT, typename FuncT, typename CallT>
struct InliningGraphNodeBase {
  using FuncOpT = FuncT;
  using CallOpT = CallT;

  /// Create the node for the given function.
  explicit InliningGraphNodeBase(FuncT func) : func(func) {}

  /// This class is only move-constructed when the node map in
  /// `InliningGraphBase` is resized. That occurs before any references are
  /// taken to instances of this object, so just default-construct all other
  /// members of this class.
  InliningGraphNodeBase(InliningGraphNodeBase &&other) : func(other.func) {}

  /// The function represented by the node.
  FuncOpT func;

  /// Nodes of functions that inline call this function. These are the child
  /// edges.
  std::vector<DerivedT *> callers;
  /// Calls and callees to inline inside this function. These are the parent
  /// edges.
  std::vector<std::pair<CallOpT, DerivedT *>> callsites;
  /// This mutex guards `callsites` and `callers` during parallel graph
  /// construction.
  llvm::sys::SmartRWMutex<true> mutex;

  /// The number of processed calls. When the value of this counter equals the
  /// size of `callsites`, then all calls for this function have been processed.
  std::atomic<size_t> numProcessedCalls = 0;
};

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
struct InliningGraphBase {
  using FuncOpT = typename NodeT::FuncOpT;
  using CallOpT = typename NodeT::CallOpT;

  explicit InliningGraphBase(LLCL::Runtime &runtime)
      : runtime(runtime), state(runtime) {}

  /// Get a reference to the derived class.
  DerivedT &getDerived() { return *static_cast<DerivedT *>(this); }

  /// Build the inlining graph for a module.
  void build(ModuleOp module, const SymbolTable &symtab);

  /// Process the graph by performing all requested inlining from the root
  /// nodes.
  void process();

  // Complete processing of a node by incrementing the number of processed calls
  // of all its callers. Note that the same function can appear in the caller
  // list N, indicating that it calls this function N times. This loop will
  // increment the `numProcessedCalls` counters N times as appropriate.
  void complete(NodeT *node);

  /// The nodes in the graph. The map does not resize after it is constructed,
  /// so references always remain valid.
  llvm::MapVector<FuncOpT, NodeT> nodes;
  /// The runtime to use.
  LLCL::Runtime &runtime;

  /// The parallelism state.
  ParallelState state;
  /// The number of nodes that complete processing. If this is not equal to the
  /// number of nodes, then there are cycles in the graph.
  std::atomic<size_t> numProcessed = 0;
};
} // namespace

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::build(ModuleOp module,
                                               const SymbolTable &symtab) {
  TimeTraceScope traceScope("InliningGraphBase::build");

  // Instantiate the nodes for each generator first.
  for (auto func : llvm::make_early_inc_range(module.getOps<FuncOpT>()))
    nodes.insert(std::make_pair(func, NodeT(func)));

  // Build the graph by walking all the calls in each function and adding edges
  // as appropriate.
  auto workFn = [this, &symtab](std::pair<FuncOpT, NodeT> &value) {
    auto &[func, node] = value;
    NodeT *callerNode = &node;
    func.getBodyRegion().walk([&](CallOpT call) {
      Operation *calleeOp = symtab.lookup(
          cast<FlatSymbolRefAttr>(
              cast<SymbolConstantAttr>(call.getCallee()).getSymbol())
              .getAttr());
      assert(calleeOp && "invalid IR?");
      // Only add the edge if the symbol we found is of the type we expect.
      auto callee = dyn_cast<FuncOpT>(calleeOp);
      if (!callee)
        return;

      NodeT *calleeNode = &nodes.find(callee)->second;
      // Filter calls that do not satisfy the inlining level.
      if (!getDerived().shouldInline(calleeNode))
        return;
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

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::complete(NodeT *node) {
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
    state.startWork();
    runtime.getWorkQueue()->addTask([caller, this] {
      // Compute the parameter use-def graph of the function as a caller.
      // Inline all callees.
      getDerived().performInlining(caller);
      complete(caller);
      state.endWork();
    });
  }
}

template <typename DerivedT, typename NodeT>
void InliningGraphBase<DerivedT, NodeT>::process() {
  TimeTraceScope traceScope("InliningGraphBase::process");

  // Populate the worklist with root nodes.
  for (auto &[func, node] : nodes) {
    // Root nodes are already complete.
    if (!node.callsites.empty())
      continue;
    NodeT *caller = &node;
    // Increment the number of in-flight tasks.
    state.startWork();
    runtime.getWorkQueue()->addTask([caller, this] {
      complete(caller);
      state.endWork();
    });
  }
  // Wait on all active work items.
  state.await();
}

//===----------------------------------------------------------------------===//
// ParametricInliningGraph
//===----------------------------------------------------------------------===//

namespace {
struct ParametricInliningGraphNode
    : public InliningGraphNodeBase<ParametricInliningGraphNode, GeneratorOp,
                                   CallOp> {
  explicit ParametricInliningGraphNode(GeneratorOp func)
      : InliningGraphNodeBase(func), level(func.getAlwaysInlineLevel()),
        calleeParamGraph(func.getBodyRegion()) {}
  ParametricInliningGraphNode(ParametricInliningGraphNode &&other)
      : InliningGraphNodeBase(other.func), level(other.level),
        calleeParamGraph(other.func.getBodyRegion()) {}

  /// Compute the caller parameter graph and declarations.
  void calculateParams(ParameterCollector::Analysis &paramCache);

  /// The inlining level of the function.
  AlwaysInlineLevel level;
  /// In parametric inlining, each function has its parameter use-def graph
  /// computed twice: once as a caller, computed when the node is being
  /// processed, and once as a callee, when the fully processed node is called
  /// from somewhere else. Stash the callee graph on the node itself.
  ParameterUseDefGraph calleeParamGraph;
  /// A set of all declarations, regardless of type, in the callee.
  llvm::SetVector<StringAttr> allDecls;
};

struct ParametricInliningGraph
    : public InliningGraphBase<ParametricInliningGraph,
                               ParametricInliningGraphNode> {
  explicit ParametricInliningGraph(AlwaysInlineLevel level,
                                   LLCL::Runtime &runtime,
                                   ParameterCollector::Analysis &paramCache)
      : InliningGraphBase(runtime), level(level),
        paramCaches(paramCache, runtime.getWorkQueue()->getParallelismLevel()),
        manglerCaches(baseManglerCache,
                      runtime.getWorkQueue()->getParallelismLevel()) {}

  /// Only inline functions that satisfy the inlining level.
  bool shouldInline(ParametricInliningGraphNode *node) const {
    assert(node->level == node->func.getAlwaysInlineLevel());
    return node->level >= level;
  }
  /// When a function is finished processing and will be inlined, compute is
  /// callee parameter graph.
  bool prepareForInlining(ParametricInliningGraphNode *node);
  /// Inline all functions by invoking the parametric inliner.
  void performInlining(ParametricInliningGraphNode *caller);

  /// The inlining level.
  AlwaysInlineLevel level;
  /// Base mangler cache instance. It is always empty.
  AttrTypeMangler::Cache baseManglerCache;
  /// Thread local parameter collector caches.
  ThreadLocalCache<ParameterCollector::Analysis> paramCaches;
  /// Thread local mangler caches.
  ThreadLocalCache<AttrTypeMangler::Cache> manglerCaches;
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

void ParametricInliningGraph::performInlining(
    ParametricInliningGraphNode *caller) {
  ParameterUseDefGraph callerParams(caller->func.getBodyRegion());
  callerParams.calculate(paramCaches.getThreadLocalCache());
  for (auto [call, callee] : caller->callsites) {
    inlineGeneratorCall(call, callee->func, callee->level, callerParams,
                        callee->calleeParamGraph, callee->allDecls,
                        manglerCaches.getThreadLocalCache());
  }
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
  explicit AlwaysInlineParametricPass(
      const AlwaysInlineParametricOptions &options = {},
      LLCL::Runtime *runtime = nullptr)
      : AlwaysInlineParametricBase(options), runtime(runtime) {}

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

  ParametricInliningGraph graph(nodebugOnly ? AlwaysInlineLevel::EnabledNoDebug
                                            : AlwaysInlineLevel::Enabled,
                                *rt, paramCache);
  graph.build(getOperation(), symtab);
  graph.process();
}

std::unique_ptr<mlir::Pass> KGEN::createAlwaysInlineParametric(
    LLCL::Runtime &runtime, const AlwaysInlineParametricOptions &options) {
  return std::make_unique<AlwaysInlineParametricPass>(options, &runtime);
}

//===----------------------------------------------------------------------===//
// inlineFunctionCall
//===----------------------------------------------------------------------===//

/// Replace the call operation with the given region using values from args for
/// the region inputs.
///
/// The region is inserted into its own scope - either a loop or async execute
/// op (depending on the type of the call). This scope is returned from the
/// function.
static std::pair<Operation *, bool> inlineRegion(IRMapping &map,
                                                 KGENCallOpInterface call,
                                                 Region &region,
                                                 bool takeBody = false) {
  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  mlir::IRRewriter b{OpBuilder(call)};
  Operation *scope;
  if (isa<CallOp>(&*call)) {
    scope = b.create<HLCF::LoopOp>(
        call.getLoc(), call->getResultTypes(), ValueRange(), label,
        HLCF::LoopUnrollFullAttr::get(call.getContext(),
                                      HLCF::LoopUnrollFull::None));
  } else if (auto asyncCall = dyn_cast<LIT::AsyncCallOp>(&*call)) {
    scope = b.create<LIT::AsyncExecuteOp>(call.getLoc(), asyncCall.getType());
  } else if (auto createClosure = dyn_cast<CreateClosureOp>(&*call)) {
    scope = b.create<StageClosureOp>(call.getLoc(), createClosure.getType());
  } else {
    llvm::report_fatal_error("unknown call operation '" +
                             call->getName().getStringRef() +
                             "' in inlining pass -- please file a bug!");
  }

  Region &scopeBody = scope->getRegion(0);
  bool returnAtEnd = isa<ReturnOp>(region.front().getTerminator());
  if (takeBody) {
    scopeBody.takeBody(region);
    for (auto [value, arg] :
         llvm::zip(call->getOperands(), scopeBody.getArguments()))
      arg.replaceAllUsesWith(value);
    scopeBody.front().eraseArguments(0, call->getNumOperands());
  } else {
    Block *block = b.createBlock(&scopeBody);
    for (auto [value, arg] :
         llvm::zip(call->getOperands(), region.getArguments()))
      map.map(arg, value);
    for (BlockArgument trailing :
         region.getArguments().drop_front(call->getNumOperands()))
      map.map(trailing,
              block->addArgument(trailing.getType(), trailing.getLoc()));
    for (Operation &op : region.getOps())
      b.clone(op, map);
  }

  unsigned numReturns = 0;
  scopeBody.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      b.setInsertionPoint(op);
      if (isa<CallOp>(&*call)) {
        b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
      } else if (isa<CreateClosureOp>(*&call)) {
        // Just `return` is ok.
      } else if (isa<LIT::AsyncCallOp>(&*call)) {
        b.replaceOpWithNewOp<LIT::AsyncReturnOp>(op, op->getOperands());
      } else {
        llvm::report_fatal_error("unknown call operation '" +
                                 call->getName().getStringRef() +
                                 "' in inlining pass -- please file a bug!");
      }

      ++numReturns;
      return WalkResult::skip();
    }
    if (isa<LIT::AsyncExecuteOp, StageClosureOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });
  b.replaceOp(call, scope->getResults());
  assert(numReturns > 0);
  return std::make_pair(scope, numReturns == 1 && returnAtEnd);
}

/// Inlining might create trivial loops with a single break at the end. This
/// function cleans it up.
static void foldTrivialLoop(Operation *op) {
  TimeTraceScope traceScope("foldTrivialLoop");

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

//===----------------------------------------------------------------------===//
// InliningGraph
//===----------------------------------------------------------------------===//

namespace {
struct InliningGraphNode
    : public InliningGraphNodeBase<InliningGraphNode, FuncOp,
                                   KGENCallOpInterface> {
  /// If the function will be inlined, removed it from the module so that it can
  /// later be erased in parallel. Ownership is passed to the node.
  explicit InliningGraphNode(FuncOp func)
      : InliningGraphNodeBase(func), level(func.getAlwaysInlineLevel()),
        signature(func.getSignature()) {
    if (shouldInline())
      func->remove();
  }

  /// This node takes ownership of the function. Everything else is
  /// default-initialized.
  explicit InliningGraphNode(InliningGraphNode &&other)
      : InliningGraphNodeBase(std::move(other)), level(other.level),
        signature(other.signature) {
    other.func = nullptr;
  }

  /// Return true if the node should be inlined.
  bool shouldInline() {
    return level != AlwaysInlineLevel::Disabled || signature.isCapturing();
  }

  /// If an error occurred during inlining, nodes can end up owning the function
  /// upon destruction. Erase the function.
  ~InliningGraphNode() {
    if (shouldInline() && func)
      func->erase();
  }

  /// The inlining level of the function.
  AlwaysInlineLevel level;
  /// The signature of the function.
  SignatureType signature;
  /// Track the number of times the function has been inlined. Once the counter
  /// reaches the number of callers, the function can be erased.
  std::atomic<size_t> numTimesInlined = 0;
};

struct InliningGraph
    : public InliningGraphBase<InliningGraph, InliningGraphNode> {
  explicit InliningGraph(LLCL::Runtime &runtime, StringAttr updateAttrName)
      : InliningGraphBase(runtime), updateAttrName(updateAttrName) {}

  /// Inline all functions marked `always_inline`.
  bool shouldInline(InliningGraphNode *node) const {
    return node->shouldInline();
  }
  /// Erase dead 'always_inline' functions.
  bool prepareForInlining(InliningGraphNode *node);
  /// Inline all functions by invoking the function inliner.
  void performInlining(InliningGraphNode *caller);

  /// When updating debug info, defer the update by tagging scope operations
  /// with an attribute. This is null if updates are not needed.
  StringAttr updateAttrName;
};
} // namespace

bool InliningGraph::prepareForInlining(InliningGraphNode *node) {
  // Skip inlining of functions with no callers. If it is an 'always_inline'
  // function, we need to erase it.
  if (node->callers.empty()) {
    if (node->shouldInline()) {
      node->func->erase();
      node->func = nullptr;
    }
    return false;
  }
  return true;
}

void InliningGraph::performInlining(InliningGraphNode *caller) {
  TimeTraceScope traceScope(
      "InliningGraph::performInlining",
      [name = caller->func.getSymName()] { return name.str(); });

  for (auto [call, callee] : caller->callsites) {
    Operation *scope;
    bool singleExit;
    // Check if this is the last use of the function.
    callee->mutex.lock_shared();
    IRMapping map;
    if (callee->numTimesInlined.fetch_add(1) + 1 == callee->callers.size()) {
      // If so, we can take the body instead of cloning it. Acquire an exclusive
      // lock to wait for all other users to finish cloning.
      callee->mutex.unlock_shared();
      llvm::sys::SmartScopedWriter<true> lock(callee->mutex);
      std::tie(scope, singleExit) =
          inlineRegion(map, call, callee->func.getBodyRegion(),
                       /*takeBody=*/true);
      // Erase the empty function.
      callee->func->erase();
      callee->func = nullptr;
    } else {
      std::tie(scope, singleExit) =
          inlineRegion(map, call, callee->func.getBodyRegion());
      callee->mutex.unlock_shared();
    }

    // If we need to perform a debug info update, defer this until inlining is
    // done. Doing an update here results in quadratic runtime as functions are
    // successively inlined and updated.
    if (updateAttrName) {
      // We don't know where the op will end up, so tag it with an attribute.
      // Encode information {singleExit, noDebug} as bits.
      uint8_t value =
          singleExit |
          ((callee->level == AlwaysInlineLevel::EnabledNoDebug) << 1);
      scope->setAttr(updateAttrName,
                     OpBuilder(scope->getContext()).getI8IntegerAttr(value));
    } else if (singleExit) {
      foldTrivialLoop(scope);
    }
  };
}

//===----------------------------------------------------------------------===//
// diagnoseInliningCycle
//===----------------------------------------------------------------------===//

namespace {
struct InliningGraphNodeRef;

struct InliningGraphNodeIterator {
  InliningGraphNode *node;
  size_t childIdx;

  bool operator==(const InliningGraphNodeIterator &rhs) const {
    return node == rhs.node && childIdx == rhs.childIdx;
  }
  bool operator!=(const InliningGraphNodeIterator &rhs) const {
    return node != rhs.node || childIdx != rhs.childIdx;
  }
  InliningGraphNodeIterator operator++() {
    ++childIdx;
    return *this;
  }
  InliningGraphNodeIterator operator++(int) {
    InliningGraphNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }
  InliningGraphNodeRef operator*();
};

struct InliningGraphNodeRef {
  InliningGraphNode *node;
  KGENCallOpInterface call;

  bool operator==(const InliningGraphNodeRef &rhs) const {
    return node == rhs.node && call == rhs.call;
  }
  bool operator!=(const InliningGraphNodeRef &rhs) const {
    return !(*this == rhs);
  }

  InliningGraphNodeIterator begin() const { return {node, 0}; }
  InliningGraphNodeIterator end() const {
    return {node, node->callsites.size()};
  }
};
} // namespace

InliningGraphNodeRef InliningGraphNodeIterator::operator*() {
  auto [call, child] = node->callsites[childIdx];
  return {child, call};
}

namespace llvm {
template <>
struct DenseMapInfo<InliningGraphNodeRef> {
  static InliningGraphNodeRef getEmptyKey() {
    return {DenseMapInfo<InliningGraphNode *>::getEmptyKey(), nullptr};
  }
  static InliningGraphNodeRef getTombstoneKey() {
    return {DenseMapInfo<InliningGraphNode *>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(const InliningGraphNodeRef &node) {
    return llvm::hash_combine(
        DenseMapInfo<InliningGraphNode *>::getHashValue(node.node),
        DenseMapInfo<Operation *>::getHashValue(node.call));
  }
  static bool isEqual(const InliningGraphNodeRef &lhs,
                      const InliningGraphNodeRef &rhs) {
    return lhs == rhs;
  }
};

template <>
struct GraphTraits<InliningGraphNode *> {
  using NodeRef = InliningGraphNodeRef;
  using ChildIteratorType = InliningGraphNodeIterator;

  static NodeRef getEntryNode(InliningGraphNode *root) {
    return {root, nullptr};
  }
  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

/// Given an inlining graph with a known cycle, diagnose the cycle error.
static void diagnoseInliningCycle(InliningGraph &g) {
  InliningGraphNode *root = nullptr;
  for (auto &[func, node] : g.nodes) {
    if (node.numProcessedCalls == node.callsites.size())
      continue;
    root = &node;
    break;
  }
  assert(root && "expected to find the root node of a cycle");
  llvm::scc_iterator<InliningGraphNode *> sccIt = llvm::scc_begin(root);
  while (!sccIt.hasCycle() && !sccIt.isAtEnd())
    ++sccIt;
  assert(sccIt.hasCycle() && "expected a cycle in the SCC");
  // Build a set of nodes in the SCC for efficient queries.
  DenseSet<InliningGraphNodeRef> sccNodes;
  for (InliningGraphNodeRef ref : *sccIt)
    sccNodes.insert(ref);

  // Determine the first cycle we can see in the SCC.
  SmallVector<InliningGraphNodeIterator> path;
  DenseSet<InliningGraphNodeRef> nodesInPath;
  InliningGraphNodeRef nextNode = sccIt->front();

  while (nodesInPath.insert(nextNode).second) {
    InliningGraphNodeIterator it = nextNode.begin();
    while (!sccNodes.contains(*it))
      ++it;
    path.push_back(it);
    nextNode = *it;
  }

  // Okay, emit the errors.
  InFlightDiagnostic diag =
      mlir::emitError(nextNode.node->func.getLoc())
      << "function has recursive call to 'always_inline' function";
  for (InliningGraphNodeIterator &edge : path) {
    InliningGraphNodeRef node = *edge;
    diag.attachNote(node.call.getLoc())
        << (&edge == &path.back() ? "call here recurses" : "through call here");
    diag.attachNote(node.node->func.getLoc())
        << (&edge == &path.back() ? "back to function here"
                                  : "to function marked 'always_inline' here");
  }
}

//===----------------------------------------------------------------------===//
// updateScopeDebugInfo
//===----------------------------------------------------------------------===//

/// Starting from an inlining scope, update debug information as appropriate and
/// fold the scope if requested. Recurse on nested scopes.
static void updateScopeDebugInfoFrom(Operation *scope, IntegerAttr tag,
                                     StringAttr updateAttrName) {
  // Unpack the bits.
  auto value = static_cast<uint8_t>(tag.getInt());
  auto singleExit = static_cast<bool>(value);
  auto noDebug = static_cast<bool>(value >> 1);

  // The scope operations contains the location of the call.
  Region &body = scope->getRegion(0);
  Location callLoc = scope->getLoc();

  // If the scope represents an `always_inline_no_debug` function, just nuke all
  // debug info and locations from here.
  if (noDebug) {
    body.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<DebugInfo::ValueOp>(op)) {
        op->erase();
        return WalkResult::skip();
      }

      op->setLoc(callLoc);
      if (isa<HLCF::LoopOp, LIT::AsyncExecuteOp, StageClosureOp>(op)) {
        auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName);
        if (tag) {
          updateScopeDebugInfoFrom(op, tag, updateAttrName);
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
  } else {
    body.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      /// DebugInfo::ValueOp instances in the inlined body will retain their
      /// DILocalVariableAttr during inlining. The scope in this needs to match
      /// the scope in the location.
      mlir::LocationAttr newLoc = mlir::CallSiteLoc::get(op->getLoc(), callLoc);
      if (auto valueOp = dyn_cast<DebugInfo::ValueOp>(op)) {
        newLoc = FusedLoc::get(op->getContext(), {newLoc},
                               valueOp.getValueInfo().getScope());
      }
      op->setLoc(newLoc);

      if (isa<HLCF::LoopOp, LIT::AsyncExecuteOp, StageClosureOp>(op)) {
        auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName);
        if (tag) {
          updateScopeDebugInfoFrom(op, tag, updateAttrName);
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
  }

  // If this scope is a trivial control-flow scope, fold it away.
  if (singleExit)
    foldTrivialLoop(scope);
}

/// Given a function, find the top-level scopes and start processing debug info
/// from there.
static void updateScopeDebugInfo(FuncOp func, StringAttr updateAttrName) {
  TimeTraceScope updateScopeDebugInfo(
      "updateScopeDebugInfo", [&func] { return func.getSymName().str(); });
  func.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<HLCF::LoopOp, LIT::AsyncExecuteOp, StageClosureOp>(op))
      return WalkResult::advance();
    auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName);
    if (!tag)
      return WalkResult::advance();
    updateScopeDebugInfoFrom(op, tag, updateAttrName);
    return WalkResult::skip();
  });
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
  explicit ForceInlinePass(const ForceInlineOptions &options = {},
                           LLCL::Runtime *runtime = nullptr)
      : ForceInlineBase(options), runtime(runtime) {}

  void runOnOperation() override;

  LLCL::Runtime *runtime;
};
} // namespace

void ForceInlinePass::runOnOperation() {
  TimeTraceScope traceScope("ForceInlinePass::runOnOperation");

  // Create a runtime instance if needed.
  auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
      runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());

  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  StringAttr updateAttrName;
  if (updateDebugInfo)
    updateAttrName = StringAttr::get(&getContext(), "inliner_debuginfo_update");
  InliningGraph graph(*rt, updateAttrName);
  graph.build(getOperation(), symtab);
  graph.process();

  // Diagnose cycles, if there are any.
  if (graph.numProcessed != graph.nodes.size()) {
    diagnoseInliningCycle(graph);
    return signalPassFailure();
  }

  // If we need to handle debug info, do that now.
  if (updateAttrName) {
    TimeTraceScope traceScope("updateDebugInfo");
    ParallelState state(*rt);
    for (auto &[func, node] : graph.nodes) {
      // Update root nodes that call `always_inline` functions.
      if (node.shouldInline() || node.callsites.empty())
        continue;
      state.startWork();
      rt->getWorkQueue()->addTask([func = func, updateAttrName, &state] {
        updateScopeDebugInfo(func, updateAttrName);
        state.endWork();
      });
    }
    state.await();
  }
}

std::unique_ptr<mlir::Pass>
KGEN::createForceInline(LLCL::Runtime &runtime,
                        const ForceInlineOptions &options) {
  return std::make_unique<ForceInlinePass>(options, &runtime);
}
