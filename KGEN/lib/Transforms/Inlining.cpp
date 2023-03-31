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
// AlwaysInlineParametricPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ALWAYSINLINEPARAMETRIC
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct AlwaysInlineParametricPass
    : impl::AlwaysInlineParametricBase<AlwaysInlineParametricPass> {
  using AlwaysInlineParametricBase::AlwaysInlineParametricBase;

  void runOnOperation() override;
};
} // namespace

/// Get the nearest declaration from the operation and the region of the
/// declaration that contains the operation.
static std::pair<DeclInterface, Region *>
getNearestDeclAndRegion(Operation *op) {
  Region *region = op->getParentRegion();
  auto decl = dyn_cast<DeclInterface>(region->getParentOp());
  while (!decl) {
    region = region->getParentRegion();
    decl = dyn_cast<DeclInterface>(region->getParentOp());
  }
  return {decl, region};
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

namespace {
/// Signature types define a nested parameter scope inside a parameter
/// expression. Manually walk and mangle parameter references in attributes and
/// types in an expression tree while accounting for name shadowing in a
/// signature type.
struct AttrTypeMangler {
  using ReflessCache = DenseSet<const void *>;

  explicit AttrTypeMangler(ReflessCache &cache) : noNestedRefs(cache) {}

  /// Populate the mangler using the decls in two potentially conflicting
  /// scopes. Returns false if there is nothing to mangle.
  bool populate(Builder &b, const ParameterUseDefGraph &curScope,
                const ParameterUseDefGraph &inlinedScope) {
    TimeTraceScope</*Enabled=*/false> traceScope("AttrTypeMangler::populate");

    bool needsMangling = false;
    for (auto &[decl, _] : inlinedScope.decls) {
      if (curScope.decls.find(decl) == curScope.decls.end()) {
        // This declaration will not collide.
        continue;
      }
      StringAttr mangledDecl;
      unsigned count = 0;
      do {
        mangledDecl = b.getStringAttr((decl.getValue() + Twine(count++)).str());
      } while (curScope.decls.find(mangledDecl) != curScope.decls.end());
      mangledDecls.try_emplace(decl, mangledDecl);
      needsMangling = true;
    }
    return needsMangling;
  }

  template <typename T, typename U = std::conditional_t<
                            std::is_base_of_v<Type, T>, Type, Attribute>>
  U mangleRefsInImpl(T value, bool &hasRefs) {
    if (noNestedRefs.contains(value.getAsOpaquePointer()))
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
      noNestedRefs.insert(value.getAsOpaquePointer());
    return changed ? value.replaceImmediateSubElements(replAttrs, replTypes)
                   : value;
  }

  Type mangleRefsIn(Type type, bool &hasRefs) {
    return mangleRefsInImpl(type, hasRefs);
  }

  Attribute mangleRefsIn(Attribute attr, bool &hasRefs) {
    if (auto ref = dyn_cast<ParamDeclRefAttr>(attr)) {
      hasRefs = true;
      if (StringAttr mangled = mangledDecls.lookup(ref.getName()))
        return ParamDeclRefAttr::get(mangled,
                                     mangleRefsIn(ref.getType(), hasRefs));
    }
    return mangleRefsInImpl(attr, hasRefs);
  }

  ParamDeclAttr mangleDecl(ParamDeclAttr decl, bool needsMangling) {
    if (!needsMangling)
      return decl;
    bool hasRefs = false;
    Type type = mangleRefsIn(decl.getType(), hasRefs);
    if (StringAttr mangled = mangledDecls.lookup(decl.getName()))
      return ParamDeclAttr::get(mangled, type);
    if (type == decl.getType())
      return decl;
    return ParamDeclAttr::get(decl.getName(), type);
  }

  void mangleElementsIn(Operation *op) {
    TimeTraceScope<> traceScope("AttrTypeMangler::mangleElementsIn");

    bool unused;
    op->setAttrs(
        cast<DictionaryAttr>(mangleRefsIn(op->getAttrDictionary(), unused)));
    op->setLoc(cast<mlir::LocationAttr>(
        mangleRefsIn(mlir::LocationAttr(op->getLoc()), unused)));

    for (OpResult result : op->getResults())
      result.setType(mangleRefsIn(result.getType(), unused));

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          arg.setLoc(cast<mlir::LocationAttr>(
              mangleRefsIn(mlir::LocationAttr(arg.getLoc()), unused)));
          arg.setType(mangleRefsIn(arg.getType(), unused));
        }
      }
    }
  }

  void recursivelyMangle(Region *scope, const ParameterUseDefGraph &graph) {
    TimeTraceScope</*Enabled=*/false> traceScope(
        "AttrTypeMangler::recursivelyMangle");

    // Exit early if the scope is parametrically isolated.
    if (cast<DeclInterface>(scope->getParentOp())
            .isIsolatedFromAbove(scope->getRegionNumber()))
      return;

    const ParameterUseDefGraph &uses = graph.nestedScopes.find(scope)->second;
    AttrTypeMangler mangler(noNestedRefs);
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

  DenseMap<StringAttr, StringAttr> mangledDecls;

  ReflessCache &noNestedRefs;
};
} // namespace

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

LogicalResult KGEN::inlineGeneratorCall(
    KGENCallOpInterface topCall, ParameterUseDefGraph &topLevelGraph,
    ParameterCollector::Analysis &paramCache,
    function_ref<ParameterUseDefGraph &(ParameterCollector::Analysis &,
                                        GeneratorOp)>
        getGraph,
    function_ref<GeneratorOp(KGENCallOpInterface)> lookupCallee) {

  // Collect all calls that inline in this function.
  struct EndStack {};
  SmallVector<SmartVariant<KGENCallOpInterface, EndStack>> calls = {topCall};

  // A cache of attributes that have no references inside them. This is used by
  // the attribute mangler.
  AttrTypeMangler::ReflessCache manglerCache;

  // Process them. Keep a callstack for a nice error when cycles are detected.
  SmallVector<Location, 16> callstack;
  llvm::SetVector<Operation *, SmallVector<Operation *, 16>,
                  SmallPtrSet<Operation *, 16>>
      seenFuncs;
  StringAttr label = StringAttr::get(topCall.getContext(), "inlined_cf_scope");
  while (!calls.empty()) {
    SmartVariant<KGENCallOpInterface, EndStack> next = calls.pop_back_val();
    if (isa<EndStack>(next)) {
      callstack.pop_back();
      seenFuncs.pop_back();
      continue;
    }
    auto call = cast<KGENCallOpInterface>(next);
    GeneratorOp callee = lookupCallee(call);
    // If the lookup returns nothing, then we end the inlining.
    if (!callee) {
      callstack.push_back(call.getLoc());
      calls.emplace_back(EndStack{});
      continue;
    }

    assert(callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled);
    TimeTraceScope<> traceScope("callee",
                                [&] { return callee.getSymName().str(); });

    // If we recursed onto the same function, give up. Don't emit an error
    // because the recursion could be resolved by the elaborator.
    if (!seenFuncs.insert(callee))
      continue;
    callstack.push_back(call.getLoc());
    calls.emplace_back(EndStack{});

    // Compute the parameter uses at the callee.
    const ParameterUseDefGraph &calleeParams = getGraph(paramCache, callee);
    // Get the parameters in-scope at the callsite.
    auto [_, scopeRegion] = getNearestDeclAndRegion(call);
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
    bool needsMangling = mangler.populate(b, *callScope, calleeParams);

    // Make sure to rebind the call operands based on the mangled types of the
    // callee's argument types.
    SmallVector<Type> argTypes = llvm::to_vector(callee.getArgumentTypes());
    if (needsMangling) {
      for (Type &type : argTypes) {
        bool unused;
        type = mangler.mangleRefsIn(type, unused);
      }
    }
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
      propagateNewDecls(newDecls, topLevelGraph, *callScope, cloned,
                        scopeRegion);
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

      // Check for a call to recursively inline.
      if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
        auto callee = lookupCallee(call);
        if (callee &&
            callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
          calls.emplace_back(call);
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
      b.replaceOp(scope,
                  scope.getBody().front().getTerminator()->getOperands());
    }
  }
  assert(callstack.empty() && seenFuncs.empty());
  return success();
}

void AlwaysInlineParametricPass::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto &paramCache = getAnalysis<ParameterCollector::Analysis>();

  // Cache the computed parameter use-def graphs of all generators. We will need
  // to keep the graphs somewhat up-to-date. Since we are inlining top-down, we
  // can compute the graphs of the inlined callees once, and since the only
  // graphs that will be modified are those of non-inlined functions, we can
  // minimally update those graphs. We need to keep up-to-date parameter
  // declarations in each scope, since those are used to mangle parameters, and
  // merge any nested graphs in.
  DenseMap<GeneratorOp, std::unique_ptr<ParameterUseDefGraph>> graphs;
  llvm::sys::SmartRWMutex<true> graphsMtx;
  auto getGraph = [&graphsMtx,
                   &graphs](ParameterCollector::Analysis &paramCache,
                            GeneratorOp gen) -> ParameterUseDefGraph & {
    {
      llvm::sys::SmartScopedReader<true> lock(graphsMtx);
      if (auto it = graphs.find(gen); it != graphs.end())
        return *it->second;
    }

    // Don't compute the graph inside the critical section. This means it's
    // possible for more than one thread to compute the graph for the same
    // generator, but it also means that computations can be parallelized if

    auto graph = std::make_unique<ParameterUseDefGraph>(gen.getBodyRegion());
    graph->calculate(paramCache);

    llvm::sys::SmartScopedWriter<true> lock(graphsMtx);
    return *graphs.try_emplace(gen, std::move(graph)).first->second;
  };

  std::vector<GeneratorOp> rootGens;
  for (auto gen : getOperation().getOps<GeneratorOp>()) {
    // Skip over functions that are force inlined. Start inlining from the tips.
    if (gen.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
      continue;
    rootGens.push_back(gen);
  }
  auto workFunc = [&symtab, &getGraph, &paramCache](GeneratorOp gen) mutable {
    ParameterCollector::Analysis cache = paramCache;
    auto lookupCallee = [&](KGENCallOpInterface call) -> GeneratorOp {
      if (auto concreteCall = dyn_cast<CallOp>(call.getOperation()))
        return symtab.lookup<GeneratorOp>(
            cast<FlatSymbolRefAttr>(concreteCall.getCallee().getSymbol())
                .getAttr());
      // Only handles concrete CallOps, not CallParam ops.
      return nullptr;
    };

    // Compute the use-def graph for this generator.
    ParameterUseDefGraph &useDefGraph = getGraph(cache, gen);

    // Walk all the calls in this generator and do the inlining if needed.
    WalkResult callWalk = gen.walk([&](CallOp call) {
      auto callee = symtab.lookup<GeneratorOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
      // If the callee doesn't exist, or it's not marked always_inline, move on.
      if (!(callee &&
            callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled))
        return WalkResult::advance();

      return WalkResult(inlineGeneratorCall(call, useDefGraph, cache, getGraph,
                                            lookupCallee));
    });
    if (callWalk.wasInterrupted())
      return failure();

    return mlir::success();
  };
  if (failed(mlir::failableParallelForEach(&getContext(), rootGens, workFunc)))
    return signalPassFailure();
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
