//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSet.h"

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
    region = op->getParentRegion();
    decl = dyn_cast<DeclInterface>(region->getParentOp());
  }
  return {decl, region};
}

void AlwaysInlineParametricPass::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Cache the computed parameter use-def graphs of all generators. We will need
  // to keep the graphs somewhat up-to-date. Since we are inlining top-down, we
  // can compute the graphs of the inlined callees once, and since the only
  // graphs that will be modified are those of non-inlined functions, we can
  // minimally update those graphs. We need to keep up-to-date parameter
  // declarations in each scope, since those are used to mangle parameters, and
  // merge any nested graphs in.
  DenseMap<Region *, std::unique_ptr<ParameterUseDefGraph>> graphs;
  auto getGraph = [&](GeneratorOp gen) -> ParameterUseDefGraph * {
    Region *region = &gen.getBodyRegion();
    auto it = graphs.find(region);
    if (it != graphs.end())
      return it->second.get();
    it = graphs
             .try_emplace(region,
                          std::make_unique<ParameterUseDefGraph>(*region))
             .first;
    it->second->calculate();
    return it->second.get();
  };

  for (auto gen : getOperation().getOps<GeneratorOp>()) {
    // Skip over functions that are force inlined. Start inlining from the tips.
    if (gen.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
      continue;

    // Compute the parameter uses from the top-level.
    ParameterUseDefGraph *topLevelGraph = getGraph(gen);

    // Collect all calls that inline in this function.
    struct EndStack {};
    SmallVector<SmartVariant<Operation *, EndStack>> calls;
    gen.walk([&](CallOp call) {
      auto callee = symtab.lookup<GeneratorOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
      if (callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
        calls.emplace_back(call);
    });

    // Process them. Keep a callstack for a nice error when cycles are detected.
    SmallVector<CallOp, 16> callstack;
    llvm::SetVector<Operation *, SmallVector<Operation *, 16>,
                    SmallPtrSet<Operation *, 16>>
        seenFuncs;
    unsigned labelCounter = 0;
    while (!calls.empty()) {
      SmartVariant<Operation *, EndStack> next = calls.pop_back_val();
      if (isa<EndStack>(next)) {
        callstack.pop_back();
        seenFuncs.pop_back();
        continue;
      }
      auto call = cast<CallOp>(cast<Operation *>(next));

      auto callee = symtab.lookup<GeneratorOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());

      // If we recursed onto the same function, give up and emit an error.
      if (!seenFuncs.insert(callee)) {
        InFlightDiagnostic diag = mlir::emitError(
            gen.getLoc(),
            "function has recursive call to 'always_inline' function");
        assert(callstack.size() == seenFuncs.size());
        for (auto [call, gen] : llvm::zip(callstack, seenFuncs)) {
          diag.attachNote(call.getLoc()) << "through call here";
          diag.attachNote(gen->getLoc())
              << "to function marked 'always_inline' here";
        }
        diag.attachNote(call.getLoc()) << "function call here recurses";
        diag.attachNote(callee.getLoc()) << "back to function here";
        return signalPassFailure();
      }
      callstack.push_back(call);
      calls.emplace_back(EndStack{});

      // Compute the parameter uses at the callee.
      ParameterUseDefGraph *calleeParams = getGraph(callee);
      // Get the parameters in-scope at the callsite.
      auto [nearestDecl, scopeRegion] = getNearestDeclAndRegion(call);
      ParameterUseDefGraph *callScope =
          nearestDecl == gen
              ? topLevelGraph
              : &topLevelGraph->nestedScopes.find(scopeRegion)->second;

      mlir::IRRewriter b{OpBuilder(call)};
      StringAttr label =
          b.getStringAttr(gen.getSymName() + "_param_inlined_cf_" +
                          callee.getSymName() + "_" + Twine(labelCounter++));
      auto scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                          ValueRange(), label);
      b.createBlock(&scope.getBody());

      IRMapping map;
      for (auto [value, arg] :
           llvm::zip(call.getOperands(), callee.getArguments()))
        map.map(arg, value);
      for (Operation &op : *callee.getBody())
        b.clone(op, map);
      AlwaysInlineLevel level = callee.getAlwaysInlineLevel();
      scope.walk([&](Operation *op) {
        if (op != scope) {
          // If this is an `always_inline(nodebug)`, erase the location of the
          // inlined operations by replacing them with the location of the call.
          // Otherwise, propagate the inlined location via a `CallSiteLoc`.
          if (level == AlwaysInlineLevel::EnabledNoDebug)
            op->setLoc(call.getLoc());
          else
            op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), call.getLoc()));
        }
        // Erase `debuginfo.value` operations when inlining without debug info.
        if (level == AlwaysInlineLevel::EnabledNoDebug) {
          if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
            value.erase();
            return;
          }
        }

        // Check for a call to recursively inline.
        if (auto call = dyn_cast<CallOp>(op)) {
          auto callee = symtab.lookup<GeneratorOp>(
              cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
          if (callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
            calls.emplace_back(call);
        }
      });

      // We only need to mangle declarations at the top-level scope of the
      // callee. Declarations in nested scopes will shadow. However, we have
      // "un-mangle" in nested scopes.
      DenseMap<StringAttr, StringAttr> mangledDecls;
      for (auto &[decl, _] : calleeParams->decls) {
        if (callScope->decls.find(decl) == callScope->decls.end()) {
          // This declaration will not collide.
          continue;
        }
        StringAttr mangledDecl;
        unsigned count = 0;
        do {
          mangledDecl =
              b.getStringAttr((decl.getValue() + Twine(count++)).str());
        } while (callScope->decls.find(mangledDecl) != callScope->decls.end());
        mangledDecls.try_emplace(decl, mangledDecl);
      }

      // Do name mangling.
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement([&](ParamDeclRefAttr ref) {
        auto it = mangledDecls.find(ref.getName());
        if (it != mangledDecls.end())
          return ParamDeclRefAttr::get(it->second, ref.getType());
        return ref;
      });
      for (Operation *user : calleeParams->paramOps) {
        // Skip the parent decl. It's handled after.
        if (user == callee)
          continue;
        Operation *cloned = map.lookup(user);
        replacer.replaceElementsIn(cloned, /*replaceAttrs=*/true,
                                   /*replaceLocs=*/true, /*replaceTypes=*/true);
      }
      for (auto &[name, decl] : calleeParams->decls) {
        // Skip the parent decl. It's handled after.
        if (decl.declOp == callee)
          continue;
        Operation *cloned = map.lookup(decl.declOp);
        replacer.replaceElementsIn(cloned, /*replaceAttrs=*/true,
                                   /*replaceLocs=*/true, /*replaceTypes=*/true);
        // Rename declarations.
        auto itf = cast<ParamOpInterface>(decl.declOp);
        SmallVector<ParamDeclAttr> newDecls;
        itf.walkDeclarations([&](ParamDeclAttr decl) {
          if (auto it = mangledDecls.find(decl.getName());
              it != mangledDecls.end()) {
            newDecls.push_back(ParamDeclAttr::get(it->second, decl.getType()));
          } else {
            newDecls.push_back(decl);
          }
        });
        cast<ParamOpInterface>(cloned).renameDeclarations(newDecls);
        // Populate the new declarations into the call scope graph.
        for (ParamDeclAttr decl : newDecls) {
          callScope->decls.try_emplace(
              decl.getName(),
              ParamDeclaration{decl.getType(), cloned, scopeRegion});
        }
      }

      // Mangle the DeclInterface declarations.
      b.setInsertionPointToStart(&scope.getBody().front());
      for (auto [origDecl, value] :
           llvm::zip(callee.getInputParamDecls(), call.getParamValues())) {
        ParamDeclAttr decl = origDecl;
        if (auto it = mangledDecls.find(decl.getName());
            it != mangledDecls.end())
          decl = ParamDeclAttr::get(it->second, decl.getType());
        auto declOp = b.create<ParamDeclareOp>(callee.getLoc(), decl,
                                               Attribute(value.getValue()));
        // Register the new declaration.
        callScope->decls.try_emplace(
            decl.getName(),
            ParamDeclaration{decl.getType(), declOp, scopeRegion});
      }

      // "Un-mangled" declarations in immediate nested declaration scopes. The
      // un-mangled declarations will be propagated to any further nested
      // scopes.
      callee.walk<mlir::WalkOrder::PreOrder>([&](DeclInterface nestedScope) {
        if (nestedScope == callee)
          return WalkResult::advance();

        nestedScope.walk([&](DeclInterface containedScope) {
          Operation *clonedScope = map.lookup(&*containedScope);
          for (auto [region, clonedRegion] : llvm::zip(
                   containedScope->getRegions(), clonedScope->getRegions())) {
            ParameterUseDefGraph &nestedGraph =
                calleeParams->nestedScopes.find(&region)->second;
            bool inserted =
                topLevelGraph->nestedScopes
                    .try_emplace(&clonedRegion, nestedGraph.copy(map))
                    .second;
            assert(inserted);
          }
        });

        for (Region &region : nestedScope->getRegions()) {
          // If we know the scope is parametrically isolated, there's nothing to
          // do.
          if (nestedScope.isIsolatedFromAbove(region.getRegionNumber()))
            continue;

          Operation *clonedScope = map.lookup(&*nestedScope);
          b.setInsertionPointToStart(
              &cast<FuncInterface>(clonedScope)
                   ->getRegions()[region.getRegionNumber()]
                   .front());

          // Determine which decls are captured from above and map them from
          // their mangled declaration.
          ParameterUseDefGraph &nestedUses =
              calleeParams->nestedScopes.find(&region)->second;
          Region *curScopeRegion =
              &clonedScope->getRegion(region.getRegionNumber());
          ParameterUseDefGraph &clonedUses =
              topLevelGraph->nestedScopes.find(curScopeRegion)->second;
          for (ParamDeclRefAttr nestedUse : nestedUses.usesFromAbove) {
            auto it = mangledDecls.find(nestedUse.getName());
            if (it == mangledDecls.end())
              continue;
            auto declOp = b.create<ParamDeclareOp>(
                nestedScope.getLoc(),
                ParamDeclAttr::get(nestedUse.getName(), nestedUse.getType()),
                ParamDeclRefAttr::get(it->second, nestedUse.getType()));
            clonedUses.decls.try_emplace(
                nestedUse.getName(),
                ParamDeclaration{nestedUse.getType(), declOp, curScopeRegion});
          }
        }

        return WalkResult::skip();
      });

      // Handle all terminators.
      auto newReturn = cast<KGEN::ReturnOp>(map.lookup(callee.getReturnOp()));
      b.setInsertionPoint(newReturn);
      SmallVector<Value> retVals;
      for (auto [retVal, retType] :
           llvm::zip(newReturn.getOperands(), call->getResultTypes())) {
        if (retVal.getType() != retType)
          retVals.push_back(
              b.create<RebindOp>(newReturn.getLoc(), retType, retVal));
        else
          retVals.push_back(retVal);
      }
      for (auto [decl, value] :
           llvm::zip(call.getParamDecls(), newReturn.getParameters()))
        b.create<ParamDeclareOp>(newReturn.getLoc(), decl, Attribute(value));
      b.create<HLCF::BreakOp>(newReturn.getLoc(), retVals, label);
      newReturn.erase();

      callee.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        // Walk over nested functions. Control-flow does not cross them.
        if (op != callee && isa<FuncInterface>(op))
          return WalkResult::skip();
        auto returnOp = dyn_cast<HLCF::ReturnOp>(op);
        if (!returnOp)
          return WalkResult::advance();
        auto cloned = cast<HLCF::ReturnOp>(map.lookup(returnOp));
        b.setInsertionPoint(cloned);
        SmallVector<Value> retVals;
        for (auto [retVal, retType] :
             llvm::zip(cloned.getOperands(), call->getResultTypes())) {
          if (retVal.getType() != retType)
            retVals.push_back(
                b.create<RebindOp>(cloned.getLoc(), retType, retVal));
          else
            retVals.push_back(retVal);
        }
        b.create<HLCF::BreakOp>(cloned.getLoc(), retVals, label);
        cloned.erase();
        return WalkResult::advance();
      });

      b.replaceOp(call, scope.getResults());
    }
    assert(callstack.empty() && seenFuncs.empty());
  }
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
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  for (auto func : getOperation().getOps<FuncOp>()) {
    // Skip over functions that are force inlined. Start inlining from the tips.
    if (func.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
      continue;

    // Collect all calls that inline in this function.
    struct EndStack {};
    SmallVector<SmartVariant<Operation *, EndStack>> calls;
    func.walk([&](CallOp call) {
      auto callee = symtab.lookup<FuncOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
      if (callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
        calls.emplace_back(call);
    });

    // Process them. Keep a callstack for a nice error when cycles are detected.
    SmallVector<Location, 16> callstack;
    llvm::SetVector<Operation *, SmallVector<Operation *, 16>,
                    SmallPtrSet<Operation *, 16>>
        seenFuncs;
    unsigned labelCounter = 0;
    while (!calls.empty()) {
      SmartVariant<Operation *, EndStack> next = calls.pop_back_val();
      if (isa<EndStack>(next)) {
        callstack.pop_back();
        seenFuncs.pop_back();
        continue;
      }
      auto call = cast<CallOp>(cast<Operation *>(next));

      auto callee = symtab.lookup<FuncOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());

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
        return signalPassFailure();
      }
      callstack.push_back(call.getLoc());
      calls.emplace_back(EndStack{});

      mlir::IRRewriter b{OpBuilder(call)};
      StringAttr label =
          b.getStringAttr(func.getSymName() + "_inlined_cf_" +
                          callee.getSymName() + "_" + Twine(labelCounter++));
      auto scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                          ValueRange(), label);
      b.createBlock(&scope.getBody());

      IRMapping map;
      for (auto [value, arg] :
           llvm::zip(call.getOperands(), callee.getArguments()))
        map.map(arg, value);
      for (Operation &op : *callee.getBody())
        b.clone(op, map);
      unsigned numReturns = 0;
      AlwaysInlineLevel level = callee.getAlwaysInlineLevel();
      scope.walk([&](Operation *op) {
        if (op != scope) {
          // If this is an `always_inline(nodebug)`, erase the location of the
          // inlined operations by replacing them with the location of the call.
          // Otherwise, propagate the inlined location via a `CallSiteLoc`.
          if (level == AlwaysInlineLevel::EnabledNoDebug)
            op->setLoc(call.getLoc());
          else
            op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), call.getLoc()));
        }
        // Erase `debuginfo.value` operations when inlining without debug info.
        if (level == AlwaysInlineLevel::EnabledNoDebug) {
          if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
            value.erase();
            return;
          }
        }

        // Check for a call to recursively inline.
        if (auto call = dyn_cast<CallOp>(op)) {
          auto callee = symtab.lookup<FuncOp>(
              cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());
          if (callee.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled)
            calls.emplace_back(call);
        }

        // Replace all returns with breaks to the control flow scope.
        if (!isa<KGEN::ReturnOp, HLCF::ReturnOp>(op))
          return;
        b.setInsertionPoint(op);
        b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
        ++numReturns;
      });
      b.replaceOp(call, scope.getResults());
      // If the scope was trivial (one return), fold it away.
      // FIXME: This is required to work around a bug (?) in one of LLVM's
      // IRTranslator passes when compiling with `kgen -O0 -debug-level=full`.
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
  }
}
