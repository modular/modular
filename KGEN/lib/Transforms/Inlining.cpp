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
// Parameteric Inlining
//===----------------------------------------------------------------------===//

void KGEN::inlineGeneratorCall(KGENCallOpInterface call, GeneratorOp callee) {
  auto parent = call->getParentOfType<GeneratorOp>();
  assert(parent && "expected call to be inlined to be inside a generator");
  // Compute parameter uses from the top-level.
  ParameterUseDefGraph parentParams(parent.getBodyRegion()),
      calleeParams(callee.getBodyRegion());
  parentParams.calculate();
  calleeParams.calculate();

  // Get the parameters in-scope at the callsite.
  auto callDecl = call->getParentOfType<DeclInterface>();
  ParameterUseDefGraph &callScope =
      callDecl == parent
          ? parentParams
          : parentParams.nestedScopes.find(&callDecl->getRegion(0))->second;

  // Wrap the callee in a loop with a unique label.
  StringSet<> takenLabels;
  callee.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk over nested functions. Control-flow does not cross them.
    if (op != callee && isa<FuncInterface>(op))
      return WalkResult::skip();
    auto loop = dyn_cast<HLCF::LoopOp>(op);
    if (loop && loop.getLabelAttr())
      takenLabels.insert(*loop.getLabel());
    return WalkResult::advance();
  });

  std::string label = "inlined_cf_scope";
  unsigned count = 0;
  while (takenLabels.contains(label))
    label = ("inlined_cf_scope_" + Twine(count++)).str();

  OpBuilder b(call);
  StringAttr loopLabel = b.getStringAttr(label);
  auto scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                      ValueRange(), loopLabel);
  scope.getBody().push_back(new Block);
  b.setInsertionPointToStart(&scope.getBody().front());

  // Clone the operations in the immediate function body.
  IRMapping bv;
  for (Operation &op : *callee.getBody()) {
    b.insert(op.clone(bv))->walk([&](Operation *op) {
      op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), call.getLoc()));
    });
  }

  // We only need to mangle delcarations at the top-level scope of the callee.
  // Declarations in nested scopes will shadow. However, we have "un-mangle" in
  // nested scopes.
  DenseMap<StringAttr, StringAttr> mangledDecls;
  for (auto &[decl, _] : calleeParams.decls) {
    if (callScope.decls.find(decl) == callScope.decls.end()) {
      // This declaration will not collide.
      continue;
    }
    StringAttr mangledDecl;
    count = 0;
    do {
      mangledDecl = b.getStringAttr((decl.getValue() + Twine(count++)).str());
    } while (callScope.decls.find(mangledDecl) != callScope.decls.end());
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
  for (Operation *user : calleeParams.paramOps) {
    // Skip the parent decl. It's handled after.
    if (user == callee)
      continue;
    Operation *cloned = bv.lookup(user);
    replacer.replaceElementsIn(cloned, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
  }
  for (auto &[name, decl] : calleeParams.decls) {
    // Skip the parent decl. It's handled after.
    if (decl.declOp == callee)
      continue;
    Operation *cloned = bv.lookup(decl.declOp);
    replacer.replaceElementsIn(cloned, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    // Rename declarations.
    if (auto itf = dyn_cast<ParamOpInterface>(decl.declOp)) {
      SmallVector<ParamDeclAttr> newDecls;
      itf.walkDeclarations([&](ParamDeclAttr decl) {
        if (auto it = mangledDecls.find(decl.getName());
            it != mangledDecls.end())
          newDecls.push_back(ParamDeclAttr::get(it->second, decl.getType()));
        else
          newDecls.push_back(decl);
      });
      cast<ParamOpInterface>(cloned).renameDeclarations(newDecls);
    }
  }

  // Mangle the DeclInterface declarations.
  b.setInsertionPointToStart(&scope.getBody().front());
  for (auto [origDecl, value] :
       llvm::zip(callee.getInputParamDecls(), call.getParamValues())) {
    ParamDeclAttr decl = origDecl;
    if (auto it = mangledDecls.find(decl.getName()); it != mangledDecls.end())
      decl = ParamDeclAttr::get(it->second, decl.getType());
    b.create<ParamDeclareOp>(callee.getLoc(), decl,
                             Attribute(value.getValue()));
  }

  // "Un-mangled" declarations in immediate nested declaration scopes. The
  // un-mangled declarations will be propagated to any further nested scopes.
  callee.walk<mlir::WalkOrder::PreOrder>([&](FuncInterface nestedScope) {
    if (nestedScope == callee)
      return WalkResult::advance();
    // If we know the scope is parametrically isolated, there's nothing to do.
    // FuncInterface things have one region, so default to zero.
    if (cast<DeclInterface>(*nestedScope).isIsolatedFromAbove(0))
      return WalkResult::skip();

    Operation *clonedScope = bv.lookup(&*nestedScope);
    b.setInsertionPointToStart(
        &cast<FuncInterface>(clonedScope)->getRegions().front().front());

    // Determine which decls are captured from above and map them from their
    // mangled declaration.
    ParameterUseDefGraph &nestedUses =
        calleeParams.nestedScopes.find(&nestedScope->getRegion(0))->second;
    for (ParamDeclRefAttr nestedUse : nestedUses.usesFromAbove) {
      auto it = mangledDecls.find(nestedUse.getName());
      if (it == mangledDecls.end())
        continue;
      b.create<ParamDeclareOp>(
          nestedScope.getLoc(),
          ParamDeclAttr::get(nestedUse.getName(), nestedUse.getType()),
          ParamDeclRefAttr::get(it->second, nestedUse.getType()));
    }

    return WalkResult::skip();
  });

  // Handle all terminators.
  auto newReturn = cast<KGEN::ReturnOp>(bv.lookup(callee.getReturnOp()));
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
  b.create<HLCF::BreakOp>(newReturn.getLoc(), retVals, loopLabel);
  newReturn.erase();

  callee.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk over nested functions. Control-flow does not cross them.
    if (op != callee && isa<FuncInterface>(op))
      return WalkResult::skip();
    auto returnOp = dyn_cast<HLCF::ReturnOp>(op);
    if (!returnOp)
      return WalkResult::advance();
    auto cloned = cast<HLCF::ReturnOp>(bv.lookup(returnOp));
    b.setInsertionPoint(cloned);
    SmallVector<Value> retVals;
    for (auto [retVal, retType] :
         llvm::zip(cloned.getOperands(), call->getResultTypes())) {
      if (retVal.getType() != retType)
        retVals.push_back(b.create<RebindOp>(cloned.getLoc(), retType, retVal));
      else
        retVals.push_back(retVal);
    }
    b.create<HLCF::BreakOp>(cloned.getLoc(), retVals, loopLabel);
    cloned.erase();
    return WalkResult::advance();
  });

  call->replaceAllUsesWith(scope.getResults());
  call.erase();
}

//===----------------------------------------------------------------------===//
// TestParametricInlinePass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_TESTPARAMETRICINLINEPASS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class TestParametricInlinePass
    : public M::KGEN::impl::TestParametricInlinePassBase<
          TestParametricInlinePass> {
public:
  using TestParametricInlinePassBase::TestParametricInlinePassBase;

  void runOnOperation() override {
    GeneratorOp parentGen;
    getOperation()->walk([&](GeneratorOp gen) {
      if (gen.getName() == parent) {
        parentGen = gen;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!parentGen)
      return signalPassFailure();

    KGENCallOpInterface callToInline;
    parentGen.walk([&](KGENCallOpInterface call) {
      if (auto cst = dyn_cast<SymbolConstantAttr>(call.getCallee())) {
        if (cst.getSymbol().getRootReference().getValue() == callee) {
          callToInline = call;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (!callToInline)
      return signalPassFailure();

    SymbolTable symtab(getOperation());

    inlineGeneratorCall(callToInline, symtab.lookup<GeneratorOp>(callee));
  }
};
} // namespace

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

      auto callee = symtab.lookup<FuncOp>(
          cast<FlatSymbolRefAttr>(call.getCallee().getSymbol()).getAttr());

      // If we recursed onto the same function, give up and emit an error.
      if (!seenFuncs.insert(callee)) {
        InFlightDiagnostic diag = mlir::emitError(
            func.getLoc(),
            "function has recursive call to 'always_inline' function");
        assert(callstack.size() == seenFuncs.size());
        for (auto [call, func] : llvm::zip(callstack, seenFuncs)) {
          diag.attachNote(call.getLoc()) << "through call here";
          diag.attachNote(func->getLoc())
              << "to function marked 'always_inline' here";
        }
        diag.attachNote(call.getLoc()) << "function call here recurses";
        diag.attachNote(callee.getLoc()) << "back to function here";
        return signalPassFailure();
      }
      callstack.push_back(call);
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
