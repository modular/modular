//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/StringSet.h"

using namespace M;
using namespace KGEN;

void KGEN::inlineGeneratorCall(KGENCallOpInterface call, GeneratorOp callee) {
  auto parent = call->getParentOfType<GeneratorOp>();
  assert(parent && "expected call to be inlined to be inside a generator");
  // Compute parameter uses from the top-level.
  ParameterDeclsAndUses parentParams, calleeParams;
  DenseMap<DeclInterface, ParameterDeclsAndUses> parentScopes =
      parentParams.calculate(parent);
  DenseMap<DeclInterface, ParameterDeclsAndUses> calleeScopes =
      calleeParams.calculate(callee);

  // Get the parameters in-scope at the callsite.
  auto callDecl = call->getParentOfType<DeclInterface>();
  ParameterDeclsAndUses &callScope =
      callDecl == parent ? parentParams : parentScopes.find(callDecl)->second;

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
  for (Operation &op : *callee.getBody())
    b.insert(op.clone(bv));

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
  // Skip over parametrically isolated attributes.
  replacer.addReplacement([&](ExprFuncAttr exprFunc) {
    return std::make_pair(exprFunc, WalkResult::skip());
  });
  auto paramDeclsAttrName = b.getStringAttr("paramDecls");
  for (auto &[user, uses] : calleeParams.usersAndDeclarers) {
    // Skip the parent decl. It's handled after.
    if (user == callee)
      continue;
    Operation *cloned = bv.lookup(user);
    replacer.replaceElementsIn(cloned, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    // Rename declarations.
    if (auto paramDecls =
            user->getAttrOfType<ParamDeclArrayAttr>(paramDeclsAttrName)) {
      SmallVector<ParamDeclAttr> newDecls;
      for (ParamDeclAttr decl : paramDecls) {
        if (auto it = mangledDecls.find(decl.getName());
            it != mangledDecls.end())
          newDecls.push_back(ParamDeclAttr::get(it->second, decl.getType()));
        else
          newDecls.push_back(decl);
      }
      cloned->setAttr(paramDeclsAttrName,
                      ParamDeclArrayAttr::get(b.getContext(), newDecls));
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
    if (cast<DeclInterface>(*nestedScope).isIsolatedFromAbove())
      return WalkResult::skip();

    Operation *clonedScope = bv.lookup(&*nestedScope);
    b.setInsertionPointToStart(
        &cast<FuncInterface>(clonedScope)->getRegions().front().front());

    // Determine which decls are captured from above and map them from their
    // mangled declaration.
    ParameterDeclsAndUses &nestedUses =
        calleeScopes.find(cast<DeclInterface>(*nestedScope))->second;
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
       llvm::zip(call->getAttrOfType<ParamDeclArrayAttr>(paramDeclsAttrName),
                 newReturn.getParameters()))
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
