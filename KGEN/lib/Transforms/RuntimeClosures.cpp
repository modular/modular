//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "runtime-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_RUNTIMECLOSURES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct RuntimeClosuresPass : impl::RuntimeClosuresBase<RuntimeClosuresPass> {
  void runOnOperation() override;
};
} // namespace

SymbolConstantAttr callSymbolOfLiftedRegion(StageClosureOp opWithRegion,
                                            SmallVector<Value> &captures,
                                            SymbolTable &symtab) {
  assert(opWithRegion->getNumRegions() == 1);
  OpBuilder builder(opWithRegion->getParentOfType<ModuleOp>());

  llvm::SetVector<Value> captureSet;
  Region &sourceRegion = opWithRegion->getRegion(0);
  operationIsIsolatedFromAbove(opWithRegion, &captureSet);
  for (Value capture : captureSet) {
    Operation *capturingOp = capture.getDefiningOp();
    // Clone ConstantLike operations into the region.
    if (capturingOp && capturingOp->hasTrait<OpTrait::ConstantLike>()) {
      ImplicitLocOpBuilder b(capturingOp->getLoc(),
                             OpBuilder::atBlockBegin(&sourceRegion.front()));
      Operation *cloned = b.clone(*capturingOp);
      for (auto [orig, replacement] :
           llvm::zip(capturingOp->getResults(), cloned->getResults()))
        replaceAllUsesInRegionWith(orig, replacement, sourceRegion);
    } else {
      // Otherwise these are captured variables and we need to pass them as
      // arguments to the block body.
      captures.emplace_back(capture);
    }
  }

  SignatureType signatureType = opWithRegion.getType();
  // Lift the body by making source region isolated from above
  // add captures in reverse so they appear the same order in
  // the parameter list as they do in the capture order
  for (int i = captures.size() - 1; i >= 0; --i) {
    Value from = captures[i];
    BlockArgument newArg =
        sourceRegion.insertArgument((unsigned)0, from.getType(), from.getLoc());
    replaceAllUsesInRegionWith(from, newArg, sourceRegion);
  }
  auto liftedValueSignature =
      FunctionType::get(builder.getContext(), sourceRegion.getArgumentTypes(),
                        signatureType.getValueResults());

  auto liftedSignature = SignatureType::get(liftedValueSignature);
  builder.setInsertionPoint(opWithRegion->getParentOfType<FuncOp>());

  std::string name = "stage_closure";
  if (opWithRegion->hasAttr("name")) {
    auto nameMaybe =
        dyn_cast_or_null<StringAttr>(opWithRegion->getAttr("name"));
    if (nameMaybe)
      name = nameMaybe.str();
  }

  auto lifted = builder.create<FuncOp>(
      opWithRegion->getLoc(), StringAttr::get(builder.getContext(), name),
      liftedSignature, AlwaysInlineLevel::Disabled);
  symtab.insert(lifted);
  auto liftedSymbol = SymbolConstantAttr::get(
      SymbolRefAttr::get(lifted.getSymNameAttr()), liftedSignature);
  IRMapping mapper;
  sourceRegion.cloneInto(&lifted.getBodyRegion(), mapper);
  return liftedSymbol;
}

void RuntimeClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Lift nested functions
  mlir::IRRewriter rewriter{OpBuilder(theModule)};
  theModule->walk([&](StageClosureOp stageClosure) {
    SmallVector<Value> captures;
    SymbolConstantAttr symbol =
        callSymbolOfLiftedRegion(stageClosure, captures, symtab);
    rewriter.setInsertionPoint(stageClosure);
    rewriter.replaceOpWithNewOp<CreateClosureOp>(
        stageClosure, stageClosure.getType(), symbol, captures);
  });
}
