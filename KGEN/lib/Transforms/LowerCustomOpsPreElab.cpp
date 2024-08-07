//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCUSTOMOPSPREELAB
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerCustomOpsPreElabPass
    : KGEN::impl::LowerCustomOpsPreElabBase<LowerCustomOpsPreElabPass> {
  void runOnOperation() override;
};
} // namespace

void LowerCustomOpsPreElabPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  Dialect *customDialect =
      theModule->getContext()->getLoadedDialect<CustomDialect>();

  // Only lazily fetch the op implementation symbols, so we don't pay this
  // cost when there is no custom operations.
  CustomOpImplArrayAttr opImpls;

  WalkResult walkSuceeded = theModule->walk([&](Operation *op) -> WalkResult {
    // Only convert operations from the custom dialect.
    if (op->getDialect() != customDialect)
      return WalkResult::advance();

    // Fetch the op implementations symbols if they were not yet fetched.
    if (!opImpls) {
      auto opImplsOp = CustomOpImplsOp::lookupOp(theModule);
      if (!opImplsOp) {
        return op->emitError() << "no '" << CustomOpImplsOp::getOperationName()
                               << "' op found at the top-level module";
      }
      opImpls = opImplsOp.getImplsAttr();
    }

    // Get our current op implementation.
    StringAttr opName = op->getName().getIdentifier();
    CustomOpImplAttr opImpl = opImpls.getOpImpl(opName);
    if (!opImpl) {
      op->emitError() << "no implementation found for custom op '"
                      << opName.strref() << "'";
      return WalkResult::interrupt();
    }

    OpBuilder builder(op);
    Location loc = op->getLoc();

    SymbolConstantAttr opImplSym = opImpl.getOpImplementation();

    Attribute implParamsAttr = op->getAttr(kCustomOpParamsAttrName);
    // If our operation has parameters, specialize the op implementation
    // symbol for these parameters.
    if (implParamsAttr) {
      // FIXME(math-fehr): Support multiple parameters
      SmallVector<TypedAttr> parameters;
      parameters.push_back(cast<TypedAttr>(implParamsAttr));

      SignatureType specializedSig =
          opImplSym.getType().getSpecializedSignature(parameters, loc);
      opImplSym = SymbolConstantAttr::get(opImplSym.getSymbol(), parameters,
                                          specializedSig);
    }

    // Replace the custom op with a call to its implementation.
    auto callOp = builder.create<KGEN::CallOp>(
        op->getLoc(), op->getResultTypes(), opImplSym, op->getOperands());
    op->replaceAllUsesWith(callOp->getResults());
    op->erase();
    return WalkResult::advance();
  });

  if (walkSuceeded == WalkResult::interrupt()) {
    signalPassFailure();
    return;
  }

  // Erase the op implementations op, as we won't use it anymore.
  // This is necessary to DCE op implementation symbols.
  auto opImplsOp = CustomOpImplsOp::lookupOp(theModule);
  if (opImplsOp)
    opImplsOp->erase();
}
