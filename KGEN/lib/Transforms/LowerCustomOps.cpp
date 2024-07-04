//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerCustomOpsPass : KGEN::impl::LowerCustomOpsBase<LowerCustomOpsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerCustomOpsPass::runOnOperation() {
  Operation *toplevelOp = getOperation();
  Dialect *customDialect =
      toplevelOp->getContext()->getLoadedDialect<CustomDialect>();

  // Only lazily fetch the op implementation symbols, so we don't pay this
  // cost when there is no custom operations.
  CustomOpImplArrayAttr opImpls;

  toplevelOp->walk([&opImpls, toplevelOp, customDialect](Operation *op) {
    // Only convert operations from the custom dialect.
    if (op->getDialect() != customDialect)
      return WalkResult::advance();

    // Fetch the op implementations symbols if they were not yet fetched.
    if (!opImpls) {
      auto opImplsOp = CustomOpImplsOp::lookupOp(toplevelOp);
      if (!opImplsOp) {
        op->emitError() << "no '" << CustomOpImplsOp::getOperationName()
                        << "' op found at the toplevel module";
        return WalkResult::interrupt();
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

    // Replace the custom op with a call to its implementation.
    OpBuilder builder(op);
    auto callOp = builder.create<KGEN::CallOp>(
        op->getLoc(), op->getResultTypes(), opImpl.getOpImplementation(),
        op->getOperands());
    op->replaceAllUsesWith(callOp->getResults());
    op->erase();
    return WalkResult::advance();
  });
}
