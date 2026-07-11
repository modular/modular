//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Target/TargetLowering.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSwitch.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_KGENVERIFIERPASS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

/// True for first-party + target specific dialects; others may be mid-lowering.
static bool shouldVerifyOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  // Curated subset of registerAllKGENDialects(); keep in sync.
  return llvm::StringSwitch<bool>(dialect->getNamespace())
      .Case("kgen", true)
      .Case("pop", true)
      .Case("lit", true)
      .Case("hlcf", true)
      .Case("co", true)
      .Case("interp", true)
      .Case("M", true)
      .Case("debuginfo", true)
      .Case("nvvm", true)
      .Case("rocdl", true)
      .Default(false);
}

namespace {
struct KGENVerifierPass : public impl::KGENVerifierPassBase<KGENVerifierPass> {
  using KGENVerifierPassBase::KGENVerifierPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    size_t numErrors = 0;
    const size_t maxErrors = *KGENPassCLOptions::kgenVerifierMaxErrors();

    // Per-op (non-recursive): skips off-allowlist ops; structural/dominance
    // checks come from the PM verifier (off in MODULAR_PRODUCTION).
    if (op->walk([&](Operation *operation) {
            if (shouldVerifyOp(operation) &&
                failed(mlir::verify(operation, /*verifyRecursively=*/false)))
              ++numErrors;
            if (numErrors >= maxErrors)
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }

    TargetInfoAttr target = lookupTargetInfo(op);
    const TargetLowering *lowering =
        target ? TargetLoweringRegistry::get().lookup(target.getTriple())
               : nullptr;
    if (useMLIRVerifierOnly || !lowering || !lowering->needsVerification()) {
      if (numErrors > 0)
        signalPassFailure();
      return;
    }

    // Target-specific verification.
    if (op->walk([&](Operation *operation) {
            if (failed(lowering->verifyOp(operation)))
              ++numErrors;
            if (numErrors >= maxErrors)
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }
    if (numErrors > 0)
      signalPassFailure();
  }
};
} // namespace
