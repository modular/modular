//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass checks value lifetime invariants, e.g. that
//
//===----------------------------------------------------------------------===//

//#include "KGEN/HLCFDialect/HLCFOps.h"
//#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITOps.h"

#include "mlir/IR/Iterators.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CHECKLIFETIMES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct CheckLifetimes : impl::CheckLifetimesBase<CheckLifetimes> {
  using CheckLifetimesBase::CheckLifetimesBase;

  void runOnOperation() override {
    // Walk all functions and update them.
    bool hadError = false;

    // TODO: How do we want to handle closures?  Their uses effectively form the
    // capture list for the closure.  Should this get materialized by
    // LowerSemanticCF before this pass?

    // Walk all of the functions to find values to process.
    getOperation()->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](Operation *op) {
          if (auto ownedArg = dyn_cast<OwnedArgDeclOp>(op))
            lowerOwnedArgDeclOp(ownedArg);
        });
    if (hadError)
      return signalPassFailure();
  }

  void lowerOwnedArgDeclOp(OwnedArgDeclOp op);
};
} // namespace

// Analyze all uses of OwnedArgDeclOp, and ultimately remove it.
void CheckLifetimes::lowerOwnedArgDeclOp(OwnedArgDeclOp op) {
  op.replaceAllUsesWith(op.getValue());
  op->erase();
}
