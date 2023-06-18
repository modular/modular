//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_CANONICALIZER
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    mlir::GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns, config);
  }

  mlir::FrozenRewritePatternSet patterns;
};
} // namespace
