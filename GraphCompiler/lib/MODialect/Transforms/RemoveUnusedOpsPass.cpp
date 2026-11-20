//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_REMOVEUNUSEDOPS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

static bool isUnusedOp(Operation *op) {
  return isa<MO::MODialect>(op->getDialect()) && op->getNumRegions() == 0 &&
         !op->hasTrait<mlir::OpTrait::IsTerminator>() &&
         !isa<MO::InvokeShapeFuncOp>(op) && op->use_empty();
}

static bool isSafeToErase(Operation *op) {
  return isa<MO::AllReduceSumOp, MO::DistributedBroadcastOp,
             MO::DistributedScatterOp>(op);
}

struct EraseDeadOpPattern : RewritePattern {
  EraseDeadOpPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isUnusedOp(op))
      return failure();

    // TODO(GEX-3289): An unused op may still have user-visible side effects:
    // the Graph/Executor Python API never returns chains as graph outputs, so
    // no uses does not imply dead. Ops like AllReduce are safe to drop when
    // their output is unused. Remove this check once chains are returned.
    if (!isSafeToErase(op))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveUnusedOpsPass
    : public M::MO::impl::RemoveUnusedOpsBase<RemoveUnusedOpsPass> {
  using RemoveUnusedOpsBase::RemoveUnusedOpsBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<EraseDeadOpPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
