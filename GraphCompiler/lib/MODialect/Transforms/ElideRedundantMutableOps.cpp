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

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>

#define DEBUG_TYPE "elide-redundant-store-load-pairs"

namespace M::MO {

#define GEN_PASS_DEF_ELIDEREDUNDANTMUTABLEOPS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

namespace {

struct EliminateRedundantLoads
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MutableLoadOp> {

  EliminateRedundantLoads(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableLoadOp loadOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    auto inChain = loadOp.getInChain();

    auto storeOp = dyn_cast_if_present<MutableStoreOp>(inChain.getDefiningOp());

    if (!inChain.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          loc, "load elision is possible only when it has exclusive access to "
               "its input chain");
    }

    if (!storeOp)
      return rewriter.notifyMatchFailure(
          loc, "input chain not produced by a mo.mutable.store op");

    if (storeOp.getInBuffer() != loadOp.getInBuffer())
      return rewriter.notifyMatchFailure(
          loc, "preceding store op does not operate on the same mo.buffer");

    rewriter.replaceAllUsesWith(loadOp.getOutTensor(), storeOp.getInTensor());
    rewriter.replaceAllUsesWith(loadOp.getOutChain(), storeOp.getOutChain());

    rewriter.eraseOp(loadOp);

    return success();
  }
};

struct EliminateRedundantStores
    : public DeviceAwareRewritePattern<OpRewritePattern, MO::MutableStoreOp> {

  EliminateRedundantStores(MLIRContext *ctx)
      : DeviceAwareRewritePattern(ctx, 1) {}

  LogicalResult
  deviceAwareMatchAndRewrite(MO::MutableStoreOp currStoreOp,
                             DeviceAwareRewriter &rewriter) const override {
    auto loc = currStoreOp.getLoc();

    auto outChain = currStoreOp.getOutChain();

    if (outChain.use_empty())
      return rewriter.notifyMatchFailure(loc, "no proceeding store op");

    if (!outChain.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          loc, "store elision is supported only when the output chain has "
               "exactly one use");
    }

    auto nextStoreOp = dyn_cast<MutableStoreOp>(*outChain.user_begin());

    if (!nextStoreOp)
      return rewriter.notifyMatchFailure(
          loc, "consumer of output chain is not another store op");

    if (currStoreOp.getInBuffer() != nextStoreOp.getInBuffer())
      return rewriter.notifyMatchFailure(
          loc, "following store op does not operate on same mo.buffer");

    rewriter.replaceAllUsesWith(currStoreOp.getOutChain(),
                                currStoreOp.getInChain());

    rewriter.eraseOp(currStoreOp);

    return success();
  }
};

} // namespace

struct ElideRedundantMutableOps
    : public impl::ElideRedundantMutableOpsBase<ElideRedundantMutableOps> {
  using ElideRedundantMutableOpsBase::ElideRedundantMutableOpsBase;

  void runOnOperation() override {
    auto graphOp = getOperation();
    MLIRContext *ctx = graphOp->getContext();
    OpBuilder builder{ctx};

    RewritePatternSet patterns{ctx};

    patterns.add<EliminateRedundantLoads, EliminateRedundantStores>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace M::MO
