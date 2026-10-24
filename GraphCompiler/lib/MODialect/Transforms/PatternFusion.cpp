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
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Patterns/Fusion.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#if defined(__APPLE__)
#include <pthread.h>
#endif

using namespace M;
using namespace M::MO;

namespace M::MOGG {

#define GEN_PASS_DEF_PATTERNFUSION
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

struct PatternFusionPass : public impl::PatternFusionBase<PatternFusionPass> {
  using PatternFusionBase::PatternFusionBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    M::DeviceRefAttr targetDeviceRef =
        targetDevice == "" ? nullptr
                           : M::DeviceRefAttr::get(ctx, targetDevice.getValue(),
                                                   targetDeviceId.getValue());

    llvm::SmallVector<MO::AllReduceSumOp> allReduceOps;
    getOperation()->walk([&](MO::AllReduceSumOp allReduceSumOp) {
      allReduceOps.push_back(allReduceSumOp);
    });
    for (auto allReduceOp : allReduceOps)
      MO::applyFuseAllReduceAddRMSNormQuantFP8(allReduceOp, targetDeviceRef);

    // Parallel-region analog: fuse `mo.parallel { mo.bundled.allreduce.sum }`
    // and the per-device add + rms_norm + quantize_fp8 chains after it.
    llvm::SmallVector<MO::BundledAllReduceSumOp> bundledAllReduceOps;
    getOperation()->walk([&](MO::BundledAllReduceSumOp op) {
      bundledAllReduceOps.push_back(op);
    });
    for (auto bundledAllReduceOp : bundledAllReduceOps) {
      // An earlier rewrite may have erased this op's parent parallel.
      if (!bundledAllReduceOp.getOperation()->getBlock())
        continue;
      MO::applyFuseParallelBundledAllReduceAddRMSNormQuantFP8(
          bundledAllReduceOp);
    }

    MO::GraphOp graphOp = getOperation();

    // RoPE fusion matches splits built from adjacent slice pairs.
    {
      RewritePatternSet sliceToSplitPatterns{ctx};
      sliceToSplitPatterns.add<SliceToSplitPattern</*AllowDynamicStop=*/true>>(
          ctx);
      if (failed(
              applyPatternsGreedily(graphOp, std::move(sliceToSplitPatterns))))
        return signalPassFailure();
    }

    MO::PackParamCache cache;
    MO::MHAOptions mhaOpts{
        enableMHAPattern.getValue(),
        mhaAllowUnsafeBCast.getValue(),
        mhaAllowInternalReshape.getValue(),
        debugMHAPattern.getValue(),
    };

    RewritePatternSet patterns{ctx};
    MO::populateFusionPatterns(patterns, cache, targetDeviceRef, mhaOpts);

    if (failed(applyPatternsGreedily(graphOp, std::move(patterns))))
      return signalPassFailure();

    // Lower splits that survived fusion back to slices.
    {
      MO::DeviceInfoMapping deviceInfoMapping(
          graphOp->getParentOfType<mlir::ModuleOp>());
      RewritePatternSet splitToSlicePatterns{ctx};
      splitToSlicePatterns.add<SplitToSlicePattern<>>(ctx);
      MO::populateFusionMatmulAddPatterns(deviceInfoMapping,
                                          splitToSlicePatterns);
      // The mega_ffn_nvfp4 kernel is MAX-only, so OSS builds (where it is
      // unregistered) disable this; the pattern arch-gates SM100 itself.
      if (enableMegaFFNPattern.getValue())
        MO::populateFusionMegaFFNPatterns(deviceInfoMapping,
                                          splitToSlicePatterns);
      if (failed(
              applyPatternsGreedily(graphOp, std::move(splitToSlicePatterns))))
        return signalPassFailure();
    }

    MO::synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace

} // namespace M::MOGG
