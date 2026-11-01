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
//
// Fuses MAP kernels by inlining producers into consumers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MOOps.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEKERNELS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseKernelsPass : public MAP::impl::FuseKernelsBase<FuseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    MO::GraphOp graphOp = getOperation();

    // One fusion round: input fusion before output fusion, so a
    // view/elementwise producer fuses into a foreach consumer before that
    // consumer is output-fused into an epilogue -- which is what lets a view
    // reach an epilogue without fusing directly into a closure. Prologue runs
    // after elementwise/view so every producer it sees is the maximal
    // elementwise/view chain.
    mlir::OpPassManager fusePm(MO::GraphOp::getOperationName());
    fusePm.addPass(createFuseElementwise());
    fusePm.addPass(createFuseViews());
    fusePm.addPass(createFusePrologue());
    fusePm.addPass(createFuseReductions());
    fusePm.addPass(createFuseEpilogue());
    fusePm.addPass(createFuseReshapeEpilogue());
    // Last, so siblings merge only once the producer-consumer fusers have
    // collapsed what they can this round.
    fusePm.addPass(createFuseHorizontal());

    // Between rounds, only CSE: the fusers dedup the kernels they fused into
    // and erase consumed producers, so a full canonicalization would be wasted
    // work.
    mlir::OpPassManager cleanupPm(MO::GraphOp::getOperationName());
    cleanupPm.addPass(mlir::createCSEPass());

    auto kernelCount = [&] {
      unsigned n = 0;
      graphOp.getBody().walk([&](IterOpInterface) {
        ++n;
        return WalkResult::skip();
      });
      return n;
    };

    // Repeat until a round fuses nothing, since folding one link of a chain can
    // expose the next. Every fusion erases a kernel, so a round that leaves the
    // kernel count unchanged fused nothing.
    unsigned fused;
    do {
      unsigned before = kernelCount();
      if (failed(runPipeline(fusePm, graphOp)))
        return signalPassFailure();
      fused = before - kernelCount();
      numKernelsFused += fused;
      if (failed(runPipeline(cleanupPm, graphOp)))
        return signalPassFailure();
    } while (fused > 0);
  }
};

} // namespace

} // namespace M::MAP
