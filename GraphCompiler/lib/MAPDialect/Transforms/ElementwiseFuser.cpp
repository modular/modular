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
// Input-fusion wrapper for a pure-elementwise producer fusing into a
// pure-elementwise consumer foreach -- the case that always shares one index
// space. A view on either side is `ViewFuser`'s job.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEELEMENTWISE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// A tensor with more static elements than the threshold (or an unknown element
// count) is "large": re-reading it is expensive (JSC-30 / MSTDL-2759).
bool isLargeTensorOperand(Type type, int64_t largeTensorElementThreshold) {
  auto tensorType = dyn_cast<MO::TensorType>(type);
  if (!tensorType)
    return false;
  std::optional<int64_t> numElements = tensorType.getNumElements();
  return !numElements.has_value() || *numElements > largeTensorElementThreshold;
}

// Cheap to duplicate iff at most one input is a large tensor: duplication
// re-reads all inputs once per extra consumer, a win only when at most one
// re-read is expensive.
bool isCheapToDuplicate(IterElementwiseOp op,
                        int64_t largeTensorElementThreshold) {
  int numLargeOperands = 0;
  for (Value input : op.getInputs())
    if (isLargeTensorOperand(input.getType(), largeTensorElementThreshold))
      if (++numLargeOperands > 1)
        return false;
  return true;
}

struct FuseElementwisePass
    : public MAP::impl::FuseElementwiseBase<FuseElementwisePass> {
  using Base::Base;
  FuseElementwisePass(const FuseElementwisePass &other) : Base(other) {}

  void runOnOperation() override {
    InputFusionConfig config;
    // Both kernels pure elementwise (no index transform).
    config.shouldFuse = [](StoreOp srcStore, LoadOp dstLoad) {
      auto src = srcStore->getParentOfType<IterElementwiseOp>();
      auto dst = dstLoad->getParentOfType<IterElementwiseOp>();
      return src && dst && !isViewForeach(src) && !isViewForeach(dst);
    };
    // Never duplicate a host kernel.
    int64_t threshold = largeTensorElementThreshold;
    config.canDuplicate = [threshold](IterElementwiseOp src) {
      return !MO::isOnHost(src) && isCheapToDuplicate(src, threshold);
    };
    config.maxFusionSetSize = maxFusionSetSize;
    numKernelsFused = fuseInput(getOperation(), config);
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
