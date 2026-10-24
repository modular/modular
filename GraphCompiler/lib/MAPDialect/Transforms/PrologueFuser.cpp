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
// Input-fusion wrapper for a sole-use `map.iter.elementwise` producer fusing
// into a `map.iter.opaque` consumer's existing prologue, so e.g. `gather(add(a,
// b), indices)` becomes one kernel whose data-operand prologue computes the add
// inline. A view producer is `ViewFuser`'s to claim first.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEPROLOGUE
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseProloguePass : public MAP::impl::FusePrologueBase<FuseProloguePass> {
  using Base::Base;
  FuseProloguePass(const FuseProloguePass &other) : Base(other) {}

  void runOnOperation() override {
    InputFusionConfig config;
    // A foreach producer fusing into an opaque (the load lives in the opaque's
    // prologue; a foreach-body load would have no opaque parent here).
    config.shouldFuse = [](StoreOp srcStore, LoadOp dstLoad) {
      return srcStore->getParentOfType<IterElementwiseOp>() &&
             dstLoad->getParentOfType<IterOpaqueOp>();
    };
    // Only a sole-use producer fuses into a prologue.
    config.canDuplicate = [](IterElementwiseOp) { return false; };
    numKernelsFused = fuseInput(getOperation(), config);
  }

  Statistic numKernelsFused{this, "num-kernels-fused",
                            "Number of kernels fused"};
};

} // namespace

} // namespace M::MAP
