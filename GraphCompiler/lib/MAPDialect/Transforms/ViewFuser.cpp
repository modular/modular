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
// Input-fusion wrapper for a view producer (a foreach whose output flows from a
// transformed load) fusing into a consumer foreach body or an existing
// `map.iter.prologue` capture of a `map.iter.opaque`. A pure elementwise
// producer is `ElementwiseFuser`'s job.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Support/InputFusion.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

#include "mlir/Support/LLVM.h"

using namespace M;

namespace M::MAP {

#define GEN_PASS_DEF_FUSEVIEWS
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

struct FuseViewsPass : public MAP::impl::FuseViewsBase<FuseViewsPass> {
  using Base::Base;

  void runOnOperation() override {
    InputFusionConfig config;
    // The consumer is a foreach or an opaque, the only kinds input fusion
    // offers.
    config.shouldFuse = [](StoreOp srcStore, LoadOp) {
      return producesThroughView(srcStore);
    };
    // A pure view is a cheap index computation, always fine to re-derive per
    // consumer. A view already fused into elementwise compute is not: cloning
    // it re-reads all of its inputs, and on a residual chain
    // (h = h + reshape(matmul(h))) that recomputes every earlier layer in each
    // later one, so it is held to the elementwise duplication rule
    // (GEX-4234).
    config.canDuplicate = [](IterElementwiseOp src) {
      return isPureViewForeach(src) || isCheapToDuplicate(src);
    };
    numKernelsFused = fuseInput(getOperation(), config);
  }
};

} // namespace

} // namespace M::MAP
