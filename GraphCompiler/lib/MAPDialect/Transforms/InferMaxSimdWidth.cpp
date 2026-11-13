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
// Sets each `map.iter.elementwise.body`'s and `map.iter.rowwise.body`'s
// `max_simd_width` from the ops in its body (see `MAP::MaxSimdWidthInterface`).
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_INFERMAXSIMDWIDTH
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// The GCD of the limit `body`'s ops need and the one it already has, which
// constrains its width too: running the pass again keeps a limit.
TypedAttr combinedLimit(Operation *body, TypedAttr existing, Region &region) {
  return GCDAttr::getSimplified(body->getContext(),
                                {existing, getMaxSimdWidth(region)});
}

struct InferMaxSimdWidthPass
    : public impl::InferMaxSimdWidthBase<InferMaxSimdWidthPass> {
  using Base::Base;

  void runOnOperation() override {
    // Both outlined bodies carry the limit of the ops in their region; the
    // kernel ops they come from (`IterOpInterface`) carry none.
    auto update = [](auto body) {
      body.setMaxSimdWidthAttr(
          combinedLimit(body, body.getMaxSimdWidthAttr(), body.getBody()));
    };
    getOperation().walk([&](Operation *op) {
      if (auto body = dyn_cast<IterElementwiseBodyOp>(op))
        update(body);
      else if (auto body = dyn_cast<IterRowwiseBodyOp>(op))
        update(body);
    });
  }
};

} // namespace

} // namespace M::MAP
