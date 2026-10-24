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

#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"

#include "MLRT/GraphRT/GRTDialect/GRTOps.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_SIMPLIFYEXTENDLIFETIMESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct SimplifyExtendLifetimesPass
    : public MGP::impl::SimplifyExtendLifetimesPassBase<
          SimplifyExtendLifetimesPass> {
  using SimplifyExtendLifetimesPassBase::SimplifyExtendLifetimesPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](MGP::ExtendLifetimeOp extend) {
      TypedValue<MEF::ChainType> chain = extend.getGate();
      MOGG::AsyncRoutineOp kernel = chain.getDefiningOp<MOGG::AsyncRoutineOp>();
      if (!kernel)
        return;

      kernel->insertOperands(kernel->getNumOperands(), extend.getInputs());
      for (Value v : extend.getInputs()) {
        // Body block arguments are `!mogg.async<T>` wrappers around operands.
        kernel.getBody().addArgument(MOGG::AsyncType::get(v.getType()),
                                     v.getLoc());
        kernel.setInOutSignatureAttr(kernel.getInOutSignature().append(
            {InOutSignatureAttr::InOutSemantics::kNone}));
      }

      extend.getResult().replaceAllUsesWith(chain);
      extend.erase();
    });
  }
};

} // namespace

} // namespace M::MGP
