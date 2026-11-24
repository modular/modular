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

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"

namespace M::MGP {

#define GEN_PASS_DEF_FIXUPOVERHANGSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class FixupOverhangsPass
    : public impl::FixupOverhangsPassBase<FixupOverhangsPass> {
  using FixupOverhangsPassBase::FixupOverhangsPassBase;

  void runOnOperation() override {
    // Make each block's terminator depend on its unused chains so they are
    // awaited.
    ModelOp modelOp = getOperation();
    BufferBuilderContext builderCtx =
        TensorBufferDialectInterface::getAllBufferBuilders(modelOp);

    modelOp->walk([&](Block *block) {
      auto outputOp = block->getTerminator();

      SmallVector<Value> hangingChains;

      for (Operation &op : block->getOperations()) {
        for (auto result : op.getResults()) {
          // A failing op sets all its results to error, so one per op
          // suffices.
          if (result.use_empty()) {
            hangingChains.push_back(result);
            break;
          }
        }
      }

      if (hangingChains.empty())
        return;

      OpBuilder builder(outputOp);
      if (failed(
              alsoWaitForChains(builder, builderCtx, outputOp, hangingChains)))
        outputOp->emitError("Could not wait on hanging chains");
    });
  }
};

} // namespace
} // namespace M::MGP
