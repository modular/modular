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

#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-defer-allocs"

namespace M::MGP {

// Generate support code for DeferAllocsPass
#define GEN_PASS_DEF_DEFERALLOCSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

/// Defers buffer allocations until their writers are ready to execute.
class DeferAllocsPass : public impl::DeferAllocsPassBase<DeferAllocsPass> {
  void runOnOperation() override {
    ModelOp model = getOperation();
    ExecuteState execState(model.getContext(), model);
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(model);

    // Guards against revisiting allocs when an op writes multiple buffers.
    DenseSet<Operation *> movedAllocOps;

    auto defer = [&](Operation *op) {
      if (movedAllocOps.contains(op))
        return;

      auto allocOp = dyn_cast<MGPBufferAllocOpInterface>(op);
      if (!allocOp)
        return;

      OpOperand *mutator = getUniqueMutator(allocOp.getOutBuffer());
      if (!mutator)
        return;

      allocOp->moveBefore(mutator->getOwner());
      movedAllocOps.insert(allocOp);

      SmallVector<Value> additionalChainsToWaitOn;
      for (OpOperand *mutatorChainOpOperand :
           getInputChainOpOperands(mutator->getOwner())) {
        additionalChainsToWaitOn.push_back(mutatorChainOpOperand->get());
      }

      execState.builder.setInsertionPoint(allocOp);
      [[maybe_unused]] auto res = alsoWaitForChains(
          execState.builder, builderContext, allocOp, additionalChainsToWaitOn);
      ASSERT_STREAM(succeeded(res),
                    << "all allocation ops must accept an input "
                       "chain to control their timing");
    };

    model.getExecute().walk(defer);
    model.getInit().walk(defer);
  }
};

} // namespace
} // namespace M::MGP
