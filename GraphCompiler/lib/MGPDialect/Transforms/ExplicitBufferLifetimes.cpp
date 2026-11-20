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

#include "GraphCompiler/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "Support/AssertStream.h"

#define DEBUG_TYPE "mgp-explicit-buffer-lifetimes"

using namespace mlir;

namespace M::MGP {

// Generate support code for ExplicitBufferLifetimesPass
#define GEN_PASS_DEF_EXPLICITBUFFERLIFETIMESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

class ExplicitBufferLifetimesPass
    : public impl::ExplicitBufferLifetimesPassBase<
          ExplicitBufferLifetimesPass> {
  using ExplicitBufferLifetimesPassBase::ExplicitBufferLifetimesPassBase;

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    int numAllocs = 0;
    int numFreesAdded = 0;

    auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();
    markAnalysesPreserved<BufferLifetimeAnalysisPass>();

    modelOp->walk([&](Block *block) {
      // Free each alloc at an op guaranteed to run after all of its uses.
      for (Operation &op : block->getOperations()) {
        auto allocOp = dyn_cast<MGP::BufferAllocOp>(op);
        if (!allocOp)
          continue;
        ++numAllocs;

        auto allocUses = getAllocUses(allocOp);
        if (allocUses.isError())
          continue;
        auto lcaNode = analysis.getDepTree(block).findLCA(*allocUses);

        // The free goes after the nearest ancestor with a chain output.
        while (lcaNode) {
          bool hasMGPChainOutput = false;
          for (auto result : lcaNode->op->getResults())
            if (isa<MEF::ChainType>(result.getType()))
              hasMGPChainOutput = true;
          if (hasMGPChainOutput)
            break;
          lcaNode = lcaNode->parent;
        }

        if (!lcaNode) {
          LLVM_DEBUG(llvm::dbgs()
                     << "could not find LCA for " << allocOp << "\n");
          continue;
        }

        auto lca = lcaNode->op;

        OpBuilder builder(lca);
        auto loc = lca->getLoc();

        OpResult chainOutput;
        for (auto result : lca->getResults()) {
          if (isa<MEF::ChainType>(result.getType())) {
            chainOutput = result;
            break;
          }
        }
        ASSERT_STREAM(chainOutput, "op should have chain output");

        auto bufferFree = MGP::BufferFreeOp::create(builder, loc, chainOutput,
                                                    allocOp.getOutBuffer());

        chainOutput.replaceAllUsesExcept(bufferFree, bufferFree);
        ++numFreesAdded;
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "added frees for " << numFreesAdded << " out of "
                            << numAllocs << " allocs\n");
  }
};
} // namespace M::MGP
