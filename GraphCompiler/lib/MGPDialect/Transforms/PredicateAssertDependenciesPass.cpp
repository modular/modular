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

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetOperations.h"

#define EXCLUDED_TYPES MGP::ContextType, MEF::MEFChainTypeInterface

using namespace mlir;

namespace M::MGP {

// Generate support code for PredicateAssertDependenciesPass.
#define GEN_PASS_DEF_PREDICATEASSERTDEPENDENCIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PredicateAssertDependenciesPass
//===----------------------------------------------------------------------===//

/// Makes every chain-taking op that uses a value feeding an assert depend on
/// that assert's chain.
class PredicateAssertDependenciesPass
    : public impl::PredicateAssertDependenciesPassBase<
          PredicateAssertDependenciesPass> {
  using PredicateAssertDependenciesPassBase::
      PredicateAssertDependenciesPassBase;

  void runOnOperation() override {
    auto &ctx = getContext();
    OpBuilder builder(&ctx);

    // TODO(#19098): The rewrite is per block, so dependencies across blocks
    // are not tracked and the asserts may not prevent errors there.
    ModelOp model = getOperation();
    BufferBuilderContext builderCtx =
        TensorBufferDialectInterface::getAllBufferBuilders(model);

    auto perBlockRewrite = [&](Block *block) {
      // TODO(#19098): This map is valid within a block.
      llvm::MapVector<Value, llvm::SetVector<AssertOp>> valuesThatFeedAsserts;
      for (auto assertOp : block->getOps<AssertOp>()) {
        DenseSet<Operation *> visited;
        std::function<void(Value)> propagateUp = [&](Value val) {
          Operation *op = val.getDefiningOp();

          if (!op) {
            valuesThatFeedAsserts[val].insert(assertOp);
            return;
          }

          if (!visited.insert(op).second)
            return;

          // All results count, not just the one feeding the assert.
          for (auto result : op->getResults())
            valuesThatFeedAsserts[result].insert(assertOp);

          for (auto operand : op->getOperands())
            propagateUp(operand);
        };

        for (auto operand : assertOp->getOperands())
          propagateUp(operand);
      }

      llvm::MapVector<Operation *, llvm::SetVector<AssertOp>>
          operationsThatDependOnAsserts;
      const auto storeAssertionDependenciesForOp =
          [&](Operation *op, const SetVector<AssertOp> &asserts) {
            auto &assertsItDependsOn = operationsThatDependOnAsserts[op];
            uint64_t before = assertsItDependsOn.size();
            assertsItDependsOn.insert(asserts.begin(), asserts.end());

            // Already visited with these asserts.
            if (assertsItDependsOn.size() == before)
              return false;

            // An op that takes a chain is predicated here, and its
            // descendants inherit that, so stop recursing. Visit order is
            // arbitrary, so chains may be redundant; SimplifyChains cleans up.
            auto inputChains = getInputChainOpOperands(op);
            if (!inputChains.empty())
              return false;

            return true;
          };

      // TODO(#19098): Propagate the assert dependencies on all users of the
      // values that contributed to the assertions in the current block.
      const std::function<void(Value, llvm::SetVector<AssertOp>)>
          propagateDown = [&](Value val, const SetVector<AssertOp> &asserts) {
            auto op = val.getDefiningOp();
            if (op && !storeAssertionDependenciesForOp(op, asserts))
              return;

            // These types would create false dependencies.
            if (isa<EXCLUDED_TYPES>(val.getType()))
              return;

            for (auto user : val.getUsers()) {
              if (!user->getNumResults())
                storeAssertionDependenciesForOp(user, asserts);
              else {
                for (auto result : user->getResults())
                  propagateDown(result, asserts);
              }
            }
          };

      for (auto &[val, asserts] : valuesThatFeedAsserts)
        propagateDown(val, asserts);

      for (auto &[op, asserts] : operationsThatDependOnAsserts) {
        auto inputChains = getInputChainOpOperands(op);
        if (inputChains.empty())
          continue;

        llvm::SetVector<AssertOp> diff = asserts;
        if (op->getNumResults()) {
          auto it = valuesThatFeedAsserts.find(op->getResult(0));
          if (it != valuesThatFeedAsserts.end())
            diff = llvm::set_difference(it->second, asserts);
        }

        // assertsYouDependOn - assertsYouContributedTo
        if (diff.empty())
          continue;

        // For readability.
        builder.setInsertionPoint(op);
        SmallVector<Value> chains;
        for (AssertOp assert : diff)
          chains.emplace_back(assert.getResult());
        if (failed(alsoWaitForChains(builder, builderCtx, op, chains)))
          op->emitError("Could not wait on additional chains");
      }
    };

    model->walk(perBlockRewrite);
  }
};

} // namespace
} // namespace M::MGP
