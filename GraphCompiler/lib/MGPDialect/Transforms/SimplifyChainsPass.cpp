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
#include "GraphCompiler/MGPDialect/Transforms/AsyncDataflowDependencies.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "Support/Compiler/Threading.h"

#include <queue>

using namespace mlir;

#define DEBUG_TYPE "mgp-simplify-chains-pass"

namespace M::MGP {

// Generate support code for SimplifyChainsPass
#define GEN_PASS_DEF_SIMPLIFYCHAINSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// SimplifyChainsPass
//===----------------------------------------------------------------------===//

class SimplifyChainsPass
    : public impl::SimplifyChainsPassBase<SimplifyChainsPass> {
  using SimplifyChainsPassBase::SimplifyChainsPassBase;

  static ValueRange getGatingOperands(Operation *op) {
    if (!op)
      return ValueRange{};

    if (auto cfOp = dyn_cast<GRT::GRTControlFlowInterface>(op))
      // Only the extra strict operands are gating for control flow ops.
      return cfOp.getStrictExtraOperands();

    if (isa<MEF::NonStrictInterface>(op))
      return ValueRange{};
    return op->getOperands();
  }

  static SmallVector<Value> accumulatePrimitiveChains(Value chain) {
    SmallVector<Value> result;

    std::queue<Value> stack;
    stack.push(chain);

    llvm::SmallDenseSet<Value> visited;

    // Look through grt.chain.create ops.
    while (!stack.empty()) {
      Value top = stack.front();
      stack.pop();

      if (!visited.insert(top).second)
        continue;

      if (auto chain = top.getDefiningOp<GRT::ChainCreateOp>()) {
        for (auto input : chain.getInputChains())
          stack.push(input);

        continue;
      }

      result.push_back(top);
    }

    return result;
  }

  /// Eliminate inputs to `grt.chain.create` operations that are implied ready
  /// by other inputs. Given a chain create op:
  ///
  /// %X = grt.chain.create(%0, %1, %2)
  ///
  /// this walks the transitive gating inputs of the producers of %0, %1, and %2
  /// and deletes any of %0, %1, %2 reachable from another.
  SmallVector<Value, 16>
  eliminateImpliedByOperands(GRT::ChainCreateOp chainCreate) {
    // Needles still in the set at the end were not found.
    llvm::SmallSetVector<Value, 16> needles;
    for (auto arg : chainCreate.getInputChains()) {
      for (auto needle : accumulatePrimitiveChains(arg))
        needles.insert(needle);
    }

    llvm::SmallDenseSet<Value> visited;
    llvm::SetVector<Value> currentGeneration;
    llvm::SetVector<Value> newGeneration;

    for (auto needle : needles)
      currentGeneration.insert_range(getGatingOperands(needle.getDefiningOp()));

    while (!currentGeneration.empty()) {
      visited.insert_range(currentGeneration);
      needles.remove_if([&](Value v) { return currentGeneration.contains(v); });

      newGeneration.clear();

      for (auto val : currentGeneration) {
        for (auto next : getGatingOperands(val.getDefiningOp())) {
          if (!visited.contains(next))
            newGeneration.insert(next);
        }
      }

      std::swap(newGeneration, currentGeneration);
    }

    return needles.takeVector();
  }

  void runOnOperation() override {
    ModelOp modelOp = getOperation();

    SmallVector<GRT::ChainCreateOp> chainCreates;

    modelOp.walk([&](GRT::ChainCreateOp op) { chainCreates.push_back(op); });

    using CacheT =
        SmallVector<std::pair<GRT::ChainCreateOp, SmallVector<Value>>>;

    auto consolidate = [](CacheT &source, MutableArrayRef<CacheT> others) {
      for (CacheT &other : others)
        source.append(std::move(other));
    };

    auto func = [&](CacheT &cache, GRT::ChainCreateOp op) {
      cache.emplace_back(op, eliminateImpliedByOperands(op));
      return success();
    };

    CacheT cache;
    cache.reserve(chainCreates.size());

    [[maybe_unused]] auto result = failableParallelForEach(
        &getContext(), chainCreates, func, cache, consolidate);

    ASSERT_STREAM(succeeded(result), "processing chain create ops failed");

    for (auto &[chainOp, values] : cache)
      chainOp.setInputChains(values);

    RewritePatternSet patterns(&getContext());
    GRT::ChainCreateOp::getCanonicalizationPatterns(patterns, &getContext());

    auto asOperations = llvm::to_vector_of<Operation *>(chainCreates);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    (void)applyOpPatternsGreedily(asOperations, std::move(patterns), config);
  }
};

} // namespace

} // namespace M::MGP
