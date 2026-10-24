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

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "Support/AssertStream.h"
#include "Support/Compiler/Threading.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-simplify-device-graph-dependencies"

namespace M::MGP {

#define GEN_PASS_DEF_SIMPLIFYDEVICEGRAPHDEPENDENCIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// SimplifyDeviceGraphDependenciesPass
//===----------------------------------------------------------------------===//

class SimplifyDeviceGraphDependenciesPass
    : public impl::SimplifyDeviceGraphDependenciesPassBase<
          SimplifyDeviceGraphDependenciesPass> {
  using Base::Base;

  /// Returns the edges of the dependencies op producing `node`, else empty.
  static ValueRange getEdges(Value node) {
    if (auto producer = node.getDefiningOp<DeviceGraphDependenciesOp>())
      return producer.getDependencies();
    return ValueRange{};
  }

  /// Eliminates edges of `dependenciesOp` implied transitively by its other
  /// edges. A node gates on all of its op's edges (even for an empty body), so
  /// an edge reachable from another is redundant. Works like
  /// SimplifyChainsPass::eliminateImpliedByOperands.
  static SmallVector<Value>
  eliminateImpliedDependencies(DeviceGraphDependenciesOp dependenciesOp) {
    llvm::SmallSetVector<Value, 16> needles;
    needles.insert_range(dependenciesOp.getDependencies());

    llvm::SmallDenseSet<Value> visited;
    llvm::SetVector<Value> currentGeneration;
    llvm::SetVector<Value> newGeneration;

    for (auto needle : needles)
      currentGeneration.insert_range(getEdges(needle));

    // The staged edge relation is a DAG, but track visited values anyway so a
    // malformed cycle cannot hang the pass.
    while (!currentGeneration.empty()) {
      visited.insert_range(currentGeneration);

      needles.remove_if([&](Value v) { return currentGeneration.contains(v); });

      newGeneration.clear();

      for (auto val : currentGeneration) {
        for (auto next : getEdges(val)) {
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

    SmallVector<DeviceGraphDependenciesOp> dependenciesOps;

    modelOp.walk<mlir::WalkOrder::PreOrder>([&](DeviceGraphDependenciesOp op) {
      dependenciesOps.push_back(op);
      // Dependencies ops do not nest.
      return WalkResult::skip();
    });

    using CacheT =
        SmallVector<std::pair<DeviceGraphDependenciesOp, SmallVector<Value>>>;

    auto consolidate = [](CacheT &source, MutableArrayRef<CacheT> others) {
      for (CacheT &other : others) {
        for (auto &entry : other)
          source.push_back(std::move(entry));

        other = {};
      }
    };

    auto func = [&](CacheT &cache, DeviceGraphDependenciesOp op) {
      cache.emplace_back(op, eliminateImpliedDependencies(op));
      return success();
    };

    CacheT cache;
    cache.reserve(dependenciesOps.size());

    LogicalResult result = failableParallelForEach(
        &getContext(), dependenciesOps, func, cache, consolidate);

    if (failed(result)) {
      modelOp.emitError("failed processing dependencies");
      signalPassFailure();
    }

    for (auto &[dependenciesOp, edges] : cache) {
      numEliminatedDependencies +=
          dependenciesOp.getDependencies().size() - edges.size();
      dependenciesOp.getDependenciesMutable().assign(edges);
    }
  }
};

} // namespace

} // namespace M::MGP
