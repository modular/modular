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
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
// This file holds their folding and canonicalization.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

using namespace M;
using namespace MGP;

LogicalResult DeviceGraphDependenciesOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  llvm::SmallSetVector<Value, 8> deduped;
  deduped.insert_range(getDependencies());

  if (deduped.size() == getDependencies().size())
    return failure();

  getDependenciesMutable().assign(deduped.getArrayRef());
  return success();
}
namespace {

/// Rebuilds an `mgp.device_graph.dependencies` op with new body results and
/// yield operands, keeping the `node` result and dependency edges.
/// `bodyResultReplacement(i)` maps original body result `i` to its position
/// among the new body results.
DeviceGraphDependenciesOp
rebuildWithBodyResults(PatternRewriter &rewriter, DeviceGraphDependenciesOp op,
                       TypeRange newBodyResultTypes,
                       ValueRange newYieldOperands,
                       ArrayRef<unsigned> bodyResultRemap) {
  SmallVector<Type> newResultTypes;
  newResultTypes.push_back(op.getNode().getType());
  llvm::append_range(newResultTypes, newBodyResultTypes);

  auto newOp = DeviceGraphDependenciesOp::create(
      rewriter, op.getLoc(), TypeRange(newResultTypes), op.getGraphBuilder(),
      op.getDependencies());
  newOp.getBody().takeBody(op.getBody());
  DeviceGraphDependenciesYieldOp newYield = newOp.getYieldOp();
  rewriter.modifyOpInPlace(newYield, [&]() {
    newYield.getOperandsMutable().assign(newYieldOperands);
  });

  SmallVector<Value> replacements(op.getNumResults());
  replacements[0] = newOp.getNode();
  for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
    if (result.use_empty())
      continue;
    replacements[idx + 1] = newOp.getResult(bodyResultRemap[idx] + 1);
  }
  rewriter.replaceOp(op, replacements);
  return newOp;
}

/// Drops unused body results of an `mgp.device_graph.dependencies` op and the
/// matching yield operands. The `node` result (the ordering edge) is kept.
struct RemoveUnusedDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getBodyResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == op.getBodyResults().size())
      return failure();

    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    SmallVector<unsigned> remap(op.getBodyResults().size());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      remap[oldIdx] = newIdx;

    rebuildWithBodyResults(rewriter, op, TypeRange(usedResultTypes),
                           newYieldOperands, remap);
    return success();
  }
};

/// De-duplicates body results of an `mgp.device_graph.dependencies` op whose
/// yield carries the same value more than once.
struct DeduplicateDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphDependenciesYieldOp oldYield = op.getYieldOp();

    unsigned numElements = oldYield.getNumOperands();

    llvm::SmallMapVector<Value, unsigned, 8> firstOccurrence;
    firstOccurrence.reserve(numElements);

    SmallVector<unsigned> resultRemap(op.getBodyResults().size());

    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, _] =
          firstOccurrence.try_emplace(operand, firstOccurrence.size());
      resultRemap[idx] = it->second;
    }

    if (firstOccurrence.size() == op.getBodyResults().size())
      return failure();

    auto dedupedOperands = llvm::to_vector(firstOccurrence.keys());
    rebuildWithBodyResults(rewriter, op, TypeRange(dedupedOperands),
                           dedupedOperands, resultRemap);
    return success();
  }
};

/// Forwards `mgp.device_graph.dependencies` body results whose
/// `dependencies.yield` operand is produced *outside* the body region.
/// Rewriting those uses leaves the result dead for
/// `RemoveUnusedDependenciesResults` to drop.
struct ForwardExternalDependenciesResults
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    Region &body = op.getBody();
    DeviceGraphDependenciesYieldOp yieldOp = op.getYieldOp();

    bool changed = false;
    for (auto [result, yielded] :
         llvm::zip(op.getBodyResults(), yieldOp.getOperands())) {
      // Skipping dead results keeps the pattern idempotent; otherwise the
      // greedy driver never converges.
      if (result.use_empty())
        continue;

      // Only forward pass-throughs of outer SSA (block args count as inside).
      if (body.isAncestor(yielded.getParentRegion()))
        continue;

      rewriter.replaceAllUsesWith(result, yielded);
      changed = true;
    }
    return success(changed);
  }
};

/// Hoists body ops whose operands are all defined outside the body in front of
/// an `mgp.device_graph.dependencies` op, so the `work` closure captures them
/// once instead of recomputing them on every graph build. Hoisting is
/// transitive within one sweep; builder-dependent ops stay in the body.
struct HoistOperations final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    Region &body = dependenciesOp.getBody();

    auto isRegionCapture = [&body](Value operand) {
      return body.isAncestor(operand.getParentRegion());
    };

    bool changed = false;
    for (Operation &op : llvm::make_early_inc_range(body.getOps())) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      WalkResult result = op.walk([&](Operation *inner) -> WalkResult {
        bool canHoist = llvm::none_of(inner->getOperands(), isRegionCapture);

        if (!canHoist)
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

      if (!result.wasInterrupted()) {
        rewriter.moveOpBefore(&op, dependenciesOp);
        changed = true;
      }
    }

    return success(changed);
  }
};

/// Absorbs the body of an `mgp.device_graph.dependencies` op into its sole
/// successor, when the two form a link in a linear chain.
struct MergeChainedDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp op,
                                PatternRewriter &rewriter) const override {
    // With exactly one edge, absorbing the producer adds no ordering. With
    // more, the absorbed body would be ordered after the other edges too.
    if (op.getDependencies().size() != 1)
      return failure();

    auto producer =
        op.getDependencies().front().getDefiningOp<DeviceGraphDependenciesOp>();
    if (!producer)
      return failure();

    if (!producer.getNode().hasOneUse())
      return failure();

    // Dataflow out of the producer is uncommon; bail.
    if (!producer.getBodyResults().empty())
      return failure();

    if (producer.getGraphBuilder() != op.getGraphBuilder() ||
        producer->getBlock() != op->getBlock())
      return failure();

    // The merged block keeps the consumer's terminator.
    rewriter.eraseOp(producer.getYieldOp());

    rewriter.inlineBlockBefore(&producer.getBodyBlock(), &op.getBodyBlock(),
                               op.getBodyBlock().begin(),
                               {op.getBuilderBlockArgument()});

    rewriter.modifyOpInPlace(op, [&] {
      op.getDependenciesMutable().assign(producer.getDependencies());
    });
    rewriter.eraseOp(producer);

    return success();
  }
};

/// Replaces a dependency edge on an empty (node-less) producer with that
/// producer's own edges, preserving transitive ordering. Bypassed ops are left
/// for DCE; duplicate edges are folded by the edge-dedup folder.
struct BypassEmptyDependencies final
    : public OpRewritePattern<DeviceGraphDependenciesOp> {
  using Base::Base;

  /// Returns the transitive dependencies of a node for an empty region.
  static std::optional<ValueRange> getTransitiveDependencies(Value dependency) {
    if (auto producer = dependency.getDefiningOp<DeviceGraphDependenciesOp>())
      if (!producer.producesNewNodes())
        return producer.getDependencies();

    return {};
  }

  LogicalResult matchAndRewrite(DeviceGraphDependenciesOp dependenciesOp,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> newDependencies;

    bool changed = false;
    for (auto dependency : dependenciesOp.getDependencies()) {
      if (auto transitiveDeps = getTransitiveDependencies(dependency)) {
        llvm::append_range(newDependencies, *transitiveDeps);
        changed = true;
      } else {
        newDependencies.push_back(dependency);
      }
    }

    if (!changed)
      return rewriter.notifyMatchFailure(dependenciesOp,
                                         "no dependencies to remove");

    rewriter.modifyOpInPlace(dependenciesOp, [&] {
      dependenciesOp.getDependenciesMutable().assign(newDependencies);
    });

    return success();
  }
};
} // namespace

void DeviceGraphDependenciesOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.add<
    BypassEmptyDependencies,
    DeduplicateDependenciesResults,
    ForwardExternalDependenciesResults,
    HoistOperations,
    MergeChainedDependencies,
    RemoveUnusedDependenciesResults
  >(ctx);
  // clang-format on
}
namespace {

/// Drops unused results of an `mgp.device_graph.staging` op and the matching
/// yield operands.
struct RemoveUnusedStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> usedResultIndices;
    SmallVector<Type> usedResultTypes;
    for (auto [idx, result] : llvm::enumerate(op.getResults())) {
      if (!result.use_empty()) {
        usedResultIndices.push_back(idx);
        usedResultTypes.push_back(result.getType());
      }
    }

    if (usedResultIndices.size() == op.getNumResults())
      return failure();

    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();
    auto newYieldOperands =
        llvm::map_to_vector(usedResultIndices, [&](unsigned idx) -> Value {
          return oldYield.getOperand(idx);
        });

    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(usedResultTypes), op.getInputs(),
        op.getConstantResources());

    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(newYieldOperands);
    });

    SmallVector<Value> replacements(op.getNumResults());
    for (auto [newIdx, oldIdx] : llvm::enumerate(usedResultIndices))
      replacements[oldIdx] = newOp.getResult(newIdx);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

/// De-duplicates results of an `mgp.device_graph.staging` op whose yield
/// carries the same value more than once.
struct DeduplicateStagingResults
    : public OpRewritePattern<DeviceGraphStagingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeviceGraphStagingOp op,
                                PatternRewriter &rewriter) const override {
    DeviceGraphStagingYieldOp oldYield = op.getYieldOp();

    llvm::SmallDenseMap<Value, unsigned> firstOccurrence;
    SmallVector<Value> dedupedOperands;
    SmallVector<Type> dedupedTypes;
    SmallVector<unsigned> resultRemap(op.getNumResults());
    for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
      auto [it, inserted] =
          firstOccurrence.try_emplace(operand, dedupedOperands.size());
      if (inserted) {
        dedupedOperands.push_back(operand);
        dedupedTypes.push_back(op.getResult(idx).getType());
      }
      resultRemap[idx] = it->second;
    }

    if (dedupedOperands.size() == op.getNumResults())
      return failure();

    auto newOp = DeviceGraphStagingOp::create(
        rewriter, op.getLoc(), TypeRange(dedupedTypes), op.getInputs(),
        op.getConstantResources());

    newOp.getBody().takeBody(op.getBody());
    DeviceGraphStagingYieldOp newYield = newOp.getYieldOp();
    rewriter.modifyOpInPlace(newYield, [&]() {
      newYield.getOperandsMutable().assign(dedupedOperands);
    });

    SmallVector<Value> replacements =
        llvm::map_to_vector(resultRemap, [&](unsigned newIdx) -> Value {
          return newOp.getResult(newIdx);
        });
    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

void DeviceGraphStagingOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RemoveUnusedStagingResults, DeduplicateStagingResults>(ctx);
}
