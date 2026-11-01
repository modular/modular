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

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Support/AssertStream.h"

#include "mlir/Pass/AnalysisManager.h"

namespace M::MO {

ParametricTopologyAnalysis::ParametricTopologyAnalysis(mlir::Operation *root) {
  // If from is in another block, every parent of to up to from's block also
  // uses from. doAtLeastOnce forces the from -> to edge, since parametric
  // dependencies never appear in the SSA graph.
  const auto applyDependencyToParentOps = [&](Operation *from, Operation *to,
                                              bool doAtLeastOnce) {
    // Parameters don't obey region dominance like SSA values. An operation can
    // use a parameter defined in a child region if that child region is not
    // held by a DeclInterface operation.
    if (to->getParentRegion()->isProperAncestor(from->getParentRegion()))
      return;

    const auto applyDependency = [&]() {
      doAtLeastOnce = true;
      metadata[from].parametricUsers.insert(to);
      metadata[to].parametricProviders.insert(from);
      to = to->getParentOp();
    };

    while (to && from->getBlock() != to->getBlock())
      applyDependency();

    // Skip the same-block edge otherwise; it would duplicate the SSA graph.
    if (to && doAtLeastOnce)
      applyDependency();
  };

  auto paramCache = KGEN::ParameterCollector::Analysis(root);
  for (auto &region : root->getRegions()) {
    KGEN::ParameterUseDefGraph useDefGraph(region);
    useDefGraph.calculate(paramCache);

    // DeclInterface ops don't appear in the root useDefGraph, so walk nested
    // graphs too.
    llvm::SmallVector<KGEN::ParameterUseDefGraph *> workList = {&useDefGraph};
    while (!workList.empty()) {
      auto &nextUseDefGraph = *workList.pop_back_val();
      for (auto &[op, references] : nextUseDefGraph.opUses) {
        for (auto reference : references) {
          auto whereItIsDeclaredIt =
              nextUseDefGraph.decls.find(reference.getName());
          if (whereItIsDeclaredIt == nextUseDefGraph.decls.end())
            continue;
          auto whereItIsDeclared = whereItIsDeclaredIt->second;
          applyDependencyToParentOps(whereItIsDeclared.declOp, op, true);
        }
      }

      for (auto &[_, nestedUseDefGraph] : nextUseDefGraph.nestedScopes)
        workList.push_back(&nestedUseDefGraph);
    }
  }

  root->walk([&](Operation *op) {
    for (auto user : op->getUsers())
      applyDependencyToParentOps(op, user, false);
  });
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::users(mlir::Operation *op) const {
  llvm::SmallSetVector<mlir::Operation *, 4> ret;
  forEachUser(op, [&](mlir::Operation *user) -> bool {
    ret.insert(user);
    return true;
  });

  return ret;
}

llvm::SmallSetVector<mlir::Operation *, 4>
ParametricTopologyAnalysis::providers(mlir::Operation *op) const {
  llvm::SmallSetVector<mlir::Operation *, 4> ret;
  forEachProvider(op, [&](mlir::Operation *provider) -> bool {
    ret.insert(provider);
    return true;
  });

  return ret;
}

/// Copied from
/// https://mlir.llvm.org/doxygen/TopologicalSortUtils_8cpp_source.html#l00107
/// and extended with parametric dependencies.

/// Return `true` if the given operation is ready to be scheduled.
bool inline ParametricTopologyAnalysis::isOpReady(
    Operation *op, DenseSet<Operation *> &unscheduledOps) {
  // A provider is ready if neither it nor any ancestor below `op` is
  // unscheduled.
  const auto isReady = [&](Operation *parent) {
    do {
      // Stop traversal when op under examination is reached.
      if (parent == op)
        return true;
      if (unscheduledOps.contains(parent))
        return false;
    } while ((parent = parent->getParentOp()));
    return true;
  };

  // Ready only if it and all its nested ops are ready.
  WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
    bool allOf = true;
    forEachProvider(nestedOp, [&](auto provider) {
      if (!allOf)
        return false;
      allOf &= isReady(provider);
      return true;
    });

    return allOf ? WalkResult::advance() : WalkResult::interrupt();
  });

  return !readyToSchedule.wasInterrupted();
}

void ParametricTopologyAnalysis::computeTopologicalSortingImpl(
    mlir::MutableArrayRef<mlir::Operation *> ops) {
  // In-degree (Kahn) scheduling: each op counts its blockers, the ops of `ops`
  // it still waits on, which are the ones `isOpReady` would find unscheduled.
  // Scheduling an op decrements the count of every op waiting on it, and an
  // op whose count reaches 0 is ready, so a later step of this sweep or the
  // next sweep schedules it. The blockers are collected once instead of
  // re-walking every pending op on every sweep: a graph out of parametric
  // order takes hundreds of sweeps. The sweeps and swaps are those of the
  // repeated-scan sort, so the order matches it exactly.
  DenseMap<Operation *, unsigned> idOf;
  for (auto [id, op] : llvm::enumerate(ops))
    idOf[op] = id;

  SmallVector<unsigned> numBlockers(ops.size(), 0);
  SmallVector<SmallVector<unsigned>> waitersOf(ops.size());
  for (auto [id, op] : llvm::enumerate(ops)) {
    DenseSet<unsigned> blockers;
    op->walk([&](Operation *nestedOp) {
      forEachProvider(nestedOp, [&](Operation *provider) {
        for (Operation *parent = provider; parent && parent != op;
             parent = parent->getParentOp())
          if (auto it = idOf.find(parent); it != idOf.end())
            blockers.insert(it->second);
        return true;
      });
    });
    numBlockers[id] = blockers.size();
    for (unsigned blocker : blockers)
      waitersOf[blocker].push_back(id);
  }

  // `ids` moves with `ops` so a ready check is an array read.
  auto ids = llvm::to_vector(llvm::seq<unsigned>(0, ops.size()));
  unsigned nextScheduledOp = 0;

  while (nextScheduledOp < ops.size()) {
    [[maybe_unused]] bool scheduledAtLeastOnce = false;

    // Schedule each ready op by swapping it to `nextScheduledOp`.
    for (unsigned i = nextScheduledOp; i < ops.size(); ++i) {
      if (numBlockers[ids[i]])
        continue;

      for (unsigned waiter : waitersOf[ids[i]])
        --numBlockers[waiter];
      std::swap(ops[i], ops[nextScheduledOp]);
      std::swap(ids[i], ids[nextScheduledOp]);
      scheduledAtLeastOnce = true;
      ++nextScheduledOp;
    }

    ASSERT_STREAM(scheduledAtLeastOnce,
                  << "No op was scheduled, potential cycle in the graph");
  }
}

void ParametricTopologyAnalysis::computeTopologicalSorting(
    mlir::MutableArrayRef<mlir::Operation *> ops) {
  computeTopologicalSortingImpl(ops);
}

// Not shared with the impl above: a common helper would over-abstract.
LogicalResult ParametricTopologyAnalysis::topologicallySortFirstLevelOpsWithin(
    mlir::Block *block) {
  if (block->empty())
    return success();

  DenseSet<Operation *> unscheduledOps;

  for (Operation &op : *block)
    unscheduledOps.insert(&op);

  Block::iterator nextScheduledOp = block->begin();
  Block::iterator end = block->end();

  while (!unscheduledOps.empty()) {
    bool scheduledAtLeastOnce = false;

    // Schedule each ready op by moving it before `nextScheduledOp`.
    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
      if (!isOpReady(&op, unscheduledOps))
        continue;

      unscheduledOps.erase(&op);
      scheduledAtLeastOnce = true;
      // Move the iterator forward if we schedule the operation at the front.
      if (&op == &*nextScheduledOp)
        ++nextScheduledOp;

      // Terminators must stay at the end of their block.
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      op.moveBefore(block, nextScheduledOp);
    }

    if (!scheduledAtLeastOnce) {
      // Scheduled ops are moved before `nextScheduledOp`, so every stuck op is
      // in the remaining range. Walk it rather than the DenseSet so the report
      // order is deterministic.
      auto stuckOps = llvm::make_filter_range(
          llvm::make_range(nextScheduledOp, end),
          [&](Operation &op) { return unscheduledOps.contains(&op); });
      InFlightDiagnostic diag =
          stuckOps.begin()->emitError()
          << "cannot schedule operation: dependency cycle in the graph";
      for (Operation &op : llvm::drop_begin(stuckOps))
        diag.attachNote(op.getLoc()) << "also unschedulable";
      return diag;
    }
  }
  return success();
}

LogicalResult
ParametricTopologyAnalysis::topologicallySortAllOpsWithin(Region *region) {
  auto regionKindInterface =
      dyn_cast<mlir::RegionKindInterface>(region->getParentOp());

  // Only graph regions can be reordered safely.
  if (!regionKindInterface)
    return success();

  if (regionKindInterface.getRegionKind(region->getRegionNumber()) !=
      RegionKind::Graph)
    return success();

  for (Block &block : *region) {
    if (failed(topologicallySortFirstLevelOpsWithin(&block)))
      return failure();
    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        if (failed(topologicallySortAllOpsWithin(&nestedRegion)))
          return failure();
  }
  return success();
}

// Returns op and its transitive users, in topological order.
SmallVector<Operation *> getRootedComponents(Operation *op) {
  SmallVector<Operation *> fringe;
  fringe.push_back(op);

  DenseSet<Operation *> enqueued;
  enqueued.insert(op);

  // Strategy is to DFS, and then sort later using built-in util.
  SmallVector<Operation *> answer;
  while (!fringe.empty()) {
    Operation *cur = fringe.back();
    fringe.pop_back();
    if (cur == nullptr)
      continue;
    answer.push_back(cur);

    for (Operation *user : cur->getUsers()) {
      if (enqueued.insert(user).second)
        fringe.push_back(user);
    }

    if (cur == op)
      continue;
    for (auto operand : cur->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (enqueued.insert(definingOp).second)
        fringe.push_back(definingOp);
    }
  }

  MO::ParametricTopologyAnalysis analysis(op->getParentOp());
  analysis.computeTopologicalSorting(answer);
  return answer;
}

} // namespace M::MO
