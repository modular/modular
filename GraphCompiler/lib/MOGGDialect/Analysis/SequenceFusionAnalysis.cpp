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

#include "GraphCompiler/MOGGDialect/Analysis/SequenceFusionAnalysis.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/AssertStream.h"

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SetOperations.h"

namespace M::MOGG {

enum class Direction { Up, Down };

/// The returned set contains all operations that are eligible, compatible with
/// one another and all paths to them from the root are void of non eligible
/// ops. A SetVector is returned for determinism. Will not traverse across
/// blocks.
/// Note: The compatibility function is used on the edges during the traversal.
/// The operands must be interpreted as <from, to>.
template <Direction direction>
mlir::SetVector<mlir::Operation *>
floodFillFrom(const mlir::SmallVector<mlir::Operation *> &roots,
              EligibilityFunction isEligible,
              CompatibilityFunction areCompatible,
              MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  mlir::SetVector<mlir::Operation *> visitedAsEligible;
  mlir::SetVector<mlir::Operation *> visitedAsNonEligible;

  mlir::Block *block = roots.front()->getBlock();
  llvm::SmallVector<std::tuple<mlir::Operation *, Operation *, bool>> workList;
  for (mlir::Operation *op : roots)
    workList.push_back({op, nullptr, true});

  while (!workList.empty()) {
    auto entry = workList.pop_back_val();
    Operation *op, *from;
    bool asEligible;
    std::tie(op, from, asEligible) = entry;

    if (op->getBlock() != block)
      continue;

    asEligible &= isEligible(op);
    if (asEligible && from) {
      if constexpr (direction == Direction::Down)
        asEligible &= areCompatible(from, op);
      else
        asEligible &= areCompatible(op, from);
    }

    if (asEligible) {
      // If the current op is eligible, then ensure it was not invalidated or
      // already visited.
      if (visitedAsNonEligible.contains(op))
        continue;

      if (!visitedAsEligible.insert(op))
        continue;
    } else if (!visitedAsNonEligible.insert(op)) {
      // If the current op is not eligible, ensure it was not already visited
      // and remove it from eligible ops if it was thought to be eligible.
      continue;
    }

    if constexpr (direction == Direction::Down) {
      completeDataflowGraph.forEachUser(op, [&](auto user) {
        workList.push_back({user, op, asEligible});
        return true;
      });
    } else {
      completeDataflowGraph.forEachProvider(op, [&](auto provider) {
        workList.push_back({provider, op, asEligible});
        return true;
      });
    }
  }

  visitedAsEligible.set_subtract(visitedAsNonEligible);

  return visitedAsEligible;
}

FusionSubGraphs floodFillSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible) {
  FusionSubGraphs subgraphs;
  region.walk([&](Block *block) {
    mlir::DenseSet<mlir::Operation *> alreadyProcessedKernelOps;
    for (mlir::Operation &root : block->getOperations()) {
      if (!isEligible(&root))
        continue;

      if (!alreadyProcessedKernelOps.insert(&root).second)
        continue;

      // Gather all eligible operations that are accessible from this op and
      // not blocked by a non eligible op.
      auto opsToFuseTogetherDownward = floodFillFrom<Direction::Down>(
          {&root}, isEligible, areCompatible, completeDataflowGraph);

      // Find the leaves of that subgraph.
      const auto isALeaf = [&](mlir::Operation *op) {
        bool userContained = false;
        completeDataflowGraph.forEachUser(op, [&](auto user) {
          userContained |= opsToFuseTogetherDownward.contains(user);
          return !userContained;
        });
        return !userContained;
      };
      mlir::SmallVector<mlir::Operation *> leaves;
      for (mlir::Operation *op : opsToFuseTogetherDownward) {
        if (isALeaf(op))
          leaves.push_back(op);
      }

      // Gather all eligible operations again but going upward now.
      auto opsToFuseTogetherUpward = floodFillFrom<Direction::Up>(
          leaves, isEligible, areCompatible, completeDataflowGraph);

      opsToFuseTogetherDownward.set_union(opsToFuseTogetherUpward);

      for (mlir::Operation *kernelOpInFusionSet : opsToFuseTogetherDownward)
        alreadyProcessedKernelOps.insert(kernelOpInFusionSet);

      if (opsToFuseTogetherDownward.size() > 1) {
        subgraphs.push_back(llvm::SmallVector<mlir::Operation *>(
            opsToFuseTogetherDownward.begin(),
            opsToFuseTogetherDownward.end()));
      }
    }
  });

  return subgraphs;
}

struct OperationDependency {
  /// Position in the bit vectors.
  uint64_t index;
  /// Bit set where 1 means that the operation is (transitively) a provider.
  llvm::BitVector transitiveProviders;
  /// Bit set where 1 means that the operation is (transitively) a user.
  llvm::BitVector transitiveUsers;
};

/// Gathers dependencies of an operation. Should be called in topological order
/// with Up direction and reverse topological order with Down direction. This
/// algorithm is O(n) but with favorable constants, thanks to the BitVector.
template <Direction direction>
static void dependencyWalk(
    mlir::Operation *op,
    llvm::MapVector<mlir::Operation *, OperationDependency> &dependencies,
    const mlir::DenseMap<mlir::Operation *, uint64_t> &opToIndexInBitVector,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  const auto grabDependencies = [&](auto nextOp) -> bool {
    auto &thisOpDependencies = dependencies[op];
    // Lookup the operation. If it is not in the index map, then it does not
    // belong to this block. Ignore it.
    auto operationIndexIt = opToIndexInBitVector.find(nextOp);
    if (operationIndexIt == opToIndexInBitVector.end())
      return true;
    uint64_t operationIndex = operationIndexIt->second;

    const auto &dependency = dependencies[nextOp];
    // Op depends (as provider or user) on nextOp.
    if constexpr (direction == Direction::Up) {
      thisOpDependencies.transitiveProviders[operationIndex] = true;
      thisOpDependencies.transitiveProviders |= dependency.transitiveProviders;
    } else {
      thisOpDependencies.transitiveUsers[operationIndex] = true;
      thisOpDependencies.transitiveUsers |= dependency.transitiveUsers;
    }

    return true;
  };

  if constexpr (direction == Direction::Up)
    completeDataflowGraph.forEachProvider(op, grabDependencies);
  else
    completeDataflowGraph.forEachUser(op, grabDependencies);
}

/// Gather dependencies for all operations in the provided block. O(n**2) but
/// with favorable constants, thanks to the BitVector. Using a MapVector to
/// guarantee a deterministic output.
static std::tuple<llvm::MapVector<mlir::Operation *, OperationDependency>,
                  llvm::BitVector>
gatherDependencies(
    Block *block, EligibilityFunction isEligible,
    ComputationallySignificantFunction isComputationallySignificant,
    MO::ParametricTopologyAnalysis &completeDataflowGraph) {
  // Maps from any op in the block to it's index in the bit vectors we create.
  mlir::DenseMap<mlir::Operation *, uint64_t> opToIndexInBitVector;
  // The return value which maps an eligible op to its transitive dependencies.
  llvm::MapVector<mlir::Operation *, OperationDependency> dependencies;
  uint64_t opIndex = 0;
  SmallVector<Operation *> allOps;
  for (auto &op : block->getOperations())
    allOps.push_back(&op);
  // Topologically sort the eligible ops and then compute the dependencies in
  // that order. This makes it possible not to traverse the graph again.
  completeDataflowGraph.computeTopologicalSorting(allOps);

  bool atLeastOneEligibleOp = false;
  for (auto op : allOps) {
    opToIndexInBitVector[op] = opIndex++;
    atLeastOneEligibleOp |= isEligible(op);
  }

  if (!atLeastOneEligibleOp)
    return {};

  for (auto op : allOps) {
    auto &entry = dependencies[op];
    entry.transitiveProviders.resize(opToIndexInBitVector.size(), false);
    entry.transitiveUsers.resize(opToIndexInBitVector.size(), false);
    entry.index = opToIndexInBitVector[op];
  }

  llvm::BitVector computationallySignificantMask;
  computationallySignificantMask.resize(opToIndexInBitVector.size(), false);
  for (auto [op, index] : opToIndexInBitVector) {
    if (isComputationallySignificant(op)) {
      ASSERT_STREAM(
          !isEligible(op),
          << "Eligible ops are considered computationally insignificant");
      computationallySignificantMask[index] = true;
    }
  }

  for (auto op : allOps) {
    dependencyWalk<Direction::Up>(op, dependencies, opToIndexInBitVector,
                                  completeDataflowGraph);
  }

  // Same logic for going down, except we use the reverse topological sort.
  for (auto op : llvm::reverse(allOps)) {
    dependencyWalk<Direction::Down>(op, dependencies, opToIndexInBitVector,
                                    completeDataflowGraph);
  }

  dependencies.remove_if([&](auto entry) { return !isEligible(entry.first); });

  return {dependencies, computationallySignificantMask};
}

/// The analysis applies the following observation:
/// M := Set of non eligible ops.
/// S := A subgraph containing eligible ops that are compatible with one
/// another.
/// providers(S) := The set of all operations that transitively feed into an
/// eligible subgraph.
/// users(S) := The set of all operations that use an eligible subgraph.
/// A subgraph of eligible and compatible operations O can be added to another
/// subgraph of eligible and compatible operations S iff no cycle is created, no
/// dependency between non eligible ops is created and all operations in both
/// sets are compatible.
/// A cycle is created iff:
///   (providers(S) U providers(O)) intersects (users(S) U users(O))
/// A dependency between non eligible ops is created iff:
///   let P_S' = providers(S) intersection M
///   let P_O' = providers(O) intersection M
///   let U_S' = users(S) intersection M
///   let U_O' = users(O) intersection M
///   |P_S' - P_O'| != 0 && |U_O' - U_S'| != 0 &&
///   |P_O' - P_S'| != 0 && |U_S' - U_O'| != 0 ||
/// The implementation computes the transitive dependencies of all eligible ops.
/// Then, for each op, it checks if there is a set it can fuse with. If it
/// can't, it extends the sets by the current op. O(N**3) but with favorable
/// constants, thanks to the BitVector. The algorithm can also ignore
/// parallelism and only guard for cycles using `preserveParallelism`.
FusionSubGraphs transitiveDependencySequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    ComputationallySignificantFunction isComputationallySignificant,
    bool preserveParallelism) {
  const auto wouldCreateANewDependency =
      [&](const OperationDependency &lhs, const OperationDependency &rhs,
          const llvm::BitVector &computationallySignificantMask) {
        const auto bitsetDifferenceNonEmpty = [&](const llvm::BitVector &lhs,
                                                  const llvm::BitVector &rhs) {
          // |S - O| != 0 implemented with bitwise ops:
          // lhs - (lhs & rhs) != 0
          static thread_local llvm::BitVector tmp;
          tmp = lhs;
          tmp &= rhs;
          return lhs.test(tmp);
        };

        static thread_local llvm::BitVector tmp1, tmp2, tmp3, tmp4;
        tmp1 = lhs.transitiveProviders;
        tmp2 = rhs.transitiveProviders;
        tmp3 = lhs.transitiveUsers;
        tmp4 = rhs.transitiveUsers;

        // S' = S intersection M
        tmp1 &= computationallySignificantMask;
        tmp2 &= computationallySignificantMask;
        tmp3 &= computationallySignificantMask;
        tmp4 &= computationallySignificantMask;

        bool createdNewDependency = (bitsetDifferenceNonEmpty(tmp1, tmp2) &&
                                     bitsetDifferenceNonEmpty(tmp4, tmp3)) ||
                                    (bitsetDifferenceNonEmpty(tmp2, tmp1) &&
                                     bitsetDifferenceNonEmpty(tmp3, tmp4));

        return createdNewDependency;
      };

  const auto wouldCreateACycle = [&](const OperationDependency &lhs,
                                     const OperationDependency &rhs) {
    static thread_local llvm::BitVector tmp1, tmp2;

    // all_providers = providers(lhs) U providers(rhs)
    tmp1 = lhs.transitiveProviders;
    tmp1 |= rhs.transitiveProviders;

    // all_users = users(lhs) U users(rhs)
    tmp2 = lhs.transitiveUsers;
    tmp2 |= rhs.transitiveUsers;

    // tmp1 = intersection(all_providers, all_users)
    tmp1 &= tmp2;

    // Is something a provider and a user at the same time?
    return tmp1.any();
  };

  struct SubgraphEntry {
    llvm::SmallVector<mlir::Operation *> operations;
    llvm::BitVector operationsAskMask;
    OperationDependency dependencies;
  };
  llvm::SmallVector<SubgraphEntry, 4> subgraphEntries;

  region.walk([&](Block *block) {
    auto [eligibleOpsDependencies, computationallySignificantMask] =
        gatherDependencies(block, isEligible, isComputationallySignificant,
                           completeDataflowGraph);
    while (!eligibleOpsDependencies.empty()) {
      auto [eligibleOp, eligibleOpDependencies] =
          std::move(eligibleOpsDependencies.front());
      eligibleOpsDependencies.erase(eligibleOpsDependencies.begin());

      bool wasInsertedInASubgraph = false;
      for (auto &[operations, operationsAskMask, subgraphDependencies] :
           subgraphEntries) {
        if (operations.front()->getBlock() != block)
          continue;

        if (!areCompatible(eligibleOp, operations.front()))
          continue;

        if (wouldCreateACycle(subgraphDependencies, eligibleOpDependencies))
          continue;

        if (preserveParallelism &&
            wouldCreateANewDependency(subgraphDependencies,
                                      eligibleOpDependencies,
                                      computationallySignificantMask)) {
          continue;
        }

        operationsAskMask[eligibleOpDependencies.index] = true;
        subgraphDependencies.transitiveProviders |=
            eligibleOpDependencies.transitiveProviders;
        subgraphDependencies.transitiveUsers |=
            eligibleOpDependencies.transitiveUsers;
        subgraphDependencies.transitiveProviders[eligibleOpDependencies.index] =
            false;
        subgraphDependencies.transitiveUsers[eligibleOpDependencies.index] =
            false;
        operations.push_back(eligibleOp);

        const auto updateDependenciesAfterFusion =
            [](OperationDependency &dependenciesToUpdate,
               const OperationDependency &subgraphDependencies,
               const llvm::BitVector &mask) {
              const bool providerOfTheSubgraph =
                  dependenciesToUpdate.transitiveProviders.anyCommon(mask);
              if (providerOfTheSubgraph) {
                dependenciesToUpdate.transitiveProviders |= mask;
                dependenciesToUpdate.transitiveProviders |=
                    subgraphDependencies.transitiveProviders;
              }
              const bool userOfTheSubgraph =
                  dependenciesToUpdate.transitiveUsers.anyCommon(mask);
              if (userOfTheSubgraph) {
                dependenciesToUpdate.transitiveUsers |= mask;
                dependenciesToUpdate.transitiveUsers |=
                    subgraphDependencies.transitiveUsers;
              }
              ASSERT_STREAM(!(providerOfTheSubgraph && userOfTheSubgraph),
                            << "Created a cycle");
            };

        // After fusing an operation with a subgraph, update the dependencies of
        // any other op or subgraph to account for the newly created
        // dependencies.
        for (auto &[op, eligibleOpDependencies_] : eligibleOpsDependencies) {
          if (op->getBlock() == block) {
            updateDependenciesAfterFusion(eligibleOpDependencies_,
                                          subgraphDependencies,
                                          operationsAskMask);
          }
        }

        for (auto &[operations_, __, subgraphDependencies_] : subgraphEntries) {
          if (operations_.front()->getBlock() == block) {
            updateDependenciesAfterFusion(
                subgraphDependencies_, subgraphDependencies, operationsAskMask);
          }
        }

        wasInsertedInASubgraph = true;

        break;
      }

      if (!wasInsertedInASubgraph) {
        llvm::BitVector operationsAskMask(
            eligibleOpDependencies.transitiveProviders.size(), false);
        operationsAskMask[eligibleOpDependencies.index] = true;
        subgraphEntries.push_back(
            {{eligibleOp}, operationsAskMask, eligibleOpDependencies});
      }
    }
  });

  FusionSubGraphs subgraphs;
  for (auto &[operations, _, __] : subgraphEntries) {
    if (operations.size() > 1)
      subgraphs.push_back(operations);
  }

  return subgraphs;
}

FusionSubGraphs sourceDestinationSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    CanDuplicate canDuplicate, uint64_t maximumFusionSetSize) {
  FusionSubGraphs subgraphs;
  region.walk([&](Block *block) {
    if (isa<AsyncRoutineOp>(block->getParentOp()))
      return;

    int64_t numOps = block->getOperations().size();
    SmallVector<Operation *> topoSortedOps;
    topoSortedOps.reserve(numOps);
    for (auto &op : block->getOperations())
      if (isEligible(&op))
        topoSortedOps.push_back(&op);

    // This mutates `topoSortedOps` in place to a sorted order.
    completeDataflowGraph.computeTopologicalSorting(topoSortedOps);

    // Map each operation in the block to the topological index
    DenseMap<Operation *, int64_t> operationToTopoIndex;
    operationToTopoIndex.reserve(topoSortedOps.size());
    for (auto [i, op] : llvm::enumerate(topoSortedOps))
      operationToTopoIndex[op] = i;

    auto areSameBlockAndCompatible =
        [&,
         cache = llvm::DenseMap<std::pair<Operation *, Operation *>, bool>()](
            Operation *parent, Operation *child) mutable {
          auto [it, inserted] = cache.try_emplace({parent, child}, false);

          if (inserted)
            it->second = isEligible(parent) && isEligible(child) &&
                         parent->getBlock() == child->getBlock() &&
                         areCompatible(parent, child);

          return it->second;
        };

    // Set of operations included in a fuseable subgraph.
    DenseSet<Operation *> includedInASubgraph;
    for (auto op : llvm::reverse(topoSortedOps)) {
      // If an op was fused, it means it will be DCE-ed later so we do not
      // attempt to fuse things into it.
      if (includedInASubgraph.contains(op))
        continue;

      llvm::SetVector<Operation *> curSubgraph;

      using FusionPair = std::tuple<mlir::Operation *, Operation *>;

      // Visit pairs in reverse topological order of their parent (the first
      // FusionPair element).
      auto comp = [&operationToTopoIndex](const FusionPair &a,
                                          const FusionPair &b) {
        auto parentA = std::get<0>(a);
        auto parentB = std::get<0>(b);
        return operationToTopoIndex.at(parentA) <
               operationToTopoIndex.at(parentB);
      };

      llvm::PriorityQueue<FusionPair, std::vector<FusionPair>, decltype(comp)>
          workQueue(comp);
      workQueue.emplace(op, nullptr);

      // Parents we've visited already. Once we've visited we know whether they
      // are part of the subgraph or not already.
      DenseSet<Operation *> visited;
      while (!workQueue.empty() && curSubgraph.size() < maximumFusionSetSize) {
        auto parentChild = workQueue.top();
        workQueue.pop();

        // A lambda captures parent, which structured bindings allow only in
        // C++20.
        auto parent = std::get<0>(parentChild);
        auto child = std::get<1>(parentChild);

        if (auto iterAndInserted = visited.insert(parent);
            !iterAndInserted.second)
          continue;

        if (child) {
          auto canDup = canDuplicate(parent);

          bool proceed =
              completeDataflowGraph.forEachUser(parent, [&](Operation *user) {
                if (!areSameBlockAndCompatible(parent, user))
                  return false;

                // If duplication is not allowed, then the only user of the
                // current op must be in the fusion subgraph already. Note that
                // if there are more than one user, this will fail because one
                // of the users will be in another subgraph from a past
                // iteration.
                return canDup || curSubgraph.contains(user);
              });

          if (!proceed)
            continue;
        }

        curSubgraph.insert(parent);
        includedInASubgraph.insert(parent);

        completeDataflowGraph.forEachProvider(parent, [&](auto grandParent) {
          // Don't include ops not in the current block
          if (operationToTopoIndex.contains(grandParent))
            workQueue.emplace(grandParent, parent);

          return true;
        });
      }

      if (curSubgraph.size() > 1)
        subgraphs.emplace_back(curSubgraph.begin(), curSubgraph.end());
    }
  });
  return subgraphs;
}

} // namespace M::MOGG
