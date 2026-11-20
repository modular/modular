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

#ifndef GENERICML_DATALOW_UTILS_H
#define GENERICML_DATALOW_UTILS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace M {

enum class WalkDirection { Producers, Consumers };

/// The traversal every other helper here composes. Visits each op reachable
/// from `op` (inclusive) once, via operand producers (`Producers`) or result
/// users (`Consumers`). The functor returns an `mlir::WalkResult`:
///   - `advance()`   expand this op's neighbors,
///   - `skip()`      do not expand this op's neighbors (prune this branch),
///   - `interrupt()` stop the whole walk.
/// Block-argument operands (null producers) are skipped. With
/// `ThroughRegionCaptures`, a `Producers` walk also follows the outer values a
/// non-isolated region (an `mo.if` branch) reads:
/// ```
/// %mm = mo.matmul(%x, %y)
/// %r = mo.if (%c) { mo.yield mo.mul(%mm, %z) } else { ... }
/// %a = mo.add(%mm, %r)
/// ```
/// `%r`'s only operand is `%c`, but its branch reads `%mm`. Missing that edge,
/// fusing `%a` into the matmul's epilogue would create a cycle.
template <WalkDirection Dir, bool ThroughRegionCaptures = false>
inline mlir::WalkResult walkDataflowGraph(
    mlir::Operation *op,
    llvm::function_ref<mlir::WalkResult(mlir::Operation *)> functor) {
  static_assert(!ThroughRegionCaptures || Dir == WalkDirection::Producers,
                "a region capture is a producer edge");
  if (!op)
    return mlir::WalkResult::advance();

  llvm::SmallVector<mlir::Operation *> worklist{op};
  llvm::DenseSet<mlir::Operation *> visited;
  while (!worklist.empty()) {
    mlir::Operation *current = worklist.pop_back_val();
    if (!current)
      continue;
    if (!visited.insert(current).second)
      continue;

    mlir::WalkResult res = functor(current);
    if (res.wasInterrupted())
      return mlir::WalkResult::interrupt();
    if (res.wasSkipped())
      continue;

    if constexpr (Dir == WalkDirection::Producers) {
      for (mlir::Value operand : current->getOperands())
        if (mlir::Operation *def = operand.getDefiningOp())
          worklist.push_back(def);
      if constexpr (ThroughRegionCaptures)
        if (!current->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
          mlir::visitUsedValuesDefinedAbove(
              current->getRegions(), [&](mlir::OpOperand *use) {
                if (mlir::Operation *def = use->get().getDefiningOp())
                  worklist.push_back(def);
              });
    } else {
      for (mlir::Operation *user : current->getUsers())
        worklist.push_back(user);
    }
  }

  return mlir::WalkResult::advance();
}

/// Returns true iff some op reachable from `op` in `Dir` (inclusive) matches
/// `matcher`.
template <WalkDirection Dir = WalkDirection::Consumers,
          bool ThroughRegionCaptures = false>
inline bool
dataflowGraphContains(mlir::Operation *op,
                      llvm::function_ref<bool(mlir::Operation *)> matcher) {
  return walkDataflowGraph<Dir, ThroughRegionCaptures>(
             op,
             [&](mlir::Operation *o) {
               return matcher(o) ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
             })
      .wasInterrupted();
}

/// Checks if `start` depends on `target` transitively.
/// Eg
/// ```
/// %1 = foo(%0)
/// %2 = bar(%1)
/// ```
/// `dataflowGraphDependsOn(%2, %0) == true`.
/// Also returns true if `start == target`.
inline bool dataflowGraphDependsOn(mlir::Value start, mlir::Value target) {
  if (start == target)
    return true;
  mlir::Operation *startDef = start.getDefiningOp();
  if (!startDef)
    return false;
  // Scan operands rather than defining ops so block-arg targets match too.
  return dataflowGraphContains<WalkDirection::Producers>(
      startDef, [&](mlir::Operation *op) {
        return llvm::is_contained(op->getOperands(), target);
      });
}

/// Checks if `start` depends on `target` transitively.
/// Eg
/// ```
/// %1 = foo(%0)
/// %2 = bar(%1)
/// ```
/// `dataflowGraphDependsOn(%2, foo) == true`.
/// Also follows outer values a non-isolated region (an `mo.if` branch) reads,
/// since fusers ask this to rule out a cycle.
inline bool dataflowGraphDependsOn(mlir::Value start, mlir::Operation *target) {
  return dataflowGraphContains<WalkDirection::Producers,
                               /*ThroughRegionCaptures=*/true>(
      start.getDefiningOp(), [&](mlir::Operation *op) { return op == target; });
}

/// Checks if `start` depends on `target` transitively, through any of its
/// operands, in one walk. Follows region captures as the overload above does.
/// Also returns true if `start == target`.
inline bool dataflowGraphDependsOn(mlir::Operation *start,
                                   mlir::Operation *target) {
  return dataflowGraphContains<WalkDirection::Producers,
                               /*ThroughRegionCaptures=*/true>(
      start, [&](mlir::Operation *op) { return op == target; });
}

/// True iff every operand-less source op in `op`'s producer cone is one of
/// `ConstantTypes` and no op in it has a block-arg operand.
template <typename... ConstantTypes>
inline bool dataflowGraphFoldsToConstants(mlir::Operation *op) {
  if (!op)
    return false;
  return !walkDataflowGraph<WalkDirection::Producers>(
              op,
              [](mlir::Operation *o) -> mlir::WalkResult {
                if (o->getNumOperands() == 0)
                  return llvm::isa<ConstantTypes...>(o)
                             ? mlir::WalkResult::advance()
                             : mlir::WalkResult::interrupt();
                for (mlir::Value operand : o->getOperands())
                  if (!operand.getDefiningOp())
                    return mlir::WalkResult::interrupt();
                return mlir::WalkResult::advance();
              })
              .wasInterrupted();
}

} // namespace M

#endif
