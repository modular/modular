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

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_PARAMETRICTOPOLOGYANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_PARAMETRICTOPOLOGYANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

namespace M::MO {

// Extends SSA dataflow with two dependencies it can't see:
// (1) Parameters are attributes, so an op consuming a parameter another op
//     creates isn't visible through op->getUsers().
// (2) An op not isolated from above depends on whatever its nested ops use
//     from enclosing blocks.
// Example:
// %0 = ...
// %1 = op1 {
//   %2 = op2(%0)
// }
// Here, %0 is a provider of both op2 AND op1.
class ParametricTopologyAnalysis {
public:
  ParametricTopologyAnalysis(mlir::Operation *root);
  ~ParametricTopologyAnalysis() = default;

  // Returns {op->getUsers()} U {parametricUsers(op)}, in insertion order so
  // passes using it emit deterministic IR.
  llvm::SmallSetVector<mlir::Operation *, 4> users(mlir::Operation *op) const;

  template <typename Func>
  inline bool forEachUser(mlir::Operation *op, Func &&func) const {
    for (auto user : op->getUsers()) {
      if (!func(user))
        return false;
    }

    auto it = metadata.find(op);
    if (it != metadata.end()) {
      for (auto user : it->second.parametricUsers) {
        if (!func(user))
          return false;
      }
    }

    return true;
  }

  // Returns {dataflowProviders(op)} U {parametricProviders(op)}, in insertion
  // order so passes using it emit deterministic IR.
  llvm::SmallSetVector<mlir::Operation *, 4>
  providers(mlir::Operation *op) const;

  template <typename Func>
  inline bool forEachProvider(mlir::Operation *op, Func &&func) const {
    auto it = metadata.find(op);
    if (it != metadata.end()) {
      for (auto provider : it->second.parametricProviders) {
        if (!func(provider))
          return false;
      }
    }

    for (auto operand : op->getOperands()) {
      if (auto provider = operand.getDefiningOp()) {
        if (!func(provider))
          return false;
      }
    }

    return true;
  }

  // Topologically sorts `ops`, accounting for parametric dependencies.
  void computeTopologicalSorting(mlir::MutableArrayRef<mlir::Operation *> ops);

  // Topologically sort operations within `region` in place, accounting for
  // parametric dependencies and ops within nested regions. Fails, emitting an
  // error, if the dependencies form a cycle.
  mlir::LogicalResult topologicallySortAllOpsWithin(mlir::Region *region);

private:
  void
  computeTopologicalSortingImpl(mlir::MutableArrayRef<mlir::Operation *> ops);

  // Topologically sort operations within `block` in place (excluding ones in
  // nested region), accounting for parametric dependencies.
  mlir::LogicalResult topologicallySortFirstLevelOpsWithin(mlir::Block *block);

  bool isOpReady(mlir::Operation *op,
                 mlir::DenseSet<mlir::Operation *> &unscheduledOps);

  struct OperationRankAndDependency {
    llvm::SmallSetVector<mlir::Operation *, 4> parametricProviders;
    llvm::SmallSetVector<mlir::Operation *, 4> parametricUsers;
  };

  mlir::DenseMap<mlir::Operation *, OperationRankAndDependency> metadata;
};

// Returns, in topological order, the ops connected to `op`'s results when
// edges are treated as undirected and `op` connects only through its results.
// Used to clean up machinery around raising functions.
mlir::SmallVector<mlir::Operation *> getRootedComponents(mlir::Operation *op);

} // namespace M::MO

#endif
