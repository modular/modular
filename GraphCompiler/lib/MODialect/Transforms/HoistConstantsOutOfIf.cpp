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

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Mojo/KGENDialect/KGENOps.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "mo-hoist-constants-out-of-if"

using namespace mlir;
using namespace M;
using namespace M::MO;

namespace M::MO {
#define GEN_PASS_DEF_HOISTCONSTANTSOUTOFIF
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"
} // namespace M::MO

namespace {

/// Returns true if the types or attributes of `op` refer to a parameter. The
/// parameter can be declared inside the branch, so the op cannot move.
static bool refersToParam(Operation *op) {
  bool found = false;
  auto visit = [&](KGEN::ParamDeclRefAttr) {
    found = true;
    return WalkResult::interrupt();
  };
  for (Type type : op->getResultTypes())
    type.walk(visit);
  op->getAttrDictionary().walk(visit);
  return found;
}

/// Returns true if a result of `op` has more elements than all its tensor
/// operands together, as a broadcast has.
static bool isExpanding(Operation *op) {
  std::optional<int64_t> operandSize;
  for (Value operand : op->getOperands())
    if (MO::TensorType type = dynCastFromTensorLike(operand.getType()))
      if (std::optional<int64_t> size = type.getNumElements())
        operandSize = operandSize.value_or(0) + *size;
  if (!operandSize)
    return false;
  return llvm::any_of(op->getResultTypes(), [&](Type type) {
    MO::TensorType tensor = dynCastFromTensorLike(type);
    if (!tensor)
      return false;
    std::optional<int64_t> size = tensor.getNumElements();
    return !size || *size > *operandSize;
  });
}

static bool canMove(Operation *op) {
  return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
         isMemoryEffectFree(op) && !refersToParam(op);
}

/// Returns true if a tensor result of `op` is on another device than a tensor
/// operand of it. The result is then a copy, which does not fuse into its
/// users.
static bool changesDevice(Operation *op) {
  auto deviceOf = [](Type type) -> DeviceRefAttr {
    auto withDevice = dyn_cast<MOHasDeviceTypeInterface>(type);
    return withDevice ? withDevice.getDeviceRef() : DeviceRefAttr();
  };
  return llvm::any_of(op->getResultTypes(), [&](Type resultType) {
    DeviceRefAttr resultDevice = deviceOf(resultType);
    return resultDevice &&
           llvm::any_of(op->getOperandTypes(), [&](Type operandType) {
             DeviceRefAttr operandDevice = deviceOf(operandType);
             return operandDevice && operandDevice != resultDevice;
           });
  });
}

/// Removes from `constant` each op in a branch that computes from an expanding
/// op and has a user that stays in the branch. Such an op fuses into that
/// user, and keeping its result for every execution would cost more memory
/// than recomputing it. A copy to another device ends the expanded data: the
/// copy is made either way, so moving it saves the copy on every execution.
static void keepExpandedDataInBranches(GraphOp graph,
                                       DenseSet<Operation *> &constant) {
  DenseSet<Operation *> expanded;
  SmallVector<Operation *> worklist;
  graph.walk([&](Operation *op) {
    if (constant.contains(op) && isExpanding(op) && op->getParentOfType<IfOp>())
      worklist.push_back(op);
  });
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!expanded.insert(op).second)
      continue;
    for (Operation *user : op->getUsers())
      if (constant.contains(user) && !changesDevice(user))
        worklist.push_back(user);
  }

  // Removing an op can leave an op that feeds it with a user that stays.
  bool removed = true;
  while (removed) {
    removed = false;
    for (Operation *op : expanded) {
      if (!constant.contains(op) ||
          llvm::all_of(op->getUsers(), [&](Operation *user) {
            return constant.contains(user);
          }))
        continue;
      constant.erase(op);
      removed = true;
    }
  }
}

/// Returns the movable ops in `graph` whose results depend only on constants
/// and weights.
static DenseSet<Operation *> constantDerivedOps(GraphOp graph) {
  DenseSet<Operation *> constant;
  SmallVector<Operation *> worklist;
  graph.walk([&](Operation *op) {
    if (op != graph.getOperation())
      worklist.push_back(op);
  });

  // The regions are graph regions, so a use can come before its definition.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (constant.contains(op) || !canMove(op))
      continue;
    // A chain created from nothing orders ops but carries no data.
    bool isSeed =
        isa<MO::ConstantLike>(op) ||
        (op->getNumOperands() == 0 &&
         llvm::all_of(op->getResultTypes(), llvm::IsaPred<ChainType>));
    bool hasConstantOperands =
        op->getNumOperands() > 0 &&
        llvm::all_of(op->getOperands(), [&](Value operand) {
          Operation *def = operand.getDefiningOp();
          return def && constant.contains(def);
        });
    if (!isSeed && !hasConstantOperands)
      continue;
    constant.insert(op);
    for (Operation *user : op->getUsers())
      worklist.push_back(user);
  }
  return constant;
}

/// Ops in the blocks of an if that compute the same value.
struct EquivalentOps {
  Operation *leader;
  SmallVector<Operation *> members;
  bool inThen = false;
  bool inElse = false;
};

/// Moves in front of `ifOp` each op in `constant` that both of its blocks
/// compute, as one op that both blocks use. An op that only one block
/// computes stays in that block.
static void hoistSharedOps(IfOp ifOp, DenseSet<Operation *> &constant) {
  // Ops are grouped in dependency order: an op joins a group once each of
  // its operands is defined outside `ifOp` or by an op already grouped, so
  // the groups of an op's operands come before its own.
  SmallVector<EquivalentOps> groups;
  DenseMap<Operation *, unsigned> groupOf;
  DenseMap<llvm::hash_code, SmallVector<unsigned, 1>> groupsByHash;

  // The value that stands for `value` in comparisons: the matching result of
  // its group's leader.
  auto leaderValue = [&](Value value) -> Value {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return value;
    auto it = groupOf.find(result.getOwner());
    if (it == groupOf.end())
      return value;
    return groups[it->second].leader->getResult(result.getResultNumber());
  };
  auto isReady = [&](Operation *op) {
    return llvm::all_of(op->getOperands(), [&](Value value) {
      if (!ifOp->isAncestor(value.getParentRegion()->getParentOp()))
        return true;
      Operation *def = value.getDefiningOp();
      return def && groupOf.contains(def);
    });
  };
  auto sameValue = [&](Value lhs, Value rhs) -> LogicalResult {
    return mlir::success(leaderValue(lhs) == leaderValue(rhs));
  };
  auto flags = OperationEquivalence::IgnoreLocations;

  SmallVector<Operation *> pending;
  for (Region &region : ifOp->getRegions())
    for (Operation &op : region.getOps())
      if (constant.contains(&op))
        pending.push_back(&op);

  // The regions are graph regions, so a use can come before its definition.
  bool grouped = true;
  while (grouped) {
    grouped = false;
    SmallVector<Operation *> waiting;
    for (Operation *op : pending) {
      if (!isReady(op)) {
        waiting.push_back(op);
        continue;
      }
      llvm::hash_code hash = OperationEquivalence::computeHash(
          op, [&](Value value) { return hash_value(leaderValue(value)); },
          OperationEquivalence::ignoreHashValue, flags);
      SmallVector<unsigned, 1> &candidates = groupsByHash[hash];
      auto match = llvm::find_if(candidates, [&](unsigned index) {
        return OperationEquivalence::isEquivalentTo(
            groups[index].leader, op, sameValue, /*markEquivalent=*/nullptr,
            flags);
      });
      unsigned index;
      if (match != candidates.end()) {
        index = *match;
      } else {
        index = groups.size();
        groups.push_back({op, {}});
        candidates.push_back(index);
      }
      EquivalentOps &group = groups[index];
      group.members.push_back(op);
      if (op->getParentRegion() == &ifOp.getThenRegion())
        group.inThen = true;
      else
        group.inElse = true;
      groupOf[op] = index;
      grouped = true;
    }
    pending = std::move(waiting);
  }

  // The operands of a group in both blocks are defined outside `ifOp` or by
  // an earlier group in both blocks, which has already moved.
  for (EquivalentOps &group : groups) {
    if (!group.inThen || !group.inElse)
      continue;
    group.leader->moveBefore(ifOp);
    for (Operation *member : group.members) {
      if (member == group.leader)
        continue;
      member->replaceAllUsesWith(group.leader);
      constant.erase(member);
      member->erase();
    }
  }
}

struct HoistConstantsOutOfIfPass
    : public M::MO::impl::HoistConstantsOutOfIfBase<HoistConstantsOutOfIfPass> {
  void runOnOperation() override {
    GraphOp graph = getOperation();

    // Post-order: an inner if moves its ops into the enclosing block first,
    // so the enclosing if can match them against its other block.
    SmallVector<IfOp> ifOps;
    graph.walk([&](IfOp ifOp) { ifOps.push_back(ifOp); });
    if (ifOps.empty())
      return;

    DenseSet<Operation *> constant = constantDerivedOps(graph);
    keepExpandedDataInBranches(graph, constant);
    for (IfOp ifOp : ifOps)
      hoistSharedOps(ifOp, constant);
  }
};

} // namespace
