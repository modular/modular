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
// mo-create-parallel-from-unbundle
//
// When each of a `mo.tensor.unbundle`'s N results has exactly one user and
// those users are distinct and structurally identical, wraps them in a fresh
// `mo.parallel` after the unbundle.
//
// The new parallel's primary input is a transient
// `bundle(unbundle(X)#0..N-1)`, which the post-pass canonicalize folds back to
// `X` so `mo-merge-adjacent-parallels` can merge it into the producer.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Support/AssertStream.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "create-parallel-from-unbundle"

namespace M::MO {

#define GEN_PASS_DEF_CREATEPARALLELFROMUNBUNDLE
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

struct WrapCandidate {
  TensorUnbundleOp unbundleOp;
  /// One consumer op per launch, in launch-index order.
  SmallVector<Operation *> consumerOps;
  /// Operand position of `unbundleOp.getResult(i)` in `consumerOps[i]`,
  /// identical across launches.
  size_t drivingOperandIdx;
};

struct InputPlan {
  /// Outer values per bundle input. Entry 0 is the driving unbundle's results;
  /// the rest cover non-driving operand positions.
  SmallVector<SmallVector<Value>> bundles;
  /// For each operand of the representative consumer, the index into
  /// `bundles` whose corresponding block argument supplies it.
  SmallVector<size_t> operandToBlockArg;
};

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Returns true if \p op can be cloned into a parallel body.
static bool isAbsorbableConsumer(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  for (Value result : op->getResults())
    if (!isa<TensorType>(result.getType()))
      return false;
  for (Value operand : op->getOperands())
    if (isa<BufferType, ChainType>(operand.getType()))
      return false;
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return false;
  if (auto iface = dyn_cast<mlir::MemoryEffectOpInterface>(op))
    return iface.hasNoEffect();
  // MO elementwise / custom ops are pure but may not implement
  // MemoryEffectOpInterface; the buffer/chain check guards chain-bearing ops.
  return true;
}

/// Returns true if \p a and \p b match structurally, ignoring the per-launch
/// `device` attribute.
static bool opsMatch(Operation *a, Operation *b) {
  if (a->getName() != b->getName() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults())
    return false;

  auto stripDevice = [](DictionaryAttr attrs) {
    SmallVector<NamedAttribute> filtered;
    filtered.reserve(attrs.size());
    for (NamedAttribute attr : attrs)
      if (attr.getName() != kOperationDeviceRefAttrName)
        filtered.push_back(attr);
    return DictionaryAttr::get(attrs.getContext(), filtered);
  };
  return stripDevice(a->getAttrDictionary()) ==
         stripDevice(b->getAttrDictionary());
}

/// Returns true if \p perLaunchValues can be one bundle input: shape and dtype
/// agree across launches, and each device matches \p launchDevices (the
/// driving unbundle's source bundle devices).
static bool canBundlePerLaunch(ArrayRef<Value> perLaunchValues,
                               ArrayRef<DeviceRefAttr> launchDevices) {
  auto firstTy = dyn_cast<TensorType>(perLaunchValues[0].getType());
  if (!firstTy)
    return false;

  for (Value v : perLaunchValues.drop_front()) {
    auto ty = dyn_cast<TensorType>(v.getType());
    if (!ty || firstTy.getShape() != ty.getShape() ||
        firstTy.getDtype() != ty.getDtype())
      return false;
  }

  for (auto [i, v] : llvm::enumerate(perLaunchValues))
    if (cast<TensorType>(v.getType()).getDeviceRef() != launchDevices[i])
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Candidate discovery
//===----------------------------------------------------------------------===//

/// Returns a wrap candidate seeded by \p unbundleOp, or std::nullopt if its
/// consumers don't qualify.
static std::optional<WrapCandidate> findCandidate(TensorUnbundleOp unbundleOp) {
  if (unbundleOp.getNumResults() == 0)
    return std::nullopt;
  if (!llvm::all_of(unbundleOp.getResults(),
                    [](Value v) { return v.hasOneUse(); }))
    return std::nullopt;
  auto consumers = llvm::map_to_vector(unbundleOp.getResults(), [](Value v) {
    return v.getUses().begin()->getOwner();
  });

  // A binary op consuming two unbundled elements (e.g. `mo.add(u#0, u#1)`)
  // would otherwise be cloned/erased twice.
  llvm::SmallDenseSet<Operation *> unique;
  for (Operation *c : consumers)
    if (!unique.insert(c).second)
      return std::nullopt;

  Operation *rep = consumers[0];
  if (!isAbsorbableConsumer(rep))
    return std::nullopt;
  for (Operation *op : llvm::drop_begin(consumers))
    if (!opsMatch(rep, op))
      return std::nullopt;

  // unbundle#0 has one use, so it appears at exactly one operand of `rep`.
  OperandRange repOperands = rep->getOperands();
  auto it = llvm::find(repOperands, unbundleOp.getResult(0));
  ASSERT_STREAM(it != repOperands.end(),
                << "rep " << *rep << " is the unique user of "
                << unbundleOp.getResult(0) << " but does not reference it");
  size_t drivingOperandIdx = std::distance(repOperands.begin(), it);

  // Peers must share the driving operand index, or substituting block args by
  // position mis-routes operands (e.g. a commutative op in swapped order).
  for (auto [i, peer] : llvm::enumerate(llvm::drop_begin(consumers)))
    if (peer->getOperand(drivingOperandIdx) != unbundleOp.getResult(i + 1))
      return std::nullopt;

  return WrapCandidate{unbundleOp, std::move(consumers), drivingOperandIdx};
}

//===----------------------------------------------------------------------===//
// Input planning
//===----------------------------------------------------------------------===//

/// Gives each non-driving operand position its own per-launch bundle. Returns
/// std::nullopt if such an operand is shared across all launches (would need
/// an outer capture) or can't be bundled.
static std::optional<InputPlan>
planInputs(WrapCandidate &candidate, ArrayRef<DeviceRefAttr> launchDevices) {
  size_t numLaunches = candidate.unbundleOp.getNumResults();
  size_t numOperands = candidate.consumerOps[0]->getNumOperands();

  InputPlan plan;
  plan.operandToBlockArg.assign(numOperands, 0);
  plan.bundles.emplace_back(candidate.unbundleOp.getResults().begin(),
                            candidate.unbundleOp.getResults().end());

  for (size_t idx = 0; idx < numOperands; ++idx) {
    if (idx == candidate.drivingOperandIdx)
      continue;

    SmallVector<Value> outerValues;
    outerValues.reserve(numLaunches);
    for (size_t launch = 0; launch < numLaunches; ++launch)
      outerValues.push_back(candidate.consumerOps[launch]->getOperand(idx));

    // TODO: Allow captured value from outside the parallel's body.
    if (llvm::all_equal(outerValues))
      return std::nullopt;
    if (!canBundlePerLaunch(outerValues, launchDevices))
      return std::nullopt;
    plan.bundles.push_back(std::move(outerValues));
    plan.operandToBlockArg[idx] = plan.bundles.size() - 1;
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

/// Replaces the consumers with a mo.parallel after \p candidate.unbundleOp
/// whose body is one clone of the representative consumer.
static void wrapCandidate(WrapCandidate &candidate, const InputPlan &plan,
                          OpBuilder &builder) {
  Location loc = candidate.unbundleOp.getLoc();
  builder.setInsertionPointAfter(candidate.unbundleOp);
  Operation *representative = candidate.consumerOps[0];

  SmallVector<Value> inputs;
  for (ArrayRef<Value> values : plan.bundles) {
    SmallVector<Type> elemTypes =
        llvm::map_to_vector(values, [](Value v) { return v.getType(); });
    auto bundleTy = BundleType::get(builder.getContext(), elemTypes);
    inputs.push_back(
        TensorBundleOp::create(builder, loc, bundleTy, values).getResult());
  }

  SmallVector<Type> outputElemTypes =
      llvm::map_to_vector(candidate.consumerOps, [](Operation *op) {
        return op->getResult(0).getType();
      });
  auto outputBundleTy = BundleType::get(builder.getContext(), outputElemTypes);

  auto parallel =
      ParallelOp::create(builder, loc, inputs, TypeRange{outputBundleTy});

  Block &body = parallel.getBodyRegion().front();
  IRMapping mapping;
  for (auto [i, blockArgIdx] : llvm::enumerate(plan.operandToBlockArg))
    mapping.map(representative->getOperand(i), body.getArgument(blockArgIdx));
  auto bodyBuilder = OpBuilder::atBlockBegin(&body);
  Operation *cloned = bodyBuilder.clone(*representative, mapping);
  YieldOp::create(bodyBuilder, loc, ValueRange{cloned->getResult(0)});

  builder.setInsertionPointAfter(parallel);
  auto newUnbundle = TensorUnbundleOp::create(builder, loc, outputElemTypes,
                                              parallel.getResult(0));
  for (auto [i, op] : llvm::enumerate(candidate.consumerOps)) {
    op->getResult(0).replaceAllUsesWith(newUnbundle.getResult(i));
    op->erase();
  }

  // Keep the original unbundle: bundle #0 reads from it until the post-pass
  // canonicalize folds the pair away.
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

/// Visits every `TensorUnbundleOp` reachable from \p root and tries to wrap
/// its consumer set.  Returns true if at least one wrap fired.
static bool runOneIteration(Operation *root, OpBuilder &builder) {
  SmallVector<TensorUnbundleOp> unbundles;
  root->walk([&](TensorUnbundleOp op) { unbundles.push_back(op); });

  bool changed = false;
  for (TensorUnbundleOp unbundleOp : unbundles) {
    // Skip ops erased by an earlier wrap this iteration.
    if (!unbundleOp.getOperation()->getParentOp())
      continue;
    auto candidate = findCandidate(unbundleOp);
    if (!candidate)
      continue;
    auto inputBundleTy =
        cast<BundleType>(candidate->unbundleOp.getInput().getType());
    auto launchDevices = llvm::map_to_vector(
        inputBundleTy.getElementTypes(), [](Type t) -> DeviceRefAttr {
          return cast<TensorType>(t).getDeviceRef();
        });
    auto plan = planInputs(*candidate, launchDevices);
    if (!plan)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "create-parallel-from-unbundle: wrapping "
                            << candidate->consumerOps.size() << " peer(s) of "
                            << candidate->unbundleOp << "\n");
    wrapCandidate(*candidate, *plan, builder);
    changed = true;
  }
  return changed;
}

struct CreateParallelFromUnbundlePass
    : public impl::CreateParallelFromUnbundleBase<
          CreateParallelFromUnbundlePass> {
  using CreateParallelFromUnbundleBase::CreateParallelFromUnbundleBase;

  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    // Each wrap's new output unbundle may seed a downstream wrap.
    while (runOneIteration(getOperation(), builder)) {
    }
  }
};

} // namespace
