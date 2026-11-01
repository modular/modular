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
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace M::MGP;
using namespace M::MO;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Each stable create op's slot index.
using StableInfos = llvm::MapVector<DeviceGraphStableCreateOp, unsigned>;

/// Stable create ops and their assigned stable addresses.
struct StableAddrState {
  StableAddrState(const StableInfos &infos, Value builderArg)
      : indices(infos), builderArg(builderArg) {}

  /// The slot index assigned to `create`.
  unsigned index(DeviceGraphStableCreateOp create) {
    return indices.find(create)->second;
  }

  /// The stable tensor standing in for `operand` in this region, or null when
  /// `operand` is not a `stable.create` result. A region that did not publish
  /// the slot reads it with `stable.get` and rebuilds the tensor spec from the
  /// create's runtime dims, resolved through `mapper`: the slot holds only a
  /// data pointer, and the source tensor that knows the shape lives in another
  /// region. Fails when a runtime dim is not available in this region.
  FailureOr<Value> get(OpBuilder &builder, Location loc, Value operand,
                       const IRMapping &mapper) {
    auto stable = operand.getDefiningOp<DeviceGraphStableCreateOp>();
    if (!stable)
      return Value();

    Value &tensor = stableTensors[stable];
    if (tensor)
      return tensor;

    SmallVector<Value> runtimeDims;
    for (auto [dim, value] : llvm::enumerate(stable.getRuntimeDims())) {
      Value mapped = mapper.lookupOrNull(value);
      if (!mapped)
        return stable.emitOpError("runtime dimension ")
               << dim
               << " is produced in another device's region; a slot carries "
                  "only the data pointer, so a consumer's shape must come "
                  "from values every region can see";
      runtimeDims.push_back(mapped);
    }

    TensorSpecType specType = stable.getTensor().getType().getTensorSpec();
    Value slot = DeviceGraphStableGetOp::create(
        builder, loc, stable.getStableType(), builderArg, index(stable));
    Value spec =
        TensorSpecCreateOp::create(builder, loc, specType, runtimeDims);
    tensor = DeviceGraphStableTensorOp::create(builder, loc, slot, spec);
    return tensor;
  }

  /// Publishes `source`, this region's copy of `stable`'s operand, into the
  /// slot. The place returns the slot sequenced after the publish, so the
  /// region's consumers read through it rather than a redundant `stable.get`.
  void place(OpBuilder &builder, Location loc, DeviceGraphStableCreateOp stable,
             Value source) {
    Value placed = DeviceGraphStablePlaceOp::create(builder, loc, builderArg,
                                                    source, index(stable));
    Value spec = TensorExtractTensorSpecOp::create(builder, loc, source);
    stableTensors[stable] =
        DeviceGraphStableTensorOp::create(builder, loc, placed, spec);
  }

private:
  /// Slot index of each DeviceGraphStableCreateOp, shared by every region.
  const StableInfos &indices;

  /// This region's `!mgp.device_graph.builder` block argument.
  Value builderArg;

  /// The stable tensor each `stable.create` resolves to in this region.
  llvm::SmallDenseMap<DeviceGraphStableCreateOp, Value> stableTensors;
};

M::DeviceRefAttr getDevice(AsyncRoutineOp op) {
  return cast<MOHasDeviceInterface>(op.getOperation()).getExecutionDevice();
}

/// Collects the compiled regions and stable creates, giving each stable slot a
/// unique index.
LogicalResult
collectOperations(DeviceGraphStagingOp staging, StableInfos &stables,
                  SmallVectorImpl<AsyncRoutineOp> &compiledRegions) {
  Block &body = staging.getBodyBlock();

  for (Operation &op : body.without_terminator()) {
    if (auto stable = dyn_cast<DeviceGraphStableCreateOp>(&op)) {
      stables.try_emplace(stable, static_cast<unsigned>(stables.size()));
    } else if (auto region = dyn_cast<AsyncRoutineOp>(&op)) {
      compiledRegions.push_back(region);
    } else {
      // Unknown ops inside a staging body are unexpected after staging.
      return op.emitOpError(
          "unexpected operation inside device-graph staging body");
    }
  }

  return success();
}

SmallVector<M::DeviceRefAttr>
collectDevices(ArrayRef<AsyncRoutineOp> compiledRegions) {
  llvm::SmallSetVector<M::DeviceRefAttr, 8> deviceOrder;

  for (auto region : compiledRegions) {
    // CPU regions are host-side bookkeeping (e.g. wrapping pointers into
    // !mgp.tensor values), duplicated into each accelerator region.
    if (M::MO::isOnHost(region))
      continue;

    deviceOrder.insert(getDevice(region));
  }

  SmallVector<M::DeviceRefAttr> devices = deviceOrder.takeVector();
  llvm::sort(devices, [](M::DeviceRefAttr a, M::DeviceRefAttr b) {
    return std::tuple(a.getLabel(), a.getId()) <
           std::tuple(b.getLabel(), b.getId());
  });

  return devices;
}

/// Partitions `mgp.device_graph.staging` into a `mgp.device_graph.collective`
/// with one region per device.
LogicalResult partitionStagingOp(DeviceGraphStagingOp stagingOp,
                                 IRRewriter &rewriter) {
  Location loc = stagingOp.getLoc();

  // The collective is not a graph region, so toposort first.
  mlir::sortTopologically(&stagingOp.getBodyBlock());

  // The body is isolated, so enclosing values are read through its block args.
  IRMapping baseMapper;
  baseMapper.map(stagingOp.getBodyInputs(), stagingOp.getInputs());

  StableInfos stableIndices;
  SmallVector<AsyncRoutineOp> compiledRegions;

  if (failed(collectOperations(stagingOp, stableIndices, compiledRegions)))
    return failure();

  SmallVector<M::DeviceRefAttr> deviceOrder = collectDevices(compiledRegions);
  if (deviceOrder.empty())
    return stagingOp.emitOpError(
        "staging body has no compiled regions; nothing to partition");

  SmallVector<StableAddressType> stableSlotTypes = llvm::map_to_vector(
      stableIndices.keys(),
      [&](DeviceGraphStableCreateOp stable) { return stable.getStableType(); });

  // Staging yield operand -> result index in the collective.
  // The collective's results are every region's yields in region order,
  // so `resultCounter` advances once per region yield.
  // Host work is cloned into every region, so a host value can
  // be yielded more than once; the staging result maps to its first copy.
  llvm::SmallDenseMap<Value, unsigned> resultMapping;
  unsigned resultCounter = 0;

  LogicalResult bodyStatus = success();

  // Build the collective with a per-device body callback.  The callback replays
  // the staged ops for its device, lowers each stable.create to its slot ops,
  // and produces the region terminator.
  rewriter.setInsertionPoint(stagingOp);
  auto collective = DeviceGraphCollectiveOp::create(
      rewriter, loc, deviceOrder, stableSlotTypes,
      [&](OpBuilder &builder, M::DeviceRefAttr device,
          BlockArgument builderArg) {
        if (failed(bodyStatus))
          return;

        IRMapping mapper = baseMapper;

        StableAddrState stableState(stableIndices, builderArg);

        // Device graphs have no notion of cross-device execution-order
        // dependencies. Cross-device chains only have meaning in the presence
        // of distributed operations, which ensure synchronization with their
        // peers internally.
        auto remapRemoteChains = [&](AsyncRoutineOp compiledRegion) {
          for (Value operand : compiledRegion.getOperands()) {
            if (!isa<M::MEF::ChainType>(operand.getType()) ||
                mapper.contains(operand))
              continue;
            Value dummyChain = M::MOGG::ChainDummyOp::create(builder, loc);
            mapper.map(operand, dummyChain);
          }
        };

        for (Operation &op : stagingOp.getBodyBlock().without_terminator()) {
          if (auto stable = dyn_cast<DeviceGraphStableCreateOp>(&op)) {
            if (stable.getDevice() != device)
              continue;

            Value source = mapper.lookupOrNull(stable.getTensor());
            if (!source) {
              bodyStatus = stable.emitOpError(
                  "stable.create source was not materialized on its device");
              return;
            }

            stableState.place(builder, loc, stable, source);
            continue;
          }

          auto compiledRegion = dyn_cast<AsyncRoutineOp>(&op);
          if (!compiledRegion) {
            bodyStatus = op.emitOpError(
                "unexpected operation inside device-graph staging body");
            return;
          }

          auto opDev = getDevice(compiledRegion);
          bool isHostOp = M::MO::isOnHost(compiledRegion);

          // Device work is placed only on its own region. Host work goes
          // wherever its operands are: bookkeeping that wraps one device's
          // buffers (tensor.create, extract.buffer) belongs to that device's
          // region alone, while host work over staging inputs lands in every
          // region. Cloning it elsewhere would leave operands pointing into
          // the staging body, which is erased once the collective is built.
          if (opDev != device && !isHostOp)
            continue;

          if (!isHostOp)
            remapRemoteChains(compiledRegion);

          // An operand is available here when it is a staging input (mapped
          // up front), a result already cloned into this region, or a stable
          // create the slot machinery resolves below.
          MutableArrayRef<OpOperand> opOperands =
              compiledRegion->getOpOperands();
          auto *unavailable =
              llvm::find_if_not(opOperands, [&](OpOperand &operand) {
                Value value = operand.get();
                return mapper.contains(value) ||
                       value.getDefiningOp<DeviceGraphStableCreateOp>();
              });
          if (unavailable != opOperands.end()) {
            if (isHostOp)
              continue;
            bodyStatus = compiledRegion.emitOpError("operand ")
                         << unavailable->getOperandNumber()
                         << " is produced in another device's region and "
                            "does not go through a stable address slot";
            return;
          }

          // Resolve the stable values this region consumes to their
          // per-region stable tensors before cloning against them.
          for (Value operand : compiledRegion.getOperands()) {
            FailureOr<Value> resolved =
                stableState.get(builder, loc, operand, mapper);
            if (failed(resolved)) {
              bodyStatus = failure();
              return;
            }
            if (*resolved)
              mapper.map(operand, *resolved);
          }

          auto cloned =
              cast<AsyncRoutineOp>(builder.clone(*compiledRegion, mapper));
          mapper.map(compiledRegion.getResults(), cloned.getResults());
        }

        // CPU work is in every region, so its yields are available everywhere;
        // device work yields only from its own region.
        SmallVector<Value> regionYields;
        for (Value origYield : stagingOp.getYieldOp().getOperands()) {
          auto producer = origYield.getDefiningOp<AsyncRoutineOp>();
          if (!producer) {
            bodyStatus = stagingOp.getYieldOp().emitOpError(
                "staging yield value is not produced by a compiled region");
            return;
          }

          Value cloned = mapper.lookupOrNull(origYield);
          if (!cloned)
            continue;

          regionYields.push_back(cloned);
          resultMapping.try_emplace(origYield, resultCounter);
          ++resultCounter;
        }

        DeviceGraphCollectiveYieldOp::create(builder, loc, regionYields);
      });

  if (failed(bodyStatus)) {
    rewriter.eraseOp(collective);
    return failure();
  }

  // Host work only lands where its operands are, so a yield can end up with
  // no region producing it.
  for (Value yielded : stagingOp.getYieldOp().getOperands()) {
    if (resultMapping.contains(yielded))
      continue;
    rewriter.eraseOp(collective);
    return stagingOp.getYieldOp().emitOpError(
        "staging yield value was not materialized in any device region");
  }

  for (auto [yielded, result] : llvm::zip(stagingOp.getYieldOp().getOperands(),
                                          stagingOp.getResults())) {
    auto collectiveResult = collective.getResult(resultMapping.at(yielded));
    rewriter.replaceAllUsesWith(result, collectiveResult);
  }

  rewriter.eraseOp(stagingOp);
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_PARTITIONDEVICEGRAPHCOLLECTIVEPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class PartitionDeviceGraphCollectivePass
    : public impl::PartitionDeviceGraphCollectivePassBase<
          PartitionDeviceGraphCollectivePass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();

    if (!model.getIsDeviceGraph())
      return;

    IRRewriter rewriter(&getContext());

    auto stagingOps = llvm::to_vector(
        model.getExecuteBlock()->getOps<DeviceGraphStagingOp>());

    for (DeviceGraphStagingOp staging : stagingOps)
      if (failed(partitionStagingOp(staging, rewriter)))
        signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
