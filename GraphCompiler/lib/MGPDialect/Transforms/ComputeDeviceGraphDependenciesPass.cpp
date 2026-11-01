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
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Reconciles `v` to `target` with one `mogg.async` pack/unpack at the
/// dissolved region boundary: the verbatim body still expects
/// `!mogg.async<T>` where the outer values are plain `T`, and vice versa for
/// results. Emits an error and returns nullptr if one wrap/unwrap cannot
/// bridge the types.
Value reconcileAcrossBoundary(OpBuilder &builder, Location loc, Value value,
                              Type target, Operation *errorOp) {
  Type valueType = value.getType();
  if (valueType == target)
    return value;

  if (auto targetAsync = dyn_cast<AsyncType>(target);
      targetAsync && targetAsync.getElementType() == valueType)
    return AsyncPackOp::create(builder, loc, target, value);

  if (auto vAsync = dyn_cast<AsyncType>(valueType);
      vAsync && vAsync.getElementType() == target)
    return AsyncUnpackOp::create(builder, loc, target, value);

  errorOp->emitError("device-graph dependency conversion: cannot reconcile ")
      << valueType << " with " << target
      << " across the region boundary (expected an async wrap/unwrap)";

  return nullptr;
}

/// Populates a `mgp.device_graph.dependencies` body with the verbatim contents
/// of `region`, dissolving its dataflow boundary. Block args map to the outer
/// values (`oldValueToNew` redirects results of already-converted siblings),
/// with `mogg.async` pack/unpack at the seam. Device-context block args are
/// re-sourced from `graphBuilder` via `builtin.unrealized_conversion_cast`, so
/// the body is tied to the device graph under construction.
LogicalResult buildDependenciesBody(OpBuilder &builder, Location loc,
                                    Value graphBuilder, AsyncRoutineOp region,
                                    const IRMapping &oldValueToNew) {
  Block &srcBlock = region.getBody().front();

  // Created lazily, one per context type, so context-free bodies emit no cast.
  // Resolved once the recording stub accepts a `!mgp.device_graph.builder`.
  llvm::SmallDenseMap<Type, Value> deviceContextCast;

  // Other outer values are referenced directly; the op is not
  // `IsolatedFromAbove`.
  IRMapping mapper;
  for (auto [blockArg, operand] :
       llvm::zip(srcBlock.getArguments(), region.getOperands())) {
    // The launch context (`!mogg.async<!mgp.device_context>` in the region
    // ABI) is re-sourced from the builder; the re-pack here folds away against
    // the body's `mogg.async.unpack`.
    Type elementType = blockArg.getType();
    if (auto async = llvm::dyn_cast<AsyncType>(elementType))
      elementType = async.getElementType();

    auto ctxType = dyn_cast<DeviceContextType>(elementType);
    if (ctxType && ctxType.getDevice().getLabel() != M::kCPULabel) {
      Value &cast = deviceContextCast[elementType];
      if (!cast)
        cast = UnrealizedConversionCastOp::create(builder, loc, elementType,
                                                  graphBuilder)
                   .getResult(0);
      Value mapped = reconcileAcrossBoundary(builder, loc, cast,
                                             blockArg.getType(), region);
      if (!mapped)
        return failure();
      mapper.map(blockArg, mapped);
      continue;
    }
    Value src = oldValueToNew.lookupOrDefault(operand);
    Value mapped =
        reconcileAcrossBoundary(builder, loc, src, blockArg.getType(), region);
    if (!mapped)
      return failure();
    mapper.map(blockArg, mapped);
  }

  for (Operation &op : srcBlock.without_terminator())
    builder.clone(op, mapper);

  // Unpack the async-packed `mogg.output` operands to the region result types.
  OutputOp outputOp = region.getBodyTerminator();
  SmallVector<Value> yielded;
  yielded.reserve(outputOp.getNumOperands());
  for (auto [operand, resultType] :
       llvm::zip(outputOp.getOperands(), region.getResultTypes())) {
    Value mapped = reconcileAcrossBoundary(
        builder, loc, mapper.lookupOrDefault(operand), resultType, region);
    if (!mapped)
      return failure();
    yielded.push_back(mapped);
  }
  DeviceGraphDependenciesYieldOp::create(builder, loc, yielded);
  return success();
}

/// The value whose producer orders a consumer of `operand`. `stable.place`
/// lives on the same device as the producer of the tensor being placed.
/// Ordering must be preserved on the same device, so derive the ordering source
/// from the producer of the placed tensor.
Value orderingSource(Value operand) {
  if (auto stableTensor = operand.getDefiningOp<DeviceGraphStableTensorOp>())
    if (auto place =
            stableTensor.getAddr().getDefiningOp<DeviceGraphStablePlaceOp>())
      return place.getTensor();
  return operand;
}

/// Replaces every `mogg.routine.async` in `collectiveOp` with a
/// `mgp.device_graph.dependencies` op, preserving per-device ordering.
LogicalResult convertCollectiveRegions(DeviceGraphCollectiveOp collectiveOp) {
  llvm::MapVector<AsyncRoutineOp, DeviceGraphDependenciesOp> oldToNew;
  IRMapping oldValueToNew;

  // Regions are converted per device; edges, including cross-device ones, are
  // derived below from the shared `oldToNew` map.
  for (DeviceGraphCollectiveOp::CollectiveRegion graph :
       collectiveOp.graphs()) {
    Value regionBuilder = graph.builder;

    for (auto region : graph.getOps<AsyncRoutineOp>()) {
      OpBuilder builder(region);

      LogicalResult bodyResult = success();
      auto depsOp = DeviceGraphDependenciesOp::create(
          builder, region.getLoc(), /*graphBuilder=*/regionBuilder,
          /*dependencies=*/ValueRange{},
          [&](OpBuilder &bodyBuilder, Location loc, Value graphBuilder) {
            bodyResult = buildDependenciesBody(bodyBuilder, loc, graphBuilder,
                                               region, oldValueToNew);
          });

      if (failed(bodyResult))
        return failure();

      oldToNew[region] = depsOp;
      oldValueToNew.map(region.getResults(), depsOp.getBodyResults());
    }
  }

  for (auto [region, dependencies] : oldToNew) {
    llvm::SmallSetVector<Value, 8> edges;
    for (Value operand : region.getOperands()) {
      Operation *def = orderingSource(operand).getDefiningOp();
      if (auto producer = dyn_cast_if_present<AsyncRoutineOp>(def))
        if (auto it = oldToNew.find(producer); it != oldToNew.end())
          edges.insert(it->second.getNode());
    }
    dependencies.getDependenciesMutable().assign(edges.getArrayRef());
  }

  for (auto [region, dependencies] : oldToNew) {
    for (auto [oldResult, newResult] :
         llvm::zip(region.getResults(), dependencies.getBodyResults()))
      oldResult.replaceAllUsesWith(newResult);
  }

  SetVector<Attribute> constantResources;
  for (auto region : oldToNew.keys()) {
    if (auto resources = region.getConstantResources())
      constantResources.insert_range(resources);
    region.erase();
  }

  Builder builder(collectiveOp->getContext());
  auto newAttr = builder.getArrayAttr(constantResources.getArrayRef());
  collectiveOp.setConstantResourcesAttr(newAttr);

  return success();
}

/// Converts the staged `mogg.routine.async` ops into
/// `mgp.device_graph.dependencies` ops, making the dependency DAG explicit.
LogicalResult computeDeviceGraphDependencies(ModelOp model) {
  Block *execute = model.getExecuteBlock();

  for (DeviceGraphCollectiveOp collectiveOp :
       llvm::make_early_inc_range(execute->getOps<DeviceGraphCollectiveOp>())) {
    if (failed(convertCollectiveRegions(collectiveOp)))
      return failure();
  }
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_COMPUTEDEVICEGRAPHDEPENDENCIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class ComputeDeviceGraphDependenciesPass
    : public impl::ComputeDeviceGraphDependenciesPassBase<
          ComputeDeviceGraphDependenciesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp theModel = getOperation();

    // Device-graph synthesis; runs after `mgp-stage-device-graphs`.
    if (theModel.getIsDeviceGraph())
      if (failed(computeDeviceGraphDependencies(theModel)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
