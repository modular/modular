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
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

using namespace M;
using namespace MGP;
using namespace mlir;

namespace {

using DeviceContextValue = mlir::TypedValue<DeviceContextType>;

// Device graphs have no chains: staged chain inputs are dead pass-throughs and
// chain outputs come from host-side `mogg.async.ready`. Chains are dropped
// from the graph I/O; the execute op produces a fresh one.
bool isChain(Value v) { return isa<MEF::ChainType>(v.getType()); }

SmallVector<Value> nonChainValues(ValueRange v) {
  return llvm::filter_to_vector(v, [](Value v) { return !isChain(v); });
}

/// Replace the builder -> device-context bridges that survive synthesis with
/// `mgp.device_graph.device_context` ops.
///
/// Most such casts fold away once consumers take the builder directly, but one
/// that genuinely needs a device context (e.g. `mgp.buffer.bulk_slice`, which
/// does host-side sub-buffer math) keeps it live, and unrealized casts cannot
/// be emitted to Mojo. Round trips are folded first: converting one of a pair
/// early would strand its inverse.
LogicalResult resolveBuilderCasts(Region &region) {
  SmallVector<UnrealizedConversionCastOp> casts;
  region.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });

  SmallVector<UnrealizedConversionCastOp> remaining;
  mlir::reconcileUnrealizedCasts(casts, &remaining);

  for (UnrealizedConversionCastOp cast : remaining) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1 ||
        !isa<DeviceGraphBuilderType>(cast.getOperand(0).getType()))
      return cast.emitError()
             << "device-graph finalization: cannot resolve this cast; only a "
                "`!mgp.device_graph.builder` to `!mgp.device_context` bridge "
                "has a device-graph equivalent";

    auto deviceContextType =
        dyn_cast<DeviceContextType>(cast.getResult(0).getType());
    if (!deviceContextType)
      return cast.emitError()
             << "device-graph finalization: cannot resolve a cast from "
                "`!mgp.device_graph.builder` to "
             << cast.getResult(0).getType();

    // Work against the backing context runs eagerly at build time instead of
    // being recorded, so only ops audited to make host-side buffer views may
    // use it. Effect-freedom is not enough: `mgp.sync` is `Pure` yet records a
    // device event. Ops that record need a builder-native form or a
    // `mgp.device_graph.recording_context`.
    for (Operation *user : cast->getUsers())
      if (!isa<BufferSliceOp, BufferBulkSliceOp, TensorSliceOp>(user))
        return cast.emitError()
               << "device-graph finalization: '" << user->getName()
               << "' is not known to be safe against the graph builder's "
                  "backing device context (it may dispatch work, which would "
                  "run eagerly at build time instead of being recorded)";

    Value graphBuilder = cast.getOperand(0);
    IRRewriter rewriter(cast);
    rewriter.replaceOpWithNewOp<DeviceGraphDeviceContextOp>(
        cast, deviceContextType, graphBuilder);
  }

  return mlir::success();
}

/// Build a `mogg.routine.async` around a device-graph build callback.
///
/// `operands` enter as `!mogg.async<T>` block arguments and are unpacked for
/// `bodyBuilder`, whose returned values are packed and yielded.
static FailureOr<MOGG::AsyncRoutineOp> wrapDeviceGraphBuildRegion(
    IRRewriter &rewriter, Location loc, DeviceRefAttr deviceRef,
    TypeRange resultTypes, ValueRange operands, StringRef fusionName,
    ArrayAttr constantResources,
    llvm::function_ref<FailureOr<SmallVector<Value>>(OpBuilder &, Location,
                                                     ValueRange)>
        bodyBuilder) {
  MLIRContext *ctx = rewriter.getContext();

  SmallVector<InOutSignatureAttr::InOutSemantics> operandSemantics(
      operands.size(), InOutSignatureAttr::InOutSemantics::kNone);

  auto regionOp = MOGG::AsyncRoutineOp::create(
      rewriter, loc, resultTypes, operands,
      KGEN::ParamDeclArrayAttr::get(ctx, {}),
      InOutSignatureAttr::get(ctx, operandSemantics),
      MOGG::FusionInfoAttr::get(
          ctx, MOGG::FusionLeafAttr::get(ctx, StringAttr::get(ctx, fusionName),
                                         /*loc=*/{}, /*signature=*/{})),
      MOGG::DispatchKindAttr::get(ctx, MOGG::DispatchKind::Immediate),
      constantResources, rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
  regionOp.setDeviceRef(deviceRef);

  Block &regionBody = regionOp.getBody().emplaceBlock();
  for (Value operand : operands)
    regionBody.addArgument(MOGG::AsyncType::get(operand.getType()), loc);
  rewriter.setInsertionPointToStart(&regionBody);

  auto unpacked =
      llvm::map_to_vector(regionBody.getArguments(), [&](Value arg) -> Value {
        return MOGG::AsyncUnpackOp::create(rewriter, loc, arg);
      });

  FailureOr<SmallVector<Value>> outputs = bodyBuilder(rewriter, loc, unpacked);
  if (failed(outputs))
    return mlir::failure();

  auto packed = llvm::map_to_vector(*outputs, [&](Value out) -> Value {
    return MOGG::AsyncPackOp::create(rewriter, loc, out);
  });

  MOGG::OutputOp::create(rewriter, loc, packed);

  return regionOp;
}

/// Collect the values used inside a collective graph `region` that are defined
/// outside of it. Chain values (dead pass-throughs) are left to later work.
/// Stable address slots are not values: the regions name them by index
/// through the builder, so nothing here has to carry them.
static SmallVector<Value> collectRegionInputs(Region &region) {
  llvm::SetVector<Value> captured;
  mlir::getUsedValuesDefinedAbove(region, captured);
  return nonChainValues(captured.getArrayRef());
}

/// Whether `region` reads `value` only for its shape: every use inside it, at
/// any depth, is an `mgp.tensor.extract.tensor_spec`. By finalization the
/// dependency pass has inlined the host bookkeeping, so the uses are direct.
/// Such an input needs no stable copy in that region's graph; only its shape
/// enters the cache key.
static bool isShapeOnlyInRegion(Value value, Region &region) {
  bool used = false;
  for (OpOperand &use : value.getUses()) {
    if (!region.isAncestor(use.getOwner()->getParentRegion()))
      continue;
    used = true;
    if (!isa<TensorExtractTensorSpecOp>(use.getOwner()))
      return false;
  }
  return used;
}

/// The captures and non-chain yields of a `mgp.device_graph.collective` region.
struct RegionInterface {
  Region *region;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

static LogicalResult
createRegionInterfaces(DeviceGraphCollectiveOp collectiveOp,
                       SmallVectorImpl<RegionInterface> &interfaces) {
  interfaces.reserve(collectiveOp.getNumRegions());

  for (auto graph : collectiveOp.graphs()) {
    RegionInterface iface;
    iface.region = graph.bodyBlock->getParent();
    iface.inputs = collectRegionInputs(*iface.region);

    auto yieldOp =
        cast<DeviceGraphCollectiveYieldOp>(graph.bodyBlock->getTerminator());
    for (Value out : yieldOp.getOperands())
      if (!isChain(out))
        iface.outputs.push_back(out);

    interfaces.push_back(std::move(iface));
  }
  return mlir::success();
}

/// Map each device referenced by `collectiveOp` to the corresponding
/// execute-block device context in `model`.
static FailureOr<SmallVector<DeviceContextValue>>
collectDeviceContexts(DeviceGraphCollectiveOp collectiveOp, ModelOp model) {
  SmallVector<DeviceContextValue> deviceContexts;
  for (auto deviceRef : collectiveOp.getDevices()) {
    DeviceContextValue deviceContext;
    for (auto [attr, execDevice] :
         llvm::zip(model.getDevices(), model.getExecuteDevices())) {
      if (attr == deviceRef) {
        deviceContext = cast<DeviceContextValue>(execDevice);
        break;
      }
    }
    if (!deviceContext)
      return collectiveOp.emitError(
                 "device-graph finalization: no execute device context matches "
                 "device ")
             << deviceRef;
    deviceContexts.push_back(deviceContext);
  }
  return deviceContexts;
}

/// Lower a single `mgp.device_graph.collective` op into a
/// `mgp.device_graph.collective.create` wrapped in a
/// `mogg.routine.async`, plus a bare
/// `mgp.device_graph.collective.execute` that replays it.
LogicalResult finalizeCollectiveOp(DeviceGraphCollectiveOp collectiveOp,
                                   ModelOp model) {
  Location loc = collectiveOp.getLoc();
  MLIRContext *ctx = collectiveOp->getContext();
  IRRewriter rewriter(collectiveOp);

  Value theContext = model.getExecuteContext();

  auto deviceContextsResult = collectDeviceContexts(collectiveOp, model);
  if (failed(deviceContextsResult))
    return mlir::failure();

  SmallVector<DeviceContextValue> deviceContexts =
      std::move(*deviceContextsResult);

  SmallVector<RegionInterface> regionInterfaces;
  if (failed(createRegionInterfaces(collectiveOp, regionInterfaces)))
    return failure();

  SmallVector<Value> allInputs;
  SmallVector<unsigned> inputCounts;
  SmallVector<int64_t> shapeOnlyInputs;
  inputCounts.reserve(regionInterfaces.size());
  for (RegionInterface &info : regionInterfaces) {
    inputCounts.push_back(info.inputs.size());
    for (Value input : info.inputs) {
      if (isa<DeviceTensorType, DeviceBufferType>(input.getType()) &&
          isShapeOnlyInRegion(input, *info.region))
        shapeOnlyInputs.push_back(allInputs.size());
      allInputs.push_back(input);
    }
  }

  auto graphTypes =
      llvm::map_to_vector(regionInterfaces, [&](RegionInterface &info) -> Type {
        return DeviceGraphType::get(ctx, info.inputs, info.outputs);
      });

  // The context, device contexts, and graph inputs enter via the async ABI;
  // the region yields one built graph per device.
  SmallVector<Value> regionOperands{theContext};
  llvm::append_range(regionOperands, deviceContexts);
  llvm::append_range(regionOperands, allInputs);

  auto regionOpResult = wrapDeviceGraphBuildRegion(
      rewriter, loc, deviceContexts.front().getType().getDevice(),
      TypeRange(graphTypes), regionOperands, collectiveOp.getOperationName(),
      collectiveOp.getConstantResources(),
      [&](OpBuilder &bodyBuilder, Location bodyLoc,
          ValueRange unpacked) -> FailureOr<SmallVector<Value>> {
        Value contextUnpacked = unpacked[0];
        SmallVector<Value> deviceContextsUnpacked =
            llvm::to_vector(unpacked.slice(1, deviceContexts.size()));
        SmallVector<Value> inputsUnpacked =
            llvm::to_vector(unpacked.drop_front(1 + deviceContexts.size()));

        Value cache =
            DeviceGraphCacheOp::create(bodyBuilder, bodyLoc, contextUnpacked);

        auto createOp = DeviceGraphCollectiveCreateOp::create(
            bodyBuilder, bodyLoc, contextUnpacked, deviceContextsUnpacked,
            cache, inputsUnpacked, inputCounts,
            [&](OpBuilder &innerBuilder, Location innerLoc, unsigned regionIdx,
                Value graphBuilder, Value /*contextArg*/,
                ValueRange inputArgs) {
              RegionInterface &info = regionInterfaces[regionIdx];
              Region &oldRegion = *info.region;
              Block &oldBlock = oldRegion.front();

              IRMapping mapper;
              mapper.map(oldRegion.getArgument(0), graphBuilder);
              mapper.map(info.inputs, inputArgs);

              for (Operation &op : oldBlock.without_terminator())
                innerBuilder.clone(op, mapper);

              auto oldYield = oldBlock.getTerminator();
              SmallVector<Value> newOutputs;
              newOutputs.reserve(info.outputs.size());
              for (Value out : oldYield->getOperands()) {
                if (isChain(out))
                  continue;
                Value mapped = mapper.lookupOrDefault(out);
                newOutputs.push_back(mapped);
              }

              DeviceGraphYieldOp::create(innerBuilder, innerLoc, newOutputs);
            });

        // The slot table the regions index into is declared on the collective;
        // emission reads it off the create op to allocate the slots.
        if (StableAddressArrayAttr stableTypes =
                collectiveOp.getStableTypesAttr();
            !stableTypes.getValue().empty())
          createOp.setStableTypesAttr(stableTypes);

        if (!shapeOnlyInputs.empty())
          createOp.setShapeOnlyInputsAttr(
              mlir::DenseI64ArrayAttr::get(ctx, shapeOnlyInputs));

        for (Region &createRegion : createOp.getGraphs())
          if (failed(resolveBuilderCasts(createRegion)))
            return mlir::failure();

        return llvm::to_vector_of<Value>(createOp.getOutGraphs());
      });

  if (failed(regionOpResult))
    return mlir::failure();
  auto regionOp = *regionOpResult;

  SmallVector<Type> outputTypes;
  for (RegionInterface &info : regionInterfaces)
    llvm::append_range(outputTypes, TypeRange(info.outputs));

  rewriter.setInsertionPoint(collectiveOp);
  auto executeOp = DeviceGraphCollectiveExecuteOp::create(
      rewriter, loc, /*chain=*/MEF::ChainType::get(ctx),
      /*outputs=*/TypeRange(outputTypes),
      /*graphs=*/regionOp.getResults(),
      /*inputs=*/allInputs);

  Value executeChain = executeOp.getChain();
  unsigned outputIdx = 0;
  for (Value result : collectiveOp.getResults()) {
    if (isChain(result))
      result.replaceAllUsesWith(executeChain);
    else
      result.replaceAllUsesWith(executeOp.getOutputs()[outputIdx++]);
  }

  rewriter.eraseOp(collectiveOp);
  return mlir::success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_FINALIZEDEVICEGRAPHSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class FinalizeDeviceGraphsPass
    : public impl::FinalizeDeviceGraphsPassBase<FinalizeDeviceGraphsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();
    if (!model.getIsDeviceGraph())
      return;

    for (DeviceGraphCollectiveOp collectiveOp : llvm::to_vector(
             model.getExecuteBlock()->getOps<DeviceGraphCollectiveOp>()))
      if (failed(finalizeCollectiveOp(collectiveOp, model)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace M::MGP
