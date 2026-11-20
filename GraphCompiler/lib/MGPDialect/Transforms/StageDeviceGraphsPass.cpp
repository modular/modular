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
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/IRMapping.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// Rewrite each `mgp.guard` in the model's execute block into a forwarding
/// `mogg.routine.async`.
///
/// `mgp.guard` lowers to a runtime primitive, not via EmitMojo, so staging
/// would leave it in the execute block. A guard between two staged regions
/// would then form a use-def cycle with the staging op; as a compiled region it
/// stages like any other.
///
/// The region forwards its non-chain inputs; the leading chain operand only
/// orders the region, preserving the guard's gating semantics.
void lowerGuardsToCompiledRegions(MLIRContext *ctx, ModelOp model) {
  auto guards = llvm::to_vector(model.getExecuteBlock()->getOps<GuardOp>());
  for (GuardOp guard : guards) {
    OpBuilder builder(guard);
    Location loc = guard.getLoc();

    ValueRange operands = guard->getOperands();

    // Only the forwarded values carry a device; in-device-graph guards forward
    // at least one.
    M::DeviceRefAttr device;
    for (Value result : guard.getResults())
      if (auto deviceType =
              dyn_cast<MGPHasDeviceTypeInterface>(result.getType())) {
        device = deviceType.getDevice();
        break;
      }

    SmallVector<M::InOutSignatureAttr::InOutSemantics> operandSemantics(
        operands.size(), M::InOutSignatureAttr::InOutSemantics::kNone);

    auto regionOp = AsyncRoutineOp::create(
        builder, loc, /*results=*/guard.getResultTypes(),
        /*operands=*/operands,
        /*outputParamDecls=*/KGEN::ParamDeclArrayAttr::get(ctx, {}),
        /*inOutSignature=*/M::InOutSignatureAttr::get(ctx, operandSemantics),
        /*fusionInfo=*/
        FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(guard)),
        /*dispatch=*/DispatchKindAttr::get(ctx, DispatchKind::Immediate),
        /*constantResources=*/builder.getArrayAttr({}),
        /*canFuseIntoOther=*/builder.getBoolAttr(false),
        /*canBeFusedInto=*/builder.getBoolAttr(false),
        /*canDuplicate=*/builder.getBoolAttr(false));
    if (device)
      regionOp.setDeviceRef(device);

    // Block args are `!mogg.async<T>`; the forwarded ones are yielded as-is and
    // the chain arg is unused.
    Block &body = regionOp.getBody().emplaceBlock();
    for (Value operand : operands)
      body.addArgument(AsyncType::get(operand.getType()), loc);
    builder.setInsertionPointToStart(&body);

    OutputOp::create(builder, loc, body.getArguments().drop_front());

    guard->replaceAllUsesWith(regionOp.getResults());
    guard.erase();
  }
}

void stageDeviceGraphOps(MLIRContext *ctx, ModelOp model) {
  lowerGuardsToCompiledRegions(ctx, model);

  Block *executeBlock = model.getExecuteBlock();

  /// Staged: every `mogg.routine.async` and every
  /// `mgp.device_graph.stable.create`. A stable create only has meaning inside
  /// device-graph execution, where the partitioner assigns it a slot, so none
  /// stays outside regardless of what produces or reads it. Block order
  /// preserves scheduling.
  llvm::SetVector<Operation *> stagedOps;
  for (Operation &op : *executeBlock)
    if (isa<AsyncRoutineOp, DeviceGraphStableCreateOp>(op))
      stagedOps.insert(&op);

  if (stagedOps.empty())
    return;

  /// Operands defined outside the staged set become the isolated staging op's
  /// `inputs`, entering as bare-`T` block arguments.
  llvm::SetVector<Value> freeValues;
  for (Operation *op : stagedOps) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || !stagedOps.contains(def))
        freeValues.insert(operand);
    }
  }

  auto inputs = freeValues.takeVector();

  /// Uses outside the staging op of values produced inside it.
  llvm::SetVector<OpOperand *> escapingUses;
  for (Operation *op : stagedOps) {
    for (OpResult result : op->getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (!stagedOps.contains(use.getOwner()))
          escapingUses.insert(&use);
      }
    }
  }

  auto locations = llvm::map_to_vector(
      stagedOps, [](Operation *op) { return op->getLoc(); });

  Location newLoc = FusedLoc::get(ctx, locations);

  auto buildGraphStagingBody = [&](OpBuilder &builder, Location loc, Value,
                                   ValueRange inputArgs) {
    /// Async pack/unpack at the seam is added later by
    /// `mgp-compute-device-graph-dependencies`.
    IRMapping freeMap;
    freeMap.map(inputs, inputArgs);

    for (Operation *op : stagedOps)
      op->moveBefore(builder.getBlock(), builder.getBlock()->end());

    for (Operation *op : stagedOps)
      for (OpOperand &operand : op->getOpOperands())
        if (Value mapped = freeMap.lookupOrNull(operand.get()))
          operand.set(mapped);

    builder.setInsertionPointToEnd(builder.getBlock());

    /// One operand per escaping use; canonicalization dedups.
    auto yieldValues = llvm::map_to_vector(
        escapingUses, [](OpOperand *use) { return use->get(); });

    DeviceGraphStagingYieldOp::create(builder, newLoc, yieldValues);
  };

  /// Anchor on the terminator, not a staged op: the callback moves every
  /// staged op into the body, which would detach the restored insertion point.
  OpBuilder builder(model.getExecuteBlock()->getTerminator());

  auto stagingOp = DeviceGraphStagingOp::create(builder, newLoc, inputs,
                                                buildGraphStagingBody);

  for (auto [oldUse, newValue] :
       llvm::zip_equal(escapingUses, stagingOp.getResults()))
    oldUse->set(newValue);
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_STAGEDEVICEGRAPHSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class StageDeviceGraphsPass
    : public impl::StageDeviceGraphsPassBase<StageDeviceGraphsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();

    if (model.getIsDeviceGraph())
      stageDeviceGraphOps(&getContext(), model);
  }
};

} // namespace
} // namespace M::MGP
