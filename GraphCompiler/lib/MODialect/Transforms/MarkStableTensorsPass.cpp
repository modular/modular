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

#include "GraphCompiler/Conversion/OperandInference.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "mo-mark-stable-tensors"

namespace M::MO {

#define GEN_PASS_DEF_MARKSTABLETENSORSPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace MO;

namespace {

//===----------------------------------------------------------------------===//
// MarkStableTensorsPass
//===----------------------------------------------------------------------===//

// Replaces each distributed collective with one bundled op per participating
// device. The distributed kernels launch on every device from a single host
// entry point, which has no device-graph form; the bundled kernels record a
// single device's launch and read their peers through stable address slots,
// so the collective partitioner can place each copy in its device's region.
static void replaceDistributedOp(Operation *op, IRRewriter &rewriter,
                                 ValueRange outputs, ValueRange outChains,
                                 ValueRange origOutputs, Value origOutChain) {
  Value outChain = ChainCreateOp::create(rewriter, op->getLoc(), outChains);
  rewriter.replaceAllUsesWith(origOutputs, outputs);
  rewriter.replaceAllUsesWith(origOutChain, outChain);
  rewriter.eraseOp(op);
}

static LogicalResult expandAllReduce(AllReduceSumOp allReduce) {
  IRRewriter rewriter(allReduce);
  Location loc = allReduce.getLoc();
  Value inChain = allReduce.getInChain();
  ValueRange origOutputs = allReduce.getOutputs();

  SmallVector<Value> outputs, outChains;
  for (auto [rank, origOutput] : llvm::enumerate(origOutputs)) {
    auto bundled = BundledAllReduceSumOp::create(
        rewriter, loc, origOutput.getType(), inChain.getType(),
        allReduce.getInputs(), allReduce.getSignalBuffers(), inChain,
        allReduce.getGroupSizeAttr(), rewriter.getI64IntegerAttr(rank));
    outputs.push_back(bundled.getOutput());
    outChains.push_back(bundled.getOutChain());
  }
  replaceDistributedOp(allReduce, rewriter, outputs, outChains, origOutputs,
                       allReduce.getOutChain());
  return success();
}

static LogicalResult expandBroadcast(DistributedBroadcastOp broadcast) {
  IRRewriter rewriter(broadcast);
  Location loc = broadcast.getLoc();
  Value inChain = broadcast.getInChain();
  ValueRange origOutputs = broadcast.getOutputs();

  // Every device's copy, the root's included, reads the root tensor; the
  // stable marking below routes that read through a slot.
  SmallVector<Value> outputs, outChains;
  for (auto [rank, origOutput] : llvm::enumerate(origOutputs)) {
    auto bundled = BundledBroadcastOp::create(
        rewriter, loc, origOutput.getType(), inChain.getType(),
        broadcast.getInput(), broadcast.getSignalBuffers(), inChain,
        broadcast.getRootAttr(), rewriter.getI64IntegerAttr(rank));
    outputs.push_back(bundled.getOutput());
    outChains.push_back(bundled.getOutChain());
  }
  replaceDistributedOp(broadcast, rewriter, outputs, outChains, origOutputs,
                       broadcast.getOutChain());
  return success();
}

static LogicalResult expandAllGather(AllGatherOp allGather) {
  ValueRange inputs = allGather.getInputs();
  auto numDevices = static_cast<int64_t>(inputs.size());
  int64_t groupSize = allGather.getGroupSize();
  if (groupSize != numDevices)
    return allGather.emitOpError()
           << "a device graph supports only a full-world allgather "
              "(group_size equal to the number of inputs, "
           << numDevices << "), but group_size is " << groupSize
           << ": a grouped allgather routes part of a shard through a paired "
              "group's buffers, which a device graph cannot express";
  if (allGather.getSliceRows())
    return allGather.emitOpError()
           << "a device graph has no per-device form of a row-sliced "
              "allgather (slice_rows)";

  IRRewriter rewriter(allGather);
  Location loc = allGather.getLoc();
  Value inChain = allGather.getInChain();

  // The distributed op's outputs are device-major: device `d` owns the run
  // `[d * N, (d + 1) * N)`, one gathered piece per peer.
  ValueRange origOutputs = allGather.getOutputs();
  SmallVector<Value> outputs, outChains;
  for (auto rank : llvm::seq(numDevices)) {
    auto types =
        llvm::map_to_vector(origOutputs.slice(rank * numDevices, numDevices),
                            [](Value v) { return v.getType(); });
    auto bundled =
        BundledAllGatherOp::create(rewriter, loc, types, inChain.getType(),
                                   inputs, allGather.getSignalBuffers(),
                                   inChain, rewriter.getI64IntegerAttr(rank));
    llvm::append_range(outputs, bundled.getOutputs());
    outChains.push_back(bundled.getOutChain());
  }
  replaceDistributedOp(allGather, rewriter, outputs, outChains, origOutputs,
                       allGather.getOutChain());
  return success();
}

static LogicalResult expandDistributedCollectives(GraphOp graphOp) {
  auto result = graphOp.walk([](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<DistributedBroadcastOp>(expandBroadcast)
        .Case<AllReduceSumOp>(expandAllReduce)
        .Case<AllGatherOp>(expandAllGather)
        .Default([](auto) { return success(); });
  });
  return success(!result.wasInterrupted());
}

struct MarkStableTensorsPass final
    : public impl::MarkStableTensorsPassBase<MarkStableTensorsPass> {
  using Base::Base;

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    if (!graphOp.getIsDeviceGraph())
      return;

    auto moduleOp = graphOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      graphOp.emitError()
          << "device-graphable mo.graph has no parent ModuleOp; cannot "
             "look up kernel registrations";
      return signalPassFailure();
    }

    // Expand first: the bundled copies declare stable peers, so the walk
    // below is what routes their inputs through `mo.stable.create`.
    if (failed(expandDistributedCollectives(graphOp)))
      return signalPassFailure();

    MOGG::MojoRegistry registry(moduleOp);

    // For `mo.custom`, the kernel registration is named by the `symbol`
    // attribute; for built-in MO ops, the op name itself is the
    // registration string.
    auto getRegistrationName = [](Operation *op) {
      if (auto custom = dyn_cast<CustomOp>(op))
        return custom.getSymbol();
      return op->getName().getStringRef();
    };

    WalkResult result =
        graphOp.walk([&](Operation *op) {
          if (op->hasTrait<OpTrait::IsTerminator>())
            return WalkResult::advance();

          StringRef registration = getRegistrationName(op);

          if (!registry.containsKernel(registration))
            return WalkResult::advance();

          DeviceRefAttr device = MOGG::MojoRegistry::getDeviceRefForOp(op);
          auto kernelOr = registry.getResolvedKernel(registration, device);
          if (kernelOr.isError()) {
            op->emitError(kernelOr.getError());
            return WalkResult::interrupt();
          }

          MOGG::KernelDeclAdaptor adaptor(*kernelOr);
          if (!adaptor.buildGraphFunction)
            return WalkResult::advance();

          const MOGG::KernelDeclFunctionAdaptor &buildGraph =
              *adaptor.buildGraphFunction;
          if (llvm::none_of(buildGraph.inputArguments,
                            [](const MOGG::KernelDeclOperandAdaptor &arg) {
                              return arg.isStableTensorType();
                            }))
            return WalkResult::advance();

          auto mappingOr = MOGG::mapMOOpToKernelImplementation(op, buildGraph);
          if (mappingOr.isError()) {
            op->emitError()
                << "cannot map operands onto the `build_graph` "
                   "function of kernel '"
                << kernelOr->getSymName() << "': " << mappingOr.getError();
            return WalkResult::interrupt();
          }

          for (const auto &[index, arg] : mappingOr->operandMapping) {
            if (!arg.isStableTensorType())
              continue;

            OpOperand &operand = op->getOpOperand(index);

            auto tensor = dyn_cast<TensorValue>(operand.get());
            if (!tensor) {
              op->emitError()
                  << "operand at position " << index
                  << " feeds a stable tensor argument of the `build_graph` "
                     "function of kernel '"
                  << kernelOr->getSymName() << "' but is not a tensor";
              return WalkResult::interrupt();
            }

            if (tensor.getType().isStable())
              continue;

            OpBuilder builder(op);
            auto stable = StableCreateOp::create(builder, op->getLoc(), tensor);
            operand.assign(stable);
          }

          return WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
