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

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_LOWERGRAPHBLOBS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Adds the ops the kernel runs to produce `value` to `producers`, each after
// the producers of its operands, and the values they take from the host to
// `crossing`. Fails on a value the host builds that can't cross the launch.
LogicalResult collectOperandProducers(Value value, Operation *call,
                                      llvm::SetVector<Operation *> &producers,
                                      llvm::SetVector<Value> &crossing) {
  Operation *def = value.getDefiningOp();
  // A routine argument crosses.
  if (!def) {
    crossing.insert(value);
    return success();
  }
  auto operand = dyn_cast<LaunchOperandInterface>(def);
  switch (operand ? operand.getLaunchPlacement() : LaunchPlacement::HostOnly) {
  case LaunchPlacement::Kernel:
    // The kernel rebuilds the value from the op's operands.
    if (producers.contains(def))
      return success();
    assert(def->getNumRegions() == 0 &&
           "the kernel rebuilds an op from its operands alone");
    for (Value input : def->getOperands()) {
      if (failed(collectOperandProducers(input, call, producers, crossing)))
        return failure();
    }
    producers.insert(def);
    return success();
  case LaunchPlacement::Argument:
    // The host builds the value, and it crosses.
    crossing.insert(value);
    return success();
  case LaunchPlacement::HostOnly: {
    // The host builds the value, and the kernel can't take it.
    InFlightDiagnostic diag = call->emitOpError(
        "reads a value that the host builds and a graph blob launch can't "
        "take");
    diag.attachNote(def->getLoc()) << "built here";
    return diag;
  }
  }
  llvm_unreachable("unknown launch placement");
}

// Returns the values `call` reads from the blob's body, other than its device
// context: its kernel operands, or a loop's shapes and captures and the values
// its body reads from outside it.
SmallVector<Value> readValues(Operation *call) {
  if (auto opaque = dyn_cast<CallOpaqueOp>(call)) {
    SmallVector<Value> values(opaque.getInputs());
    llvm::append_range(values, opaque.getOutputs());
    return values;
  }
  auto foreach = cast<CallForeachOp>(call);
  llvm::SetVector<Value> values;
  values.insert(foreach.getShape());
  values.insert(foreach.getCapturing().begin(), foreach.getCapturing().end());
  if (Value dims = foreach.getDynamicMaxSimdWidth())
    values.insert(dims);
  getUsedValuesDefinedAbove(foreach->getRegions(), values);
  return values.takeVector();
}

Value deviceContext(Operation *call) {
  if (auto foreach = dyn_cast<CallForeachOp>(call))
    return foreach.getDeviceContext();
  return cast<CallOpaqueOp>(call).getDeviceContexts().front();
}

// Creates `call` in a `mogg.graph_blob.call`, reading the values `mapping`
// gives for its operands.
void createDeviceCall(OpBuilder &builder, Operation *call, IRMapping &mapping) {
  auto mapped = [&](ValueRange values) {
    return llvm::map_to_vector(
        values, [&](Value value) { return mapping.lookup(value); });
  };
  if (auto opaque = dyn_cast<CallOpaqueOp>(call)) {
    CallExecuteDeviceOp::create(
        builder, opaque.getLoc(), opaque.getCalleeAttr(),
        mapped(opaque.getInputs()), mapped(opaque.getOutputs()),
        opaque.getKernelParamAttr());
    return;
  }
  auto foreach = cast<CallForeachOp>(call);
  Value dims = foreach.getDynamicMaxSimdWidth();
  auto lowered = GraphBlobForeachOp::create(
      builder, foreach.getLoc(), mapping.lookup(foreach.getShape()),
      mapped(foreach.getCapturing()), dims ? mapping.lookup(dims) : Value(),
      foreach.getMaxSimdWidthAttr());
  foreach
    .getBody().cloneInto(&lowered.getBody(), mapping);
}

LogicalResult lowerBlob(GraphBlobOp blob) {
  Block &body = blob.getBody().front();
  SmallVector<Operation *> calls;
  for (Operation &op : body) {
    if (op.hasTrait<CallLike>())
      calls.push_back(&op);
  }

  // Find what each call's region builds and what crosses the launch, before
  // changing any IR.
  llvm::SetVector<Value> crossing;
  SmallVector<llvm::SetVector<Operation *>> producers(calls.size());
  for (auto [call, callProducers] : llvm::zip_equal(calls, producers)) {
    for (Value value : readValues(call)) {
      if (failed(collectOperandProducers(value, call, callProducers, crossing)))
        return failure();
    }
  }

  // Move all the host code to the top, leaving only the calls at the end.
  for (Operation &op : llvm::make_early_inc_range(body)) {
    if (!op.hasTrait<CallLike>())
      op.moveBefore(blob);
  }

  MLIRContext *ctx = blob.getContext();
  Location loc = blob.getLoc();
  OpBuilder builder(blob);
  Type configType = LaunchConfigType::get(ctx);
  Value context = deviceContext(calls.front());

  // Generate a call to `execute_configuration` on the host for each
  // declarative call, passing the same inputs and outputs.
  auto configs = llvm::map_to_vector(calls, [&](Operation *call) -> Value {
    auto opaque = dyn_cast<CallOpaqueOp>(call);
    if (!opaque)
      return Value();
    return CallExecuteConfigurationOp::create(
        builder, opaque.getLoc(), configType, opaque.getCalleeAttr(),
        opaque.getInputs(), opaque.getOutputs(), opaque.getKernelParamAttr());
  });

  // Get the whole-device launch configuration and each call's part of it. A
  // loop, which has no configuration, runs on all of it.
  Value launchConfig =
      GraphBlobLaunchConfigOp::create(builder, loc, configType, context);
  SmallVector<Value> args(crossing.begin(), crossing.end());
  for (Value config : configs)
    args.push_back(GraphBlobActiveThreadsOp::create(
        builder, loc, builder.getI32Type(), config ? config : launchConfig,
        launchConfig));

  // Launch one kernel that takes the crossing values, then the active counts.
  auto launch =
      GraphBlobLaunchOp::create(builder, loc, context, launchConfig, args);
  auto argLocs =
      llvm::map_to_vector(args, [](Value arg) { return arg.getLoc(); });
  Block *kernel = builder.createBlock(&launch.getBody(), {},
                                      ValueRange(args).getTypes(), argLocs);
  for (auto [index, call] : llvm::enumerate(calls)) {
    // Create a barrier between each call.
    if (index > 0)
      GraphBlobBarrierOp::create(builder, loc);

    // Run the call on its active threads.
    auto region = GraphBlobCallOp::create(
        builder, call->getLoc(), kernel->getArgument(crossing.size() + index));
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&region.getBody());

    // Rebuild the call's operands from the launch's arguments. Each call
    // builds its own, so no value is shared across calls.
    IRMapping mapping;
    mapping.map(crossing.getArrayRef(),
                kernel->getArguments().take_front(crossing.size()));
    for (Operation *op : producers[index])
      builder.clone(*op, mapping);

    createDeviceCall(builder, call, mapping);
  }

  blob.erase();
  return success();
}

struct LowerGraphBlobsPass
    : public impl::LowerGraphBlobsBase<LowerGraphBlobsPass> {
  void runOnOperation() override {
    // A post-order walk may erase the blob it visits.
    WalkResult result = getOperation()->walk([](GraphBlobOp blob) {
      return failed(lowerBlob(blob)) ? WalkResult::interrupt()
                                     : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
