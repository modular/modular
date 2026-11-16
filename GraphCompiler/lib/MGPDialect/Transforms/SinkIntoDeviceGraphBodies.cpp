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
// Sinks the ops that feed only a device-graph build body back into it. This
// pass runs before outlining to improve compilation time by pushing more Mojo
// code into outlined functions.
//
// Without sinking, the device graph synthesis pipeline produces a single Mojo
// function with most of the Mojo code in the top of the function. The code was
// previously hoisted out of other region-carrying operations to improve CSE and
// canonicalization. The single, large Mojo function hurts compilation time, so
// this pass sinks single-use producers into MLIR blocks that will then be
// outlined, causing outlining to outline more code.
//
//   %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
//   %dim = mgp.tensor_spec.get_dim[0] %spec : <2x3xf32>
//   %shared = mgp.tensor.extract.tensor_spec %in : <2x3xf32, <"gpu", 0>>
//   %node = mgp.device_graph.dependencies builder(%b) dependencies() {
//   ^bb0(%builder: !mgp.device_graph.builder):
//     "test.use"(%dim) : (index) -> ()
//     "test.use_common"(%shared) : (!mgp.tensor_spec<2x3xf32>) -> ()
//     mgp.device_graph.dependencies.yield
//   }
//
// Here `%spec` and `%dim` are used only by this body, so they move back in,
// while `%shared` is shared with another body and stays outside:
//
//   %shared = mgp.tensor.extract.tensor_spec %in : <2x3xf32, <"gpu", 0>>
//   %node = mgp.device_graph.dependencies builder(%b) dependencies() {
//   ^bb0(%builder: !mgp.device_graph.builder):
//     %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
//     %dim = mgp.tensor_spec.get_dim[0] %spec : <2x3xf32>
//     "test.use"(%dim) : (index) -> ()
//     "test.use_common"(%shared) : (!mgp.tensor_spec<2x3xf32>) -> ()
//     mgp.device_graph.dependencies.yield
//   }
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace M;
using namespace M::MGP;
using namespace mlir;

namespace M::MGP {

#define GEN_PASS_DEF_SINKINTODEVICEGRAPHBODIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

} // namespace M::MGP

namespace {

/// Whether `op` may sink back into the body that consumes it.
///
/// A graph input registration cannot: the inputs are registered in
/// graph-signature order at the top of a build region, and sinking one past
/// another would reorder them.
///
/// Constants are not moved to avoid specializing the outlined function of
/// constant values to improve function merging in MergeIdenticalFuncs.
/// Constants instead become function arguments.
bool canSinkIntoBody(Operation *op) {
  return !isa<MGP::BufferConstantOp, MOGG::BufferConstantOp,
              DeviceGraphAddInputOp>(op) &&
         !op->hasTrait<mlir::OpTrait::ConstantLike>();
}

/// Move values consumed only within a DeviceGraphDependenciesOp into the
/// DeviceGraphDependenciesOp's body.
///
/// This pushes additional work into the DeviceGraphDependenciesOp. The goal is
/// to push more work into MLIR blocks that will then be outlined, better
/// distributing the code among the various Mojo functions and improving
/// compilation time.
void sinkOpsIntoBody(DeviceGraphDependenciesOp op) {
  Region &body = op.getBody();
  Block *bodyBlock = &op.getBodyBlock();
  Block *parent = op->getBlock();

  auto sinksHere = [&](Operation *candidate) {
    return candidate->getBlock() == parent && canSinkIntoBody(candidate) &&
           llvm::all_of(candidate->getResults(), [&](Value result) {
             return llvm::all_of(result.getUsers(), [&](Operation *user) {
               return body.isAncestor(user->getParentRegion());
             });
           });
  };

  SmallVector<Operation *> worklist;
  auto queueProducersOf = [&](Operation *owner) {
    for (Value operand : owner->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        worklist.push_back(producer);
  };

  llvm::SetVector<Value> captured;
  mlir::getUsedValuesDefinedAbove(body, captured);
  for (Value value : captured)
    if (Operation *producer = value.getDefiningOp())
      worklist.push_back(producer);

  // Each candidate sinks to the front, so a producer queued by its consumer's
  // move lands ahead of that consumer.
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    if (!sinksHere(candidate))
      continue;
    queueProducersOf(candidate);
    candidate->moveBefore(bodyBlock, bodyBlock->begin());
  }

  captured.clear();

  mlir::getUsedValuesDefinedAbove(body, captured);
  auto ownBuilder = op.getBuilderBlockArgument();
  for (Value capture : captured)
    if (isa<DeviceGraphBuilderType>(capture.getType()))
      replaceAllUsesInRegionWith(capture, ownBuilder, body);
}

} // namespace

namespace M::MGP {
namespace {

struct SinkIntoDeviceGraphBodiesPass
    : public impl::SinkIntoDeviceGraphBodiesPassBase<
          SinkIntoDeviceGraphBodiesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk<WalkOrder::PreOrder>([&](DeviceGraphDependenciesOp op) {
      // A body that yields values stays inline, and an empty body has nothing
      // to gather around; sinking into either would leave the region no more
      // outlinable than it already is.
      if (!op.getBodyResults().empty() ||
          op.getBodyBlock().without_terminator().empty())
        return WalkResult::skip();

      sinkOpsIntoBody(op);
      return WalkResult::skip();
    });
  }
};

} // namespace
} // namespace M::MGP
