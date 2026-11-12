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
// Device-graph ops of the MGP dialect, declared in MGPDeviceGraphOps.td.
// This file holds their verifiers.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MGP;

using namespace M;
using namespace MGP;

/// Host tensor inputs are assumed to encode launch parameters, so their
/// contents join the cache key. Warn on large ones, which bloat the cache,
/// until there is a knob to control this.
static constexpr uint64_t kHostInputWarnBytes = 1024;

/// Verifies the residency rules for the graph's tensor/buffer inputs.
///
/// - Device-resident tensors are copied into device-pool stable locations (or
///   recorded in place when `mut`).
///
/// - Host-placed *immutable* tensors are allowed: they get a host-pool stable
///   location populated before recording, and their contents join the cache
///   key, because recorded ops read host values at enqueue time.
///
/// - Host-placed `mut` tensors are rejected (in-place writes would invalidate
///   their own cache entry), as are host-placed raw buffers, which should
///   already be lowered to mut tensors.
static LogicalResult verifyInputResidency(Operation *op, OperandRange inputs) {
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    DeviceRefAttr device = graphInputDevice(input);
    if (!device || !M::isHostPlaced(device))
      continue;

    auto tensorType = dyn_cast<MGP::DeviceTensorType>(input.getType());
    if (!tensorType)
      return op->emitOpError("input #")
             << idx << " is a buffer on " << device.toString()
             << ", but host-placed device graph inputs must be immutable "
                "tensors";
    if (tensorType.getMut())
      return op->emitOpError("input #")
             << idx << " is a mutable tensor on " << device.toString()
             << "; in-place writes contradict the value-keyed replay of a "
                "host-placed input";

    std::optional<uint64_t> byteSize =
        tensorType.getTensorSpec().getSizeInBytes();
    if (!byteSize)
      continue; // Dynamic shape: the runtime keys whatever size shows up.
    if (*byteSize > kHostInputWarnBytes)
      op->emitWarning("input #")
          << idx << " is a host tensor of " << *byteSize
          << " bytes; its contents are baked into the graph cache key, so "
             "every distinct value records a new graph";
  }
  return success();
}

LogicalResult DeviceGraphDependenciesOp::verify() {
  // The sole block argument is the `!mgp.device_graph.builder` that the body's
  // device-context uses are re-sourced from.
  Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("body has ")
           << body.getNumArguments()
           << " block argument(s), expected exactly 1 (the "
              "`!mgp.device_graph.builder`)";
  if (!llvm::isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();

  // The yield's operands must match this op's `bodyResults`.
  DeviceGraphDependenciesYieldOp yieldOp = getYieldOp();
  if (!llvm::equal(yieldOp.getOperands().getTypes(),
                   getBodyResults().getTypes()))
    return emitOpError(
        "`mgp.device_graph.dependencies.yield` operand types must "
        "match the op's body result types");
  return success();
}

LogicalResult DeviceGraphStagingOp::verify() {
  Block &body = getBody().front();

  // Block args are [builder, input_0, ..., input_n-1]; inputs carry the
  // `inputs` operand types verbatim (bare `T`, no `!mogg.async`).
  if (body.getNumArguments() != 1 + getInputs().size())
    return emitOpError("staging body has ")
           << body.getNumArguments() << " block argument(s), expected "
           << (1 + getInputs().size())
           << " (the device-graph builder plus one "
              "per input)";
  if (!isa<DeviceGraphBuilderType>(body.getArgument(0).getType()))
    return emitOpError("staging body argument #0 must be a "
                       "`!mgp.device_graph.builder`, but is ")
           << body.getArgument(0).getType();
  for (auto [idx, inputType] : llvm::enumerate(getInputs().getTypes())) {
    Type argType = body.getArgument(idx + 1).getType();
    if (argType != inputType)
      return emitOpError("staging body argument #")
             << (idx + 1) << " has type " << argType << ", expected "
             << inputType;
  }

  DeviceGraphStagingYieldOp yieldOp = getYieldOp();
  if (yieldOp.getNumOperands() != getNumResults())
    return emitOpError("`mgp.device_graph.staging.yield` has ")
           << yieldOp.getNumOperands() << " operand(s), expected "
           << getNumResults();
  for (auto [idx, resultType] :
       llvm::enumerate(getStagedResults().getTypes())) {
    Type operandType = yieldOp.getOperand(idx).getType();
    if (operandType != resultType)
      return emitOpError("`mgp.device_graph.staging.yield` operand #")
             << idx << " has type " << operandType << ", expected "
             << resultType;
  }
  return success();
}

LogicalResult DeviceGraphStableCreateOp::verify() {
  if (getTensor().getType().getStable())
    return emitOpError("input tensor is already marked `stable`");

  size_t numDynamicDims =
      getTensor().getType().getTensorSpec().getNumDynamicDims();
  if (getRuntimeDims().size() != numDynamicDims)
    return emitOpError("expected ")
           << numDynamicDims << " runtime dimension(s), one per dynamic "
           << "dimension of the tensor, but got " << getRuntimeDims().size();
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceGraphStableGetOp
//===----------------------------------------------------------------------===//

/// The `stableTypes` of the nearest collective enclosing `op`, or null when
/// there is none (dialect-level tests write the ops standalone). Lowering
/// wraps the op in compiled regions and dependency units on its way to
/// emission, so this walks rather than checking the immediate parent.
///
/// TODO(spenser): carry the slot types on `!mgp.device_graph.builder` itself,
/// so `stable.get`/`stable.place` verify against their operand's type and the
/// walk (and the collective-side annotation) go away.
static StableAddressArrayAttr enclosingStableTypes(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto collective = dyn_cast<DeviceGraphCollectiveOp>(parent))
      return collective.getStableTypesAttr();
    if (auto create = dyn_cast<DeviceGraphCollectiveCreateOp>(parent))
      return create.getStableTypesAttr();
  }
  return {};
}

/// Checks that slot `index` exists in the enclosing collective and is declared
/// with `slotType`. Slots are named by index rather than by SSA value, so this
/// is the only place the two agree.
static LogicalResult verifyStableSlot(Operation *op, int64_t index,
                                      Type slotType) {
  if (index < 0)
    return op->emitOpError("stable slot index must not be negative");

  StableAddressArrayAttr stableTypes = enclosingStableTypes(op);
  if (!stableTypes)
    return success();

  ArrayRef<StableAddressType> slots = stableTypes.getValue();
  if (static_cast<uint64_t>(index) >= slots.size())
    return op->emitOpError("stable slot index ")
           << index << " is out of range; the enclosing collective declares "
           << slots.size() << " slot(s)";
  if (slots[index] != slotType)
    return op->emitOpError("stable slot #")
           << index << " is declared as " << slots[index]
           << ", but the op uses " << slotType;
  return success();
}

LogicalResult DeviceGraphStableGetOp::verify() {
  return verifyStableSlot(*this, getIndex(), getResult().getType());
}

LogicalResult DeviceGraphStablePlaceOp::verify() {
  return verifyStableSlot(*this, getIndex(), getPlacedAddr().getType());
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveOp
//===----------------------------------------------------------------------===//

// Verifies:
//   1. `$devices.size() == #regions`.
//   2. Every region has exactly one block argument, a
//      `!mgp.device_graph.builder`.
//   3. Every region terminates with `DeviceGraphCollectiveYieldOp`.
//   4. The total yielded operand count across regions equals `$outputs.size()`
//      (yields append in region order).
// `$stableTypes` needs no check here: its attribute type only admits
// `!mgp.stable_addr` elements.
LogicalResult DeviceGraphCollectiveOp::verify() {
  if (getGraphs().size() != getDevices().size())
    return emitOpError("`devices` attribute has ")
           << getDevices().size() << " entries, but the op has "
           << getGraphs().size() << " regions";

  if (getDevices().empty())
    return success();

  unsigned outputIdx = 0;
  for (auto [idx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.getNumArguments() != 1)
      return emitOpError("region #")
             << idx << " has " << block.getNumArguments()
             << " block argument(s); expected exactly one, the "
                "`!mgp.device_graph.builder`";

    if (!isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << idx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    if (!block.mightHaveTerminator() ||
        !isa<DeviceGraphCollectiveYieldOp>(block.getTerminator()))
      return emitOpError("region #")
             << idx
             << " must terminate with `mgp.device_graph.collective.yield`";

    auto yield = cast<DeviceGraphCollectiveYieldOp>(block.getTerminator());
    for (auto operand : yield.getOperands()) {
      if (outputIdx >= getOutputs().size())
        return emitOpError("region #")
               << idx << " yields more operands than declared outputs ("
               << getOutputs().size() << ")";
      if (operand.getType() != getOutputs()[outputIdx].getType())
        return emitOpError("region #")
               << idx << " yield operand type " << operand.getType()
               << " does not match output #" << outputIdx << " type "
               << getOutputs()[outputIdx].getType();
      ++outputIdx;
    }
  }

  if (outputIdx != getOutputs().size())
    return emitOpError("total `mgp.device_graph.collective.yield` operand "
                       "count across regions is ")
           << outputIdx << ", expected " << getOutputs().size()
           << " (one per op result)";

  return success();
}

LogicalResult DeviceGraphCollectiveCreateOp::verify() {
  if (std::optional<ArrayRef<int64_t>> shapeOnly = getShapeOnlyInputs()) {
    for (int64_t index : *shapeOnly) {
      if (index < 0 || static_cast<size_t>(index) >= getInputs().size())
        return emitOpError("`shapeOnlyInputs` names input #")
               << index << ", but the op has " << getInputs().size()
               << " inputs";
      if (!graphInputDevice(getInputs()[index]))
        return emitOpError("`shapeOnlyInputs` names input #")
               << index << ", which is not a tensor or buffer";
    }
  }

  if (getGraphs().size() != getDeviceContexts().size())
    return emitOpError("`deviceContexts` has ")
           << getDeviceContexts().size() << " operands, but the op has "
           << getGraphs().size() << " regions";

  if (getOutGraphs().size() != getGraphs().size())
    return emitOpError() << "number of result graphs (" << getOutGraphs().size()
                         << ") does not match the number of body regions ("
                         << getGraphs().size() << ")";

  if (getGraphs().empty())
    return emitOpError("must have at least one device graph region");

  Type theContextType = getTheContext().getType();
  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  size_t inputIdx = 0;

  for (auto [regionIdx, region] : llvm::enumerate(getGraphs())) {
    Block &block = region.front();

    if (block.empty() || !llvm::isa<DeviceGraphYieldOp>(block.back()))
      return emitOpError("region #")
             << regionIdx << " must terminate with `mgp.device_graph.yield`";

    if (block.getNumArguments() < 2)
      return emitOpError("region #")
             << regionIdx << " has " << block.getNumArguments()
             << " block argument(s), expected at least 2 (the builder and the "
                "context)";

    if (!llvm::isa<DeviceGraphBuilderType>(block.getArgument(0).getType()))
      return emitOpError("region #")
             << regionIdx << " block argument #0 has type "
             << block.getArgument(0).getType()
             << "; expected `!mgp.device_graph.builder`";

    if (block.getArgument(1).getType() != theContextType)
      return emitOpError("region #")
             << regionIdx
             << " block argument #1 must be the `theContext` operand type "
             << theContextType << ", but is " << block.getArgument(1).getType();

    size_t numRegionInputs = block.getNumArguments() - 2;
    if (inputIdx + numRegionInputs > getInputs().size())
      return emitOpError("region #")
             << regionIdx << " has " << numRegionInputs
             << " input block argument(s), but only "
             << (getInputs().size() - inputIdx) << " input operand(s) remain";

    for (size_t i = 0; i < numRegionInputs; ++i) {
      Type argType = block.getArgument(2 + i).getType();
      Type inputType = getInputs()[inputIdx + i].getType();
      if (argType != inputType)
        return emitOpError("region #")
               << regionIdx << " block argument #" << (2 + i) << " has type "
               << argType << ", expected " << inputType;
    }

    auto yieldOp = cast<DeviceGraphYieldOp>(block.back());
    auto graphType = cast<DeviceGraphType>(getOutGraphs()[regionIdx].getType());
    FunctionType sig = graphType.getSignature();

    ArrayRef<Type> regionInputTypes(
        ArrayRef<Type>(inputTypes).slice(inputIdx, numRegionInputs));
    if (!llvm::equal(sig.getInputs(), regionInputTypes))
      return emitOpError("result graph #")
             << regionIdx
             << " signature inputs do not match this region's input operands";

    if (!llvm::equal(sig.getResults(), yieldOp.getOutputs().getTypes()))
      return emitOpError("result graph #")
             << regionIdx
             << " signature results do not match the region's yielded value "
                "types";

    inputIdx += numRegionInputs;
  }

  if (inputIdx != getInputs().size())
    return emitOpError("the op has ")
           << getInputs().size()
           << " input operand(s), but the regions consume only " << inputIdx;

  return verifyInputResidency(*this, getInputs());
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceGraphCollectiveExecuteOp::verify() {
  size_t expectedInputCount = 0;
  size_t expectedOutputCount = 0;

  for (auto graph : getGraphs()) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();
    expectedInputCount += sig.getInputs().size();
    expectedOutputCount += sig.getResults().size();
  }

  if (getInputs().size() != expectedInputCount)
    return emitOpError("`inputs` has ")
           << getInputs().size()
           << " operands, but the graph signatures consume "
           << expectedInputCount;

  if (getOutputs().size() != expectedOutputCount)
    return emitOpError("`outputs` has ")
           << getOutputs().size()
           << " results, but the graph signatures produce "
           << expectedOutputCount;

  SmallVector<Type> inputTypes;
  llvm::append_range(inputTypes, getInputs().getTypes());
  SmallVector<Type> outputTypes;
  llvm::append_range(outputTypes, getOutputs().getTypes());

  size_t inputIdx = 0;
  size_t outputIdx = 0;
  for (auto [graphIdx, graph] : llvm::enumerate(getGraphs())) {
    auto sig = cast<DeviceGraphType>(graph.getType()).getSignature();

    if (!llvm::equal(
            sig.getInputs(),
            ArrayRef<Type>(inputTypes).slice(inputIdx, sig.getInputs().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature inputs do not match the corresponding `inputs` "
                "operands";
    inputIdx += sig.getInputs().size();

    if (!llvm::equal(sig.getResults(),
                     ArrayRef<Type>(outputTypes)
                         .slice(outputIdx, sig.getResults().size())))
      return emitOpError("graph #")
             << graphIdx
             << " signature results do not match the corresponding `outputs` "
                "results";
    outputIdx += sig.getResults().size();
  }

  return verifyInputResidency(*this, getInputs());
}
