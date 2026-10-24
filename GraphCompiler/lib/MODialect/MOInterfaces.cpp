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
// This file implements the MO dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MO;

#define DEBUG_TYPE "mo-interfaces"

//===----------------------------------------------------------------------===//
// MatMulLike
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyMatchingRanks(ArrayRef<TensorValue> values) {
  SmallVector<size_t> staticRanks;
  for (auto v : values) {
    TensorType t = v.getType();
    // Layout annotations may change ranks.
    if (t.getLayoutAttr())
      return success();
    staticRanks.emplace_back(t.getRank());
  }

  return success(llvm::all_equal(staticRanks));
}

LogicalResult Impl::checkMatmulCompatibility(Operation *op, TensorValue A,
                                             TensorValue B, TensorValue C) {
  // Implicit broadcasting is not supported.
  SmallVector<TensorValue> tensors{A, B, C};

  TensorType aType = A.getType();
  TensorType bType = B.getType();

  if (aType.getRank() < 2) {
    return op->emitOpError(
               "expected input tensors to have at least rank 2, got ")
           << aType.getRank();
  }

  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return success();

  // Lowering may add attributes that implicitly transpose operands.
  if (auto bmm = dyn_cast<BatchMatmulOp>(op))
    if (bmm.getTransposeB())
      return success();

  // Likewise for layout annotations added during lowering.
  if (aType.getLayoutAttr() || bType.getLayoutAttr())
    return success();

  ArrayRef<TypedAttr> aMatShape = aType.getDims().take_back(2);
  ArrayRef<TypedAttr> bMatShape = bType.getDims().take_back(2);
  if (aMatShape[1] != bMatShape[0])
    return op->emitOpError("incompatible input/result matmul shapes");

  ArrayRef<TypedAttr> aMatBatches = aType.getDims().drop_back(2);
  ArrayRef<TypedAttr> bMatBatches = bType.getDims().drop_back(2);
  if (!aMatBatches.equals(bMatBatches))
    return op->emitOpError("incompatible input/result batch dimensions");

  TensorType resType = C.getType();

  // Dynamic results are left to shape inference.
  if (!resType.hasStaticShape())
    return success();

  ArrayRef<TypedAttr> resMatShape = resType.getDims().take_back(2);
  if (resMatShape[0] != aMatShape[0] || resMatShape[1] != bMatShape[1])
    return op->emitOpError("incompatible input/result matmul shapes");

  ArrayRef<TypedAttr> resMatBatches = resType.getDims().drop_back(2);
  if (!aMatBatches.equals(resMatBatches))
    return op->emitOpError("incompatible input/result batch dimensions");

  return success();
}

LogicalResult Impl::verifyMatmulLike(MatmulLike op) {
  return checkMatmulCompatibility(op.getOperation(), op.getTensorA(),
                                  op.getTensorB(), op.getTensorResult());
}

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyReduction(Reduction op) {
  // TODO: test
  if (failed(verifyMatchingRanks({op.getInput(), op.getResult()})))
    return op.emitOpError("input and output must either both have dynamic "
                          "ranks or both have the same size static ranks");

  TensorType inputTy = op.getInput().getType();
  int64_t rank = inputTy.getRank();
  if (IntegerAttr axisAttr = op.getAxisAttribute()) {
    int64_t axis = axisAttr.getInt();
    if (axis < -rank || axis > rank - 1)
      return op.emitOpError() << "axis must be between [-" << rank << ", "
                              << (rank - 1) << "] found " << axis;
    return success();
  }

  // Without `axis`, the result shape names the reduced dimension.
  if (inputTy.getShape() == op.getResult().getType().getShape())
    return success();
  ErrorOr<int64_t> axis =
      getReductionAxis(op.getInput(), op.getResult().getType());
  if (axis.isError())
    return op.emitOpError() << axis.getError();
  return success();
}

ShapeMaterializeResult
Impl::materializeReductionShapeDefs(Reduction op,
                                    ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  std::optional<int64_t> axis = op.getReductionAxis();
  OpBuilder builder(op);
  return materializeReduceShape(graphOp, loc, inputShape,
                                axis ? builder.getIndexAttr(*axis)
                                     : TypedAttr());
}

//===----------------------------------------------------------------------===//
// ScatterLike
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyScatterLike(ScatterLike op) {
  TensorType inputTy = op.getInput().getType();
  TensorType resTy = op.getInput().getType();
  if (!inputTy.hasStaticShape() || !resTy.hasStaticShape())
    return success();
  TypedAttr inputShapeAttr = inputTy.getShape();
  TypedAttr resShapeAttr = resTy.getShape();
  if (inputShapeAttr != resShapeAttr) {
    return op->emitError()
           << "input and result shapes must match if both static: "
           << "input shape = " << MOSH::shapeTypedAttrToStr(inputShapeAttr)
           << ", result shape = " << MOSH::shapeTypedAttrToStr(resShapeAttr);
  }

  uint64_t expectedRank;
  auto getOptionalRankAndSetExpectedRank =
      [&](TensorValue tensor) -> std::optional<uint64_t> {
    TensorType tensorTy = tensor.getType();
    if (!tensorTy.hasRank()) {
      return std::nullopt;
    }
    expectedRank = tensorTy.getRank();

    return expectedRank;
  };

  auto rankIsSameAsExpected = [&](std::optional<uint64_t> rank) -> bool {
    return rank.value_or(expectedRank) == expectedRank;
  };

  auto inputRank = getOptionalRankAndSetExpectedRank(op.getInput());
  auto updatesRank = getOptionalRankAndSetExpectedRank(op.getUpdates());
  auto indicesRank = getOptionalRankAndSetExpectedRank(op.getIndices());
  auto outputRank = getOptionalRankAndSetExpectedRank(op.getResult());

  if (!rankIsSameAsExpected(inputRank) || !rankIsSameAsExpected(updatesRank) ||
      !rankIsSameAsExpected(indicesRank) || !rankIsSameAsExpected(outputRank)) {
    return op->emitError() << "Output and inputs (except axis) of a "
                              "scatter-like op must have the same rank";
  }

  return success();
}

ShapeMaterializeResult
Impl::materializeScatterLikeShapeDefs(ScatterLike op,
                                      ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr updatesShape = inputAttrs[1];
  TypedAttr indicesShape = inputAttrs[2];
  TypedAttr axis = inputAttrs[3];
  return materializeScatterShape(graphOp, loc, inputShape, updatesShape,
                                 indicesShape, axis);
}

//===----------------------------------------------------------------------===//
// ScatterNdLike
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyScatterNdLike(ScatterNdLike op) {
  TensorType inputTy = op.getInput().getType();
  TensorType resTy = op.getResult().getType();

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank != resRank) {
    return op->emitError()
           << "input and result ranks must match if both static: "
           << "input rank = " << inputRank << ", result rank = " << resRank;
  }

  if (!inputTy.hasStaticShape() || !resTy.hasStaticShape())
    return success();
  TypedAttr inputShapeAttr = inputTy.getShape();
  TypedAttr resShapeAttr = resTy.getShape();
  if (inputShapeAttr != resShapeAttr) {
    return op->emitError()
           << "input and result shapes must match if both static: "
           << "input shape = " << MOSH::shapeTypedAttrToStr(inputShapeAttr)
           << ", result shape = " << MOSH::shapeTypedAttrToStr(resShapeAttr);
  }

  {
    int64_t dataRank = op.getInput().getType().getRank();
    int64_t indicesRank = op.getIndices().getType().getRank();
    int64_t indicesLastDim =
        op.getIndices().getType().getIntDim(indicesRank - 1);

    int64_t updatesRank = op.getUpdates().getType().getRank();
    int64_t expectedUpdatesRank = dataRank + indicesRank - indicesLastDim - 1;
    if (indicesLastDim != mlir::ShapedType::kDynamic &&
        updatesRank != expectedUpdatesRank) {
      return op->emitError("updates rank must be: data_rank + indices_rank - "
                           "indices_shape[-1] - 1");
    }
  }

  return success();
}

ShapeMaterializeResult
Impl::materializeScatterNdLikeShapeDefs(ScatterNdLike op,
                                        ArrayRef<TypedAttr> inputAttrs) {
  auto graphOp = op->getParentOfType<GraphOp>();
  Location loc = op->getLoc();
  TypedAttr inputShape = inputAttrs[0];
  TypedAttr updatesShape = inputAttrs[1];
  TypedAttr indicesShape = inputAttrs[2];
  return materializeScatterNdShape(graphOp, loc, inputShape, updatesShape,
                                   indicesShape);
}

//===----------------------------------------------------------------------===//
// SlidingWindow
//===----------------------------------------------------------------------===//

LogicalResult SlidingWindow::verifyWindowParamShape(TensorValue windowParam,
                                                    StringRef name) {
  auto input =
      cast<MOTensorOpInterface>(this->getOperation()).getInputTensor(0);
  if (TensorType inputTy = input.getType(); inputTy.hasRank()) {
    if (TensorType windowParamTy = windowParam.getType();
        windowParamTy.hasStaticShape()) {
      if (name == "paddings") {
        if (static_cast<size_t>(windowParamTy.getIntDim(0)) !=
            2 * (inputTy.getRank() - 2)) {
          return emitError(name)
                 << " with static size must match 2 * (input rank - 2)";
        }
      } else if (static_cast<size_t>(windowParamTy.getIntDim(0)) !=
                 inputTy.getRank() - 2) {
        return emitError(name) << " with static size must match input rank - 2";
      }
    }
  }
  return success();
}

LogicalResult
SlidingWindow::verifyWindowParamSizes(ArrayRef<TensorValue> values) {
  SmallPtrSet<Attribute, 4> sizes;
  for (auto v : values)
    if (TensorType t = v.getType(); t.hasRank())
      sizes.insert(t.getDim(0));

  if (sizes.size() > 1)
    return emitError("window parameter sizes must match if static");

  return success();
}

LogicalResult Impl::verifySlidingWindow(SlidingWindow op) {
  if (failed(op.verifyWindowParamShape(op.getStrides(), "strides")))
    return failure();
  if (failed(op.verifyWindowParamShape(op.getDilations(), "dilations")))
    return failure();
  if (failed(op.verifyWindowParamShape(op.getPaddings(), "paddings")))
    return failure();

  auto moOp = cast<MOTensorOpInterface>(op.getOperation());
  if (failed(verifyMatchingRanks(
          {moOp.getInputTensor(0), moOp.getOutputTensor(0)}))) {
    return op.emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsAndResultElementType
//===----------------------------------------------------------------------===//

static Type getElementType(Value v) {
  return cast<MO::TensorType>(v.getType()).getElementType();
}

/// This follows the builtin verifySameOperandsAndResultElementType.
LogicalResult Impl::verifyMOSameOperandsAndResultElementType(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementType(op->getResult(0));

  for (auto result : llvm::drop_begin(op->getResults(), 1)) {
    if (getElementType(result) != elementType) {
      return op->emitOpError(
          "requires the same element type for all operands and results");
    }
  }

  for (auto operand : op->getOperands()) {
    if (getElementType(operand) != elementType) {
      return op->emitOpError(
          "requires the same element type for all operands and results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsAndResultElementShapeAttr
//===----------------------------------------------------------------------===//

static TypedAttr getShapeAttr(Value v) {
  return cast<MO::TensorType>(v.getType()).getShape();
}

LogicalResult Impl::verifyMOSameOperandsAndResultShapeAttr(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto shapeAttr = getShapeAttr(op->getResult(0));

  for (auto result : llvm::drop_begin(op->getResults(), 1)) {
    if (getShapeAttr(result) != shapeAttr) {
      return op->emitOpError(
          "requires the same shape for all operands and results");
    }
  }

  for (auto operand : op->getOperands()) {
    if (getShapeAttr(operand) != shapeAttr) {
      return op->emitOpError(
          "requires the same shape for all operands and results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameOperandsElementType
//===----------------------------------------------------------------------===//

/// This follows the builtin verifySameOperandsAndResultElementType.
LogicalResult Impl::verifyMOSameOperandsElementType(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementType(op->getOperand(0));

  for (auto operand : op->getOperands())
    if (getElementType(operand) != elementType)
      return op->emitOpError("requires the same element type for all operands");

  return success();
}

//===----------------------------------------------------------------------===//
// MOSameFirstOperandAndResultsElementType
//===----------------------------------------------------------------------===//

/// This follows the builtin verifySameOperandsAndResultElementType.
LogicalResult
Impl::verifyMOSameFirstOperandAndResultsElementType(Operation *op) {
  if (failed(mlir::OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(mlir::OpTrait::impl::verifyAtLeastNResults(op, 1)))
    return failure();

  auto elementType = getElementType(op->getOperand(0));

  for (auto result : op->getResults())
    if (getElementType(result) != elementType)
      return op->emitOpError(
          "requires the element type of results match first operand.");

  return success();
}

//===----------------------------------------------------------------------===//
// PadLike
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyPadLike(PadLike op) {
  TensorType inputTy = op.getInput().getType();
  TensorType paddingsTy = op.getPaddings().getType();
  if ((int64_t)inputTy.getRank() * 2 != paddingsTy.getIntDim(0))
    return op.emitOpError("paddings shape does not match static input rank");

  return success();
}

//===----------------------------------------------------------------------===//
// BufferOpConformance
//===----------------------------------------------------------------------===//

LogicalResult Impl::verifyBufferOpConformance(Operation *op) {
  auto numBufferOutputs = llvm::count_if(
      op->getResultTypes(), [](Type t) { return isa<MO::BufferType>(t); });

  auto numBufferInputs = llvm::count_if(
      op->getOperandTypes(), [](Type t) { return isa<MO::BufferType>(t); });

  if (numBufferInputs == 0 && numBufferOutputs == 0)
    return success();

  // Non-compute ops. mo.yield is excluded: alias analysis for mo.buffer
  // values is not supported.
  if (isa<MO::ChainCreateOp, MO::OutputOp>(op))
    return success();

  // These ops check this in their own verifiers.
  // TODO(GEX-1947): Unify their logic there with this trait verifier.
  if (isa<MO::YieldOp, MO::GuardOp>(op))
    return success();

  // Deliberately the only op that creates a buffer; to initialize one, pass
  // its result to the initializing op. If the extra memory round-trip matters,
  // implement mo.mutable.store as a kernel with fusion on its input.
  if (isa<MO::BufferCreateOp>(op))
    return success();

  if (numBufferOutputs != 0)
    return op->emitError("Ops are not allowed to create new mo.buffer values.");

  auto chainInputCount = llvm::count_if(
      op->getOperandTypes(), [](Type t) { return isa<MO::ChainType>(t); });

  auto chainOutputCount = llvm::count_if(
      op->getResultTypes(), [](Type t) { return isa<MO::ChainType>(t); });

  if (chainInputCount != chainOutputCount)
    return op->emitError(
        llvm::formatv("number of input chains ({}) does not match number "
                      "of output chains ({})",
                      chainInputCount, chainOutputCount));

  if (chainInputCount == 0)
    return op->emitError(
        "buffer consuming op must have at least one chain input/output");

  if (!isa<ChainType>(op->getResults().back().getType())) {
    return op->emitError("A mo.chain must be the last output for ops "
                         "with buffer inputs.");
  }

  if (!isa<ChainType>(op->getOperands().back().getType())) {
    return op->emitError("A mo.chain must be the last input for ops "
                         "with mo.buffer inputs.");
  }

  return success();
}

LogicalResult checkDeviceConstraint(Operation *op, const Twine &name, Type type,
                                    DeviceRefAttr constrainedTo) {
  auto deviceType = dyn_cast<MOHasDeviceTypeInterface>(type);
  if (!deviceType)
    return success();
  auto device = deviceType.getDeviceRef();
  if (!device)
    return op->emitError(name) << " is missing a device attribute.";
  if (constrainedTo && device != constrainedTo) {
    return op->emitError(op->getName().getStringRef())
           << ": " << name << " is constrained to be on device "
           << constrainedTo << " but has device " << device << ".";
  }
  return success();
}

LogicalResult checkOperandDeviceConstraint(Operation *op, size_t index,
                                           DeviceRefAttr constrainedTo) {
  Value operand = op->getOperand(index);
  std::string operandDesc = "Operand " + Twine(index).str();

  if (Operation *defOp = operand.getDefiningOp()) {
    operandDesc += " - " + defOp->getName().getStringRef().str();
  }

  operandDesc += " (using zero-based indexing)";

  return checkDeviceConstraint(op, operandDesc, operand.getType(),
                               constrainedTo);
}

LogicalResult checkResultDeviceConstraint(Operation *op, size_t index,
                                          DeviceRefAttr constrainedTo) {
  return checkDeviceConstraint(op, "Result " + Twine(index),
                               op->getResult(index).getType(), constrainedTo);
}

LogicalResult checkExecutionDeviceConstraint(MOHasDeviceInterface op,
                                             DeviceRefAttr constrainedTo) {
  if (constrainedTo && op.getExecutionDevice() != constrainedTo) {
    return op->emitError(op->getName().getStringRef())
           << ": Op is constrained to be on device " << constrainedTo
           << " but has device " << op.getExecutionDevice() << ".";
  }
  return success();
}

DeviceRefAttr Impl::getFirstResultDevice(Operation *op) {
  if (op->getNumResults() >= 1) {
    auto resultType = op->getResult(0).getType();
    if (auto deviceType = dyn_cast<MOHasDeviceTypeInterface>(resultType))
      return deviceType.getDeviceRef();
  }
  op->emitError(
      "Expected the first result to have a device type for MO_SingleDevice");
  return nullptr;
}

LogicalResult Impl::verifyHostOnly(Operation *op) {
  auto hostDevice = getHostDevice(op);

  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    if (!hasDevice.getExecutionDevice())
      return success();
    if (failed(checkExecutionDeviceConstraint(hasDevice, hostDevice)))
      return failure();
  }
  for (size_t i = 0; i < op->getNumOperands(); ++i)
    if (failed(checkOperandDeviceConstraint(op, i, hostDevice)))
      return failure();

  for (size_t i = 0; i < op->getNumResults(); ++i)
    if (failed(checkResultDeviceConstraint(op, i, hostDevice)))
      return failure();

  return success();
}

LogicalResult Impl::verifyExecutesOnHost(Operation *op) {
  auto hostDevice = getHostDevice(op);

  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    if (!hasDevice.getExecutionDevice())
      return success();
    // Control ops get a device even though they need no device context.
    if (isa<MO::MOControlOpInterface, MO::ChainCreateOp, MO::DebugPrintOp>(op))
      return success();
    if (failed(checkExecutionDeviceConstraint(hasDevice, hostDevice)))
      return failure();
  }
  return success();
}

LogicalResult Impl::verifySingleDevice(Operation *op,
                                       ArrayRef<size_t> hostOperandIndices,
                                       ArrayRef<size_t> deviceOperandIndices) {
  // The first device operand decides the device, else the first result.
  DeviceRefAttr singleDevice =
      deviceOperandIndices.empty()
          ? getFirstResultDevice(op)
          : getDeviceOrDefaultAny(
                op->getOperand(deviceOperandIndices.front()).getType());

  if (auto hasDevice = dyn_cast<MOHasDeviceInterface>(op)) {
    if (!hasDevice.getExecutionDevice())
      return success();
    if (failed(checkExecutionDeviceConstraint(hasDevice, singleDevice)))
      return failure();
  }

  // Empty deviceOperandIndices means all non-host operands.
  llvm::SmallVector<size_t> operandIndices;
  for (size_t i = 0; i < op->getNumOperands(); ++i)
    if (!llvm::is_contained(hostOperandIndices, i))
      operandIndices.push_back(i);
  if (deviceOperandIndices.empty())
    deviceOperandIndices = operandIndices;
  for (size_t i : deviceOperandIndices)
    if (failed(checkOperandDeviceConstraint(op, i, singleDevice)))
      return failure();
  for (size_t i : hostOperandIndices)
    if (failed(checkOperandDeviceConstraint(op, i, getHostDevice(op))))
      return failure();
  for (size_t i = 0; i < op->getNumResults(); ++i)
    if (failed(checkResultDeviceConstraint(op, i, singleDevice)))
      return failure();
  return success();
}

LogicalResult
Impl::verifyPeerInputDevices(Operation *op,
                             ArrayRef<size_t> hostOperandIndices) {
  DeviceRefAttr executionDevice = getFirstResultDevice(op);
  if (!executionDevice)
    return failure();
  DeviceRefAttr hostDevice = getHostDevice(op);

  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    auto deviceType = dyn_cast<MOHasDeviceTypeInterface>(operand.getType());
    if (!deviceType)
      continue;
    DeviceRefAttr operandDevice = deviceType.getDeviceRef();
    // Host operands must match the host label exactly, which keeps host
    // scalars (e.g. epsilon) usable with kernels whose device operands live
    // on peer GPUs.
    if (llvm::is_contained(hostOperandIndices, i)) {
      if (operandDevice.getLabel() != hostDevice.getLabel())
        return op->emitError("Host operand ")
               << i << " device label '" << operandDevice.getLabel()
               << "' does not match host device label '"
               << hostDevice.getLabel() << "'.";
      continue;
    }
    if (operandDevice.getLabel() != executionDevice.getLabel())
      return op->emitError("Operand ")
             << i << " device label '" << operandDevice.getLabel()
             << "' does not match execution device label '"
             << executionDevice.getLabel() << "'.";
  }
  return success();
}

LogicalResult Impl::verifyExplicitDevice(Operation *op) {
  if (op->getAttr(MO::kOperationDeviceRefAttrName))
    return success();
  return op->emitOpError(
      "Missing device attribute for op with explicit device trait");
}

LogicalResult Impl::verifyBundledCollectiveInterface(Operation *op) {
  if (op->getParentOfType<ParallelOp>())
    return success();
  if (auto graph = op->getParentOfType<GraphOp>())
    if (graph.getIsDeviceGraph())
      return success();
  return op->emitOpError(
      "must be inside a mo.parallel region or a mo.graph with the "
      "isDeviceGraph attribute");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MODialect/MOInterfaces.cpp.inc"
