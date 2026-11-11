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
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
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

void DeviceGraphYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesOp
//===----------------------------------------------------------------------===//

DeviceGraphDependenciesYieldOp DeviceGraphDependenciesOp::getYieldOp() {
  return cast<DeviceGraphDependenciesYieldOp>(
      getBody().front().getTerminator());
}

TypedValue<DeviceGraphBuilderType>
DeviceGraphDependenciesOp::getBuilderBlockArgument() {
  return cast<TypedValue<DeviceGraphBuilderType>>(
      getBody().front().getArgument(0));
}

void DeviceGraphDependenciesOp::build(
    OpBuilder &builder, OperationState &result, Value graphBuilder,
    ValueRange dependencies,
    llvm::function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(graphBuilder);
  result.addOperands(dependencies);

  // The synthesized node is always the first result.
  result.addTypes(DeviceGraphNodeType::get(builder.getContext()));

  // The callback populates the body and is expected to emit the
  // dependencies.yield. The body otherwise references enclosing SSA directly.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value builderArg = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  if (bodyBuilder)
    bodyBuilder(builder, result.location, builderArg);

  // Supply the operandless terminator (no body results) if the callback did not
  // emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  // The remaining results are exactly the dependencies.yield operands.
  auto yieldOp = cast<DeviceGraphDependenciesYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperands().getTypes());
}

Block &DeviceGraphDependenciesOp::getBodyBlock() { return getBody().front(); }

//===----------------------------------------------------------------------===//
// DeviceGraphDependenciesYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphDependenciesYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingOp
//===----------------------------------------------------------------------===//

DeviceGraphStagingYieldOp DeviceGraphStagingOp::getYieldOp() {
  return cast<DeviceGraphStagingYieldOp>(getBody().front().getTerminator());
}

void DeviceGraphStagingOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    llvm::function_ref<void(OpBuilder &, Location, Value, ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(inputs);

  // The callback populates the body and is expected to emit the staging.yield.
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion);
  Value graphBuilder = body->addArgument(
      DeviceGraphBuilderType::get(builder.getContext()), result.location);
  SmallVector<Value> inputArgs;
  inputArgs.reserve(inputs.size());
  for (Value input : inputs)
    inputArgs.push_back(body->addArgument(input.getType(), result.location));
  if (bodyBuilder)
    bodyBuilder(builder, result.location, graphBuilder, inputArgs);

  // Supply the operandless terminator (zero staged results) if the callback did
  // not emit one.
  if (!body->mightHaveTerminator())
    ensureTerminator(*bodyRegion, builder, result.location);

  auto yieldOp = cast<DeviceGraphStagingYieldOp>(body->getTerminator());
  result.addTypes(yieldOp.getOperandTypes());
}

//===----------------------------------------------------------------------===//
// DeviceGraphStagingYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphStagingYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's SingleBlockImplicitTerminator; operands
  // are populated after insertion.
}

void DeviceGraphCollectiveOp::build(
    OpBuilder &builder, OperationState &state,
    ArrayRef<M::DeviceRefAttr> devices, ArrayRef<StableAddressType> stableTypes,
    function_ref<void(OpBuilder &, M::DeviceRefAttr, BlockArgument)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  MLIRContext *ctx = builder.getContext();

  // Build one region per device, each with the builder as its only block
  // argument.
  SmallVector<Type> resultTypes;
  for (auto dev : devices) {
    Region *region = state.addRegion();
    Block *block = builder.createBlock(region);

    auto builderArg =
        block->addArgument(DeviceGraphBuilderType::get(ctx), state.location);

    builder.setInsertionPointToEnd(block);
    bodyBuilder(builder, dev, builderArg);
    ensureTerminator(*region, builder, state.location);

    llvm::append_range(resultTypes, block->getTerminator()->getOperandTypes());
  }

  auto stableTypesAttr = StableAddressArrayAttr::get(ctx, stableTypes);

  // Call into the tablegen builder with `graphsCount=0`. We've already added
  // the child regions to `state`.
  DeviceGraphCollectiveOp::build(builder, state, resultTypes, devices,
                                 stableTypesAttr,
                                 /*constantResources=*/ArrayAttr(),
                                 /*graphsCount=*/0);
}

//===----------------------------------------------------------------------===//
// DeviceGraphCollectiveYieldOp
//===----------------------------------------------------------------------===//

void DeviceGraphCollectiveYieldOp::build(OpBuilder &, OperationState &) {
  // No-operand stub for the parent's `SingleBlockImplicitTerminator`; operands
  // are populated after insertion.
}

void DeviceGraphCollectiveCreateOp::build(
    OpBuilder &builder, OperationState &result, Value theContext,
    llvm::ArrayRef<Value> deviceContexts, Value cache,
    llvm::ArrayRef<Value> inputs, llvm::ArrayRef<unsigned> inputCounts,
    llvm::function_ref<void(OpBuilder &, Location, unsigned, Value, Value,
                            ValueRange)>
        bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  unsigned numRegions = inputCounts.size();
  unsigned inputTotal = 0;
  for (unsigned count : inputCounts)
    inputTotal += count;
  assert(inputTotal == inputs.size() &&
         "inputCounts must sum to inputs.size()");

  MLIRContext *ctx = builder.getContext();
  SmallVector<Type> graphTypes;
  graphTypes.reserve(numRegions);

  size_t inputOffset = 0;

  for (unsigned i : llvm::seq(numRegions)) {
    Region *region = result.addRegion();
    Block *block = builder.createBlock(region);

    Value graphBuilder =
        block->addArgument(DeviceGraphBuilderType::get(ctx), result.location);
    Value contextArg =
        block->addArgument(theContext.getType(), result.location);

    unsigned count = inputCounts[i];
    ValueRange argSlice =
        llvm::ArrayRef<Value>(inputs).slice(inputOffset, count);
    SmallVector<Location> locs(count, result.location);
    auto inputArgs = block->addArguments(argSlice.getTypes(), locs);

    if (bodyBuilder)
      bodyBuilder(builder, result.location, i, graphBuilder, contextArg,
                  ValueRange(inputArgs));

    if (!block->mightHaveTerminator())
      ensureTerminator(*region, builder, result.location);

    auto yieldOp = cast<DeviceGraphYieldOp>(block->getTerminator());
    FunctionType signature = FunctionType::get(ctx, argSlice.getTypes(),
                                               yieldOp.getOutputs().getTypes());
    graphTypes.push_back(DeviceGraphType::get(ctx, signature));

    inputOffset += count;
  }

  build(builder, result, graphTypes, theContext, deviceContexts, cache, inputs,
        /*stableTypes=*/StableAddressArrayAttr(),
        /*shapeOnlyInputs=*/mlir::DenseI64ArrayAttr(), /*graphsCount=*/0);
}
