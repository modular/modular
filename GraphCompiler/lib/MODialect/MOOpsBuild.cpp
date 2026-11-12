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
// This file implements the builders of the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GraphCompiler/Matchers.h"
#include "GraphCompiler/Support/Dims.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

using namespace M;
using namespace MO;
//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

void TransferOp::build(OpBuilder &builder, OperationState &state,
                       TensorValue inputValue, DeviceRefAttr destDevice,
                       ChainValue inChain, bool alwaysElideSameDeviceCopy) {
  auto resultTy = inputValue.getType().cloneWith(destDevice);
  auto chainTy = ChainType::get(builder.getContext());
  build(builder, state, resultTy, chainTy, inputValue,
        alwaysElideSameDeviceCopy, inChain);
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

/// Traverse the given types and collect unique parameter references, then
/// return a set of parameter declaration for those references.
static llvm::SetVector<KGEN::ParamDeclAttr>
getParamDeclsFromShapes(TypeRange types) {
  llvm::SetVector<KGEN::ParamDeclAttr> uniqueDecls;
  for (Type type : types) {
    type.walk([&](KGEN::ParamDeclRefAttr declRef) {
      uniqueDecls.insert(KGEN::ParamDeclAttr::get(declRef));
    });
  }
  return uniqueDecls;
}

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    TypeRange inputTypes, TypeRange resultTypes,
                    bool isSubgraph) {
  // Specializes `buildWithEntryBlock`, which doesn't work because we use
  // `signature` and not `function_type`.
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  if (isSubgraph) {
    state.addAttribute(getIsSubgraphAttrName(state.name),
                       builder.getUnitAttr());
  }
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, funcType);
  state.addAttribute(getSignatureAttrName(state.name), TypeAttr::get(sig));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));

  // Declare input/result parameters, if any is used in the input/result types.
  MLIRContext *ctx = builder.getContext();
  auto inputDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(inputTypes).getArrayRef());
  auto resultDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(resultTypes).getArrayRef());
  state.addAttribute(getInputParametersAttrName(state.name), inputDecls);
  state.addAttribute(getResultParametersAttrName(state.name), resultDecls);

  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

void TensorBundleOp::build(OpBuilder &builder, OperationState &state,
                           ValueRange inputs) {
  SmallVector<Type> elementTypes =
      llvm::map_to_vector(inputs, [](Value v) { return v.getType(); });
  build(builder, state, BundleType::get(builder.getContext(), elementTypes),
        inputs);
}

void TensorUnbundleOp::build(OpBuilder &builder, OperationState &state,
                             Value input) {
  build(builder, state, cast<BundleType>(input.getType()).getElementTypes(),
        input);
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, TypeRange resultTypes) {
  build(builder, state, inputs, /*buffers=*/{}, /*inChain=*/Value{},
        resultTypes);
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, ValueRange buffers, Value inChain,
                       TypeRange resultTypes) {
  assert(!inputs.empty() &&
         "auto-derived ParallelOp::build requires at least one input bundle; "
         "use the explicit-launchDevices overload for bundle-less parallels");
  build(builder, state, inputs, buffers, inChain, resultTypes,
        deriveLaunchDevicesFromBundle(
            cast<BundleType>(inputs.front().getType())));
}

void ParallelOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, ValueRange buffers, Value inChain,
                       TypeRange resultTypes,
                       ArrayRef<DeviceRefAttr> launchDevices) {
  assert(buffers.empty() == !inChain &&
         "buffers and inChain must be both present or both absent");
  state.addOperands(inputs);
  state.addOperands(buffers);
  if (inChain)
    state.addOperands(inChain);
  state.addTypes(resultTypes);
  state.addAttribute(
      ParallelOp::getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                                    static_cast<int32_t>(buffers.size()),
                                    static_cast<int32_t>(inChain ? 1 : 0)}));
  state.addAttribute(
      ParallelOp::getLaunchDevicesAttrName(state.name),
      builder.getArrayAttr(llvm::map_to_vector(
          launchDevices, [](DeviceRefAttr d) -> Attribute { return d; })));
  // Block args are the first element type of each bundle input, plus a
  // trailing !mo.chain when chained: the in-chain enters the body through it
  // (not captured), and the body threads it through its buffer ops and yields
  // the resulting chain.
  Block &block = state.addRegion()->emplaceBlock();
  SmallVector<Type> argTypes;
  for (Value input : inputs)
    argTypes.push_back(
        cast<BundleType>(input.getType()).getElementTypes().front());
  if (inChain)
    argTypes.push_back(ChainType::get(builder.getContext()));
  SmallVector<Location> locs(argTypes.size(), state.location);
  block.addArguments(argTypes, locs);
}

void SequenceOp::build(OpBuilder &builder, OperationState &state,
                       ValueRange inputs, TypeRange resultTypes,
                       uint64_t streamId) {
  state.addOperands(inputs);
  state.addTypes(resultTypes);
  state.addAttribute(
      SequenceOp::getStreamIdAttrName(state.name),
      builder.getIntegerAttr(builder.getIntegerType(64, /*isSigned=*/false),
                             streamId));
  // One block argument per input, typed 1:1.
  Block &block = state.addRegion()->emplaceBlock();
  SmallVector<Type> argTypes(inputs.getTypes());
  SmallVector<Location> locs(argTypes.size(), state.location);
  block.addArguments(argTypes, locs);
}

void BroadcastToOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TensorValue input, TensorValue broadcastLike,
                          TensorValue broadcastShape) {
  TensorType resultTy =
      input.getType().cloneWith(broadcastLike.getType().getShape());

  // Note broadcastShape may be null
  if (!broadcastShape)
    broadcastShape =
        MO::ShapeOfOp::create(odsBuilder, input.getLoc(), broadcastLike);

  build(odsBuilder, odsState, resultTy, input, broadcastShape);
}
namespace {

// Create an ElementsAttr from a list of primitive values.

template <typename T>
ElementsAttr createElementsAttr(OpBuilder &odsBuilder, ArrayRef<T> values,
                                Type elementType) {
  if constexpr (std::is_same_v<T, bool>) {
    // For bool, we convert to i8 type.
    auto rankedAttrTy =
        RankedTensorType::get(values.size(), odsBuilder.getI8Type());
    auto apIntVals = llvm::map_to_vector(
        values, [](bool val) { return APInt(8, val, /*isSigned=*/false); });
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, apIntVals);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_integral_v<T>) {
    return cast<ElementsAttr>(odsBuilder.getAttr<IntArrayElementsAttr>(
        values, mlir::IntegerType::Signed));
  } else if constexpr (std::is_floating_point_v<T>) {
    auto apFloatValues =
        llvm::map_to_vector(values, [](T value) { return APFloat(value); });
    return cast<ElementsAttr>(
        FloatArrayElementsAttr::get(apFloatValues, elementType));
  } else if constexpr (std::is_same_v<T, APInt>) {
    auto rankedAttrTy = RankedTensorType::get(values.size(), elementType);
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, values);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_same_v<T, APFloat>) {
    return cast<ElementsAttr>(FloatArrayElementsAttr::get(values, elementType));
  }
}

template <typename T>
void buildConstant(OpBuilder &odsBuilder, OperationState &odsState,
                   ArrayRef<T> values, Type elementType,
                   DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get({(int64_t)values.size()}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(odsBuilder, values, elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}

template <typename T>
void buildConstantScalar(OpBuilder &odsBuilder, OperationState &odsState,
                         T value, Type elementType, DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get(SmallVector<int64_t>{}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(
      odsBuilder, ArrayRef<T>(SmallVector<T>{value}), elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}
} // namespace

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Type resultType, ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int64_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(64, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int32_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(32, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APInt> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APFloat> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<bool> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values, odsBuilder.getI1Type(),
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(64, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int32_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(32, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APFloat value, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstantScalar(odsBuilder, odsState, value, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APInt value, Type elementType, DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, ArrayRef<APInt>(value), elementType,
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       float value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF32Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       double value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF64Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       bool value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getI1Type(),
                      deviceRef);
}

void ConstantScalarOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             TypedAttr value) {
  auto type = value.getType();

  DType dtype;

  if (auto integerType = dyn_cast<IntegerType>(type))
    dtype = M::getEquivalentDType(integerType);
  else
    dtype = M::getEquivalentDType(cast<FloatType>(type));

  auto resultType = odsBuilder.getType<ScalarType>(dtype);
  build(odsBuilder, odsState, resultType, value);
}

//===----------------------------------------------------------------------===//
// BufferCreateOp
//===----------------------------------------------------------------------===//

void BufferCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           Type result, Attribute initValue) {
  if (initValue)
    odsState.addAttribute(getInitValueAttrName(odsState.name), initValue);
  odsState.addTypes(result);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
}

template <typename TOp>
void buildReductionOp(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t axis, TensorType outputTy) {
  TOp::build(odsBuilder, odsState, outputTy, input,
             odsBuilder.getIndexAttr(axis));
}

void ArgMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}

void ArgMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//

void BatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type resultTy, Value input_a, Value input_b,
                          ArrayRef<KGEN::ParamDeclAttr> decls) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({resultTy});
  if (!decls.empty()) {
    odsState.addAttribute(
        getOutputParamDeclsAttrName(odsState.name),
        KGEN::ParamDeclArrayAttr::get(odsBuilder.getContext(), decls));
  }
}

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t width) {
  ASSERT_STREAM(llvm::is_contained({32, 64}, width),
                << "`mo.shape_of` returns si32 or si64");
  Type intTy = odsBuilder.getIntegerType(width, /*isSigned=*/true);

  int64_t outputSize = mlir::ShapedType::kDynamic;
  if (input.getType().hasRank())
    outputSize = input.getType().getRank();

  // TODO(#18749): don't introduce `?` with operations.
  build(odsBuilder, odsState,
        MO::TensorType::get({outputSize}, intTy,
                            getHostDevice(odsBuilder.getContext())),
        input);
}

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input) {
  build(odsBuilder, odsState, input, /*width=*/32);
}

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond,
                     const Twine &message) {
  auto attr = StringAttr::get(message.str(),
                              KGEN::StringType::get(builder.getContext()));
  build(builder, state, {}, cond, cast<TypedAttr>(attr));
}

/// Derives the per-launch device list from each bundle element's deviceRef.
/// `BundleType::verify` guarantees every element is a TensorType.
SmallVector<DeviceRefAttr>
M::MO::deriveLaunchDevicesFromBundle(BundleType bundle) {
  return llvm::map_to_vector(bundle.getElementTypes(), [](Type t) {
    return cast<TensorType>(t).getDeviceRef();
  });
}

//===----------------------------------------------------------------------===//
// ReduceAddOp, ReduceMinOp, ReduceMaxOp, ReduceMulOp

static void buildImplicitAxisReduction(OpBuilder &builder,
                                       OperationState &state, TensorValue input,
                                       int64_t axis) {
  TensorType inputTy = input.getType();
  state.addOperands(input);
  state.addTypes(
      inputTy.cloneWithNewDim(axis < 0 ? axis + inputTy.getRank() : axis, 1));
}

void ReduceAddOp::build(OpBuilder &builder, OperationState &state,
                        TensorValue input, int64_t axis) {
  buildImplicitAxisReduction(builder, state, input, axis);
}

void ReduceMinOp::build(OpBuilder &builder, OperationState &state,
                        TensorValue input, int64_t axis) {
  buildImplicitAxisReduction(builder, state, input, axis);
}

void ReduceMaxOp::build(OpBuilder &builder, OperationState &state,
                        TensorValue input, int64_t axis) {
  buildImplicitAxisReduction(builder, state, input, axis);
}

void ReduceMulOp::build(OpBuilder &builder, OperationState &state,
                        TensorValue input, int64_t axis) {
  buildImplicitAxisReduction(builder, state, input, axis);
}
