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
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Base64.h"
#include <string>

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static llvm::StringRef primitiveMojoName(MOGG::PrimitiveDeclOp decl) {
  // ImportPackages sets sym_name to the fully-qualified Mojo name.
  return decl.getSymNameAttr().strref();
}

ErrorOr<std::optional<uint64_t>>
M::MGP::getImpliedAlignment(ElementsAttr value) {
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    if (!blob && !isa<AlignedBytesType>(resourceAttr.getType())) {
      return Error(Twine("dense_resource blob is not available, so require "
                         "type annotation to supply an alignment"));
    }

    if (blob) {
      if (static_cast<int64_t>(blob->getData().size()) !=
          value.getNumElements())
        return Error(Twine("the resource size does not match the "
                           "attribute's element size (" +
                           Twine(blob->getData().size()) + " vs " +
                           Twine(value.getNumElements()) + ")"));
    }
    // Falls through: DenseResourceElementsAttr implements
    // HasAlignedBytesInterface.
  }

  if (auto hasAlignedBytesAttr = dyn_cast<HasAlignedBytesInterface>(value)) {
    return hasAlignedBytesAttr.getAlignedBytesType().getAlign();
  }

  return Error("cannot supply an alignment from type annotation for "
               "this attribute value.");
}

/// Verifies the types of all operands of op uses devices drawn from the
/// given devices.
LogicalResult M::MGP::verifyConsistentDevices(
    Operation *op, std::initializer_list<DeviceRefAttr> expectedDevices) {
  llvm::SmallSetVector<DeviceRefAttr, 2> allDevices;
  for (auto device : expectedDevices)
    allDevices.insert(device);
  size_t expectedNumDevices = allDevices.size();
  std::string expectedStr;
  {
    llvm::raw_string_ostream os(expectedStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  for (OpOperand &opOperand : op->getOpOperands()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(opOperand.get().getType())) {
      allDevices.insert(hasDeviceType.getDevice());
    }
  }
  for (OpResult opResult : op->getResults()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(opResult.getType())) {
      allDevices.insert(hasDeviceType.getDevice());
    }
  }

  if (allDevices.size() == expectedNumDevices)
    return success();

  std::string actualStr;
  {
    llvm::raw_string_ostream os(actualStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  return op->emitOpError("expected all operands, results, and attributes to "
                         "use the device(s) {")
         << expectedStr << "}, but they instead use the devices(s) {"
         << actualStr << "}.";
}

//===----------------------------------------------------------------------===//
// Custom parsers
//===----------------------------------------------------------------------===//

/// Parses a !mgp.buffer type, extracts its device attribute,
/// assigns chains the type !mef.chain.
static ParseResult parseBufferTypeToDeviceImpl(OpAsmParser &p, Type &buffer,
                                               DeviceRefAttr &device,
                                               ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(buffer))
    return failure();
  auto bufferType = dyn_cast<DeviceBufferType>(buffer);
  if (bufferType == nullptr)
    return p.emitError(loc) << "expected an !mgp.buffer type.";
  device = bufferType.getDevice();
  for (Type *chain : chains)
    *chain = MEF::ChainType::get(p.getContext());
  return success();
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device) {
  return parseBufferTypeToDeviceImpl(p, buffer, device, {});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device) {
  p << buffer;
}

static ParseResult parseBufferTypeToDevice(OpAsmParser &p, Type &buffer,
                                           DeviceRefAttr &device, Type &chain) {
  return parseBufferTypeToDeviceImpl(p, buffer, device, {&chain});
}

static void printBufferTypeToDevice(OpAsmPrinter &p, Operation *op,
                                    DeviceBufferType buffer,
                                    DeviceRefAttr device,
                                    MEF::ChainType chain) {
  p << buffer;
}

static ParseResult parseBufferTypeToChain(OpAsmParser &p, Type &buffer,
                                          Type &chain) {
  DeviceRefAttr device;
  return parseBufferTypeToDeviceImpl(p, buffer, device, {&chain});
}

static void printBufferTypeToChain(OpAsmPrinter &p, Operation *op,
                                   DeviceBufferType buffer,
                                   MEF::ChainType chain) {
  p << buffer;
}

/// Parses a !mgp.tensor type and extracts its device, buffer type, and
/// tensor spec type, and assigns chain the type !mef.chain.
/// If isShapeOnly is true, the tensor spec will have 'unk' dtype.
static ParseResult parseTensorTypeToDeviceBufferSpecImpl(
    OpAsmParser &p, Type &tensor, DeviceRefAttr &device, Type &buffer,
    Type &tensorSpec, ArrayRef<Type *> chains) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(tensor))
    return failure();
  auto tensorType = dyn_cast<DeviceTensorType>(tensor);
  if (tensorType == nullptr)
    return p.emitError(loc) << "expected an !mgp.tensor type.";
  device = tensorType.getDevice();

  tensorSpec = tensorType.getTensorSpec();
  buffer = DeviceBufferType::get(tensorType);
  for (Type *chain : chains)
    *chain = MEF::ChainType::get(p.getContext());
  return success();
}

static ParseResult parseTensorTypeToDeviceBufferSpec(OpAsmParser &p,
                                                     Type &tensor, Type &buffer,
                                                     Type &tensorSpec,
                                                     Type &chain) {
  DeviceRefAttr device;
  return parseTensorTypeToDeviceBufferSpecImpl(p, tensor, device, buffer,
                                               tensorSpec, {&chain});
}

static void printTensorTypeToDeviceBufferSpec(OpAsmPrinter &p, Operation *op,
                                              DeviceTensorType tensorType,
                                              DeviceBufferType buffer,
                                              TensorSpecType tensorSpec,
                                              MEF::ChainType chain) {
  p << tensorType;
}

/// Parses a function type consisting of one input buffer and one output buffer
/// type, assigns their devices to srcDevice and dstDevice, and assigns inChain
/// the type !mef.chain.
static ParseResult
parseSrcDstBufferTypesToDevices(OpAsmParser &p, Type &srcBuffer,
                                Type &dstBuffer, DeviceRefAttr &srcDevice,
                                DeviceRefAttr &dstDevice, Type &inChain) {
  mlir::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(srcBuffer))
    return failure();
  auto srcBufferType = dyn_cast<DeviceBufferType>(srcBuffer);
  if (!srcBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  if (p.parseArrow())
    return failure();
  if (p.getCurrentLocation(&loc) || p.parseType(dstBuffer))
    return failure();
  auto dstBufferType = dyn_cast<DeviceBufferType>(dstBuffer);
  if (!dstBufferType)
    return p.emitError(loc, "expected an !mgp.buffer type");
  srcDevice = srcBufferType.getDevice();
  dstDevice = dstBufferType.getDevice();
  inChain = MEF::ChainType::get(p.getContext());
  return success();
}

static void printSrcDstBufferTypesToDevices(OpAsmPrinter &p, Operation *op,
                                            DeviceBufferType srcBuffer,
                                            DeviceBufferType dstBuffer,
                                            DeviceRefAttr srcDevice,
                                            DeviceRefAttr dstDevice,
                                            MEF::ChainType inChain) {
  p << srcBuffer;
  p << " -> ";
  p << dstBuffer;
}

//===----------------------------------------------------------------------===//
// ResourceConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ResourceConstantOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl =
      state.registry.lookupPrimitive(BufferConstantOp::getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));
  auto &[ptr, bytecount] =
      state.mojoConstantResources[getResourceSlot().getZExtValue()];
  MOGG::MojoValue result = state.builder.var(function({}, {ptr, bytecount}));
  state.setMojoValueFor(getOutBuffer(), result);

  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Parses an elements attribute, and sets rawAlign to its alignment constraint.
static ParseResult parseElementsAttrWithAlign(AsmParser &p, ElementsAttr &value,
                                              IntegerAttr &rawAlign) {
  llvm::SMLoc loc;
  Attribute anyAttr;
  if (p.getCurrentLocation(&loc) || p.parseAttribute(anyAttr))
    return failure();
  value = dyn_cast<ElementsAttr>(anyAttr);
  if (!value)
    return p.emitError(loc, "expected elements attribute");

  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (const char *err = errOrOptAlign.getError())
    return p.emitError(loc, err);

  Type ui64 = IntegerType::get(p.getContext(), 64, IntegerType::Unsigned);
  APInt apint(ui64.getIntOrFloatBitWidth(), optSizeToRawSize(*errOrOptAlign),
              ui64.isSignedInteger());
  rawAlign = IntegerAttr::get(ui64, apint);
  return success();
}

static void printElementsAttrWithAlign(AsmPrinter &p, Operation *op,
                                       ElementsAttr value,
                                       IntegerAttr rawAlign) {
  p << value;
}

LogicalResult BufferConstantOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  auto elementsAttr = getValue();
  auto kernelOp = getOperation()->getParentOfType<MOGG::FuncOp>();
  auto constantResources = kernelOp.getConstantResources();
  auto it = llvm::find(constantResources, elementsAttr);
  if (it == constantResources.end())
    return emitError("The constant resource can't be found in the kernel");

  uint64_t resourceIndex = std::distance(constantResources.begin(), it);
  auto &[ptr, bytecount] = state.mojoConstantResources[resourceIndex];

  MOGG::MojoValue result = state.builder.var(function({}, {ptr, bytecount}));
  state.setMojoValueFor(getOutBuffer(), result);

  return success();
}

//===----------------------------------------------------------------------===//
// TensorSpecCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSpecCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  TensorSpecType tensorSpec = getResult().getType();
  size_t rank = tensorSpec.getRank();
  MojoValue rawDimsRankParam = rank;

  // Emit dims as an IntTuple, mapping dynamic dims to -1 (UNKNOWN_VALUE).
  SmallVector<int64_t> dimValues;
  dimValues.reserve(rank);
  llvm::transform(tensorSpec.getDims(), std::back_inserter(dimValues),
                  [](const std::optional<int64_t> &d) -> int64_t {
                    return d.value_or(-1);
                  });
  MojoValue rawDimsParam = EmitMojoState::getAsIntTupleMojoValue(dimValues);

  MojoValue result = state.builder.var(function(
      {rawDimsParam, rawDimsRankParam}, state.getMojoValuesFor(getOperands())));
  state.setMojoValueFor(getResult(), result);

  return success();
}
void TensorSpecCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               TensorSpecType spec, ValueRange runtimeDims) {
  TensorSpecCreateOp::build(odsBuilder, odsState,
                            /*runtimeDims=*/runtimeDims,
                            /*rawDims=*/spec.getRawDims(),
                            /*rawDType=*/spec.getRawDType());
}

/// Parses a !mgp.tensor_spec type where the element type must be known,
/// but represented in 'raw form'.
static ParseResult parseRawDimsAndDType(AsmParser &p,
                                        IntArrayElementsAttr &rawDims,
                                        IntegerAttr &rawDType) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mgp.tensor_spec type.";
  if (!tensorSpec.getOptDType())
    return p.emitError(loc) << "expected !mgp.tensor_spec type to have a "
                               "known element type.";

  rawDims = tensorSpec.getRawDims();
  rawDType = tensorSpec.getRawDType();

  return success();
}

/// Prints a !mgp.tensor_spec type, but represented in 'raw form'.
static void printRawDimsAndDType(AsmPrinter &p, const TensorSpecCreateOp &op,
                                 IntArrayElementsAttr rawDims,
                                 IntegerAttr rawDType) {
  p << TensorSpecType::get(rawDims, rawDType);
}

LogicalResult TensorSpecCreateOp::verify() {
  auto impliedSpecType =
      TensorSpecType::getChecked([this]() { return emitOpError(); },
                                 getContext(), getRawDims(), getRawDType());
  if (impliedSpecType == nullptr)
    return failure();
  TensorSpecType actualSpecType = getResult().getType();
  if (actualSpecType != impliedSpecType)
    return emitOpError()
           << "expected result type is " << actualSpecType
           << ", but static dimensions and element type arguments "
              "suggest it should be "
           << impliedSpecType << ".";
  if (getRuntimeDims().size() != impliedSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension operands to match " << actualSpecType
                         << ", but given " << getRuntimeDims().size() << ".";
  return success();
}

LogicalResult TensorSpecCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto rawDimsAttr = adaptor.getRawDimsAttr();
  auto rawDTypeAttr = adaptor.getRawDTypeAttr();
  ASSERT_STREAM(rawDimsAttr && rawDTypeAttr, << "mismatched operand type");
  inferredReturnTypes.emplace_back(
      TensorSpecType::get(rawDimsAttr, rawDTypeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferGetSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult BufferGetSizeOp::fold(FoldAdaptor adaptor) {
  auto bufferAlloc = getBuffer().getDefiningOp<BufferAllocOp>();
  if (bufferAlloc)
    return bufferAlloc.getRuntimeSize();

  auto bufferConstant = getBuffer().getDefiningOp<BufferConstantOp>();
  if (bufferConstant)
    return IntegerAttr::get(IndexType::get(getContext()),
                            bufferConstant.getValue().size());

  return {};
}

LogicalResult BufferGetSizeOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue getSizeFunc(primitiveMojoName(primDecl));

  MojoValue bufferVar = state.getMojoValueFor(getBuffer());
  state.setMojoValueFor(getResult(), getSizeFunc({}, {bufferVar}));

  return success();
}

//===----------------------------------------------------------------------===//
// TensorSpecGetDimOp
//===----------------------------------------------------------------------===//

LogicalResult TensorSpecGetDimOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue getDimFunc(primitiveMojoName(primDecl));

  MOGG::MojoValue axisParam = MOGG::MojoValue(getAxis());
  MOGG::MojoValue rankParam = getSpec().getType().getRank();
  MOGG::MojoValue specVar = state.getMojoValueFor(getSpec());

  state.setMojoValueFor(getResult(), state.builder.var(getDimFunc(
                                         {rankParam, axisParam}, {specVar})));

  return success();
}

LogicalResult TensorSpecGetDimOp::verify() {
  TensorSpecType specType = getSpec().getType();
  if (getAxis() >= specType.getRank())
    return emitOpError() << "expected axis " << getAxis()
                         << " to be less than the rank of the tensor spec "
                         << specType.getRank();
  return success();
}

OpFoldResult TensorSpecGetDimOp::fold(FoldAdaptor adaptor) {
  Operation *specProducer = getSpec().getDefiningOp();
  TensorSpecType specType = getSpec().getType();
  if (getAxis() >= specType.getRank())
    return {};
  if (!specType.isDynamicDim(getAxis())) {
    auto indexTy = IndexType::get(getContext());
    uint64_t dim = specType.getDims()[getAxis()].value();
    return IntegerAttr::get(indexTy, dim);
  }
  if (auto specCreate = dyn_cast_or_null<TensorSpecCreateOp>(specProducer)) {
    return specCreate.getRuntimeDims()[specType.getDynamicDimIndex(getAxis())];
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

LogicalResult BufferGetCachedOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue mgpBufferGetCacheFunc(primitiveMojoName(primDecl));
  auto res = state.builder.var(
      mgpBufferGetCacheFunc({}, {state.getMojoValueFor(getTheContext()),
                                 state.getMojoValueFor(getBBufferSlot())}));
  // The primitive returns a single storage-carrying `OwnedByteBuffer`.
  state.setMojoValueFor(getResult(0), res);

  return success();
}

LogicalResult BufferGetCachedOp::canonicalize(BufferGetCachedOp op,
                                              PatternRewriter &rewriter) {
  auto memStorageHandle = op.getOptionalMemStorageHandle();
  if (memStorageHandle && memStorageHandle.use_empty()) {
    auto cachedBufferOp = BufferGetCachedOp::create(
        rewriter, op->getLoc(), op.getOutBuffer().getType(), op->getOperands(),
        op->getAttrs());
    rewriter.replaceAllUsesWith(op.getOutBuffer(),
                                cachedBufferOp.getOutBuffer());
    rewriter.eraseOp(op);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IntCacheOp
//===----------------------------------------------------------------------===//

LogicalResult IntCacheOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));
  MOGG::MojoValue bIntSlotParam = MOGG::MojoValue(getBIntSlot());
  state.builder.call(function, {bIntSlotParam},
                     {state.getMojoValueFor(getTheContext()),
                      state.getMojoValueFor(getValue())});
  return success();
}

//===----------------------------------------------------------------------===//
// IntGetCachedOp
//===----------------------------------------------------------------------===//

LogicalResult IntGetCachedOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));
  auto res =
      state.builder.var(function({}, {state.getMojoValueFor(getTheContext()),
                                      state.getMojoValueFor(getBIntSlot())}));
  state.setMojoValueFor(getResult(), res);
  return success();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

bool AssertOp::isThrows() { return true; }

LogicalResult AssertOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  MOGG::MojoValue conditionVar = state.getMojoValueFor(getCond());
  MOGG::MojoValue messageLiteral = MOGG::MojoValue(getMessage()).quote();
  MOGG::MojoValue messageString = state.builder.var(
      MOGG::MojoValue(MOGG::kStructString)({}, {messageLiteral}));
  MOGG::MojoStringRef message(messageString);
  MOGG::MojoValue messagePtr = message.unsafePtr();
  MOGG::MojoValue messageSize = message.byteLength();

  state.builder.call(function, {}, {conditionVar, messagePtr, messageSize});

  return success();
}

SmallVector<Attribute> AssertOp::getLiftableAttributes() {
  return {getMessageAttr()};
}

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  bool isInKernel = false;
  if (op->getParentOfType<MOGG::AsyncRoutineOp>())
    isInKernel = true;

  if (op->getParentOfType<MOGG::FuncOp>())
    isInKernel = true;

  BoolAttr condAttr;
  if (mlir::matchPattern(op.getCond(), mlir::m_Constant(&condAttr))) {
    if (condAttr.getValue()) {
      if (isInKernel) {
        rewriter.replaceOpWithNewOp<MOGG::ChainDummyOp>(op);
      } else {
        rewriter.replaceOpWithNewOp<GRT::ChainCreateOp>(op);
      }
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//
LogicalResult TensorExtractBufferOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  MojoValue res =
      state.builder.var(function({}, state.getMojoValueFor(getTensor())));
  state.setMojoValueFor(getOutBuffer(), res);

  return success();
}

LogicalResult TensorExtractBufferOp::verify() {
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getTensor().getType());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected the result buffer to match the type "
                       "implied by the tensor type (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return success();
}

LogicalResult TensorExtractBufferOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = dyn_cast<DeviceTensorType>(adaptor.getTensor().getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(adaptor.getTensor().getLoc()),
                           "expected tensor operand.");
  }
  inferredReturnTypes.emplace_back(DeviceBufferType::get(tensorType));
  return success();
}

bool TensorExtractBufferOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

OpFoldResult TensorExtractBufferOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getBuffer();

  return {};
}

//===----------------------------------------------------------------------===//
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//
LogicalResult TensorExtractTensorSpecOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  DeviceTensorType tensorType = getTensor().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  auto tensorSpec = tensorType.getTensorSpec();
  MojoValue tensorDtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  MojoValue result = state.builder.var(
      function({tensorRankParam, bufferRankParam, tensorDtypeParam},
               state.getMojoValueFor(getTensor())));
  state.setMojoValueFor(getResult(), result);
  return success();
}

LogicalResult TensorExtractTensorSpecOp::verify() {
  TensorSpecType expectedSpecType = getTensor().getType().getTensorSpec();
  if (expectedSpecType != getOutSpec().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutSpec().getType() << " vs " << expectedSpecType << ").";
  }
  if (!expectedSpecType.getOptDType().has_value())
    return emitError() << "Expected to have DType information for operand";
  return success();
}

LogicalResult TensorExtractTensorSpecOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto tensorType = dyn_cast<DeviceTensorType>(adaptor.getTensor().getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(adaptor.getTensor().getLoc()),
                           "expected tensor operand.");
  }
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec());
  return success();
}

OpFoldResult TensorExtractTensorSpecOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getTensorSpec();
  return {};
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

LogicalResult GuardOp::verify() {
  if (getInputs().size() != getResults().size()) {
    return emitError() << "Expected the number of inputs and outputs to be "
                          "exactly the same";
  }
  return success();
}

LogicalResult
GuardOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  TypeRange inputTypes = adaptor.getInputs().getTypes();
  inferredReturnTypes.append(inputTypes.begin(), inputTypes.end());
  return success();
}

bool GuardOp::mayAlias(size_t operandIndex, size_t resultIndex) { return true; }

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  DeviceTensorType tensorType = getOutTensor().getType();
  MojoValue tensorRankParam = tensorType.getRank();
  MojoValue bufferRankParam =
      tensorType.getRank() == 0 ? MojoValue{1} : tensorRankParam;
  auto tensorSpec = tensorType.getTensorSpec();
  MojoValue tensorDtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  SmallVector<MojoValue> params = {tensorRankParam, bufferRankParam,
                                   tensorDtypeParam};
  // `kind` names how the tensor participates as a device graph input; the
  // Mojo default (`data`) is left implicit.
  if (tensorType.getMut())
    params.push_back(MojoValue(MOGG::kParameterKind)
                         .equal(MojoValue(MOGG::kStructDeviceGraphInputKind)
                                    .get(MOGG::kInputKindMutable)));
  else if (M::isHostPlaced(tensorType.getDevice()))
    params.push_back(MojoValue(MOGG::kParameterKind)
                         .equal(MojoValue(MOGG::kStructDeviceGraphInputKind)
                                    .get(MOGG::kInputKindHost)));

  MojoValue result = state.builder.var(
      function(params, state.getMojoValuesFor(getOperands().drop_front())));
  state.setMojoValueFor(getOutTensor(), result);

  return success();
}

bool TensorCreateOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// TensorSliceOp
//===----------------------------------------------------------------------===//

bool TensorSliceOp::isThrows() { return true; }

LogicalResult TensorSliceOp::emitMojo(MOGG::EmitMojoState &state) {
  using namespace MOGG;

  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MojoValue function(primitiveMojoName(primDecl));

  DeviceTensorType inputType = getInput().getType();
  MojoValue rankParam = inputType.getRank();
  auto tensorSpec = inputType.getTensorSpec();
  MojoValue dtypeParam =
      state.mlirDTypeToMojoType(tensorSpec.getOptDType().value());

  // input, outputSpec, and start are all runtime operands.
  MojoValue result = state.builder.var(
      function({rankParam, dtypeParam}, state.getMojoValuesFor(getOperands())));
  state.setMojoValueFor(getOutput(), result);

  return success();
}

LogicalResult TensorSliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = dyn_cast<DeviceTensorType>(adaptor.getInput().getType());
  if (!inputType)
    return mlir::emitError(location.value_or(UnknownLoc::get(context)),
                           "expected tensor operand");
  auto specType = dyn_cast<TensorSpecType>(adaptor.getOutputSpec().getType());
  if (!specType)
    return mlir::emitError(location.value_or(UnknownLoc::get(context)),
                           "expected tensor spec operand");
  inferredReturnTypes.emplace_back(
      DeviceTensorType::get(context, inputType.getDevice(), specType,
                            inputType.getMut(), inputType.getStable()));
  return success();
}

LogicalResult TensorSliceOp::verify() {
  DeviceTensorType inputType = getInput().getType();
  TensorSpecType specType = getOutputSpec().getType();
  if (inputType.getTensorSpec().getOptDType() != specType.getOptDType())
    return emitOpError("input and output dtypes must match");
  DeviceTensorType startType = getStart().getType();
  if (startType.getDevice().getLabel() != kCPULabel)
    return emitOpError("start must be on the host (cpu) device");
  if (getDeviceContext().getType().getDevice() != inputType.getDevice())
    return emitOpError(
        "device context must be on the same device as the input");
  return success();
}

bool TensorSliceOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getInputMutable().getOperandNumber();
}

bool TensorSliceOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getInputMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

bool TensorExtractBufferOp::extendsLifetime(size_t operandIndex,
                                            size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto targetModel =
      symbolTable.lookupNearestSymbolFrom<ModelOp>(*this, getCalleeAttr());
  if (!targetModel)
    return emitOpError() << "failed to find symbol " << getCalleeAttr();

  Block *executeBlock = targetModel.getExecuteBlock();
  // Until the index is globalized, the execute block's last arg is the index.
  auto expectedArgs =
      executeBlock->getArguments().drop_back(!getIsGlobalizedIndex());
  auto actualArgTypes = getOperandTypes();
  if (expectedArgs.size() != actualArgTypes.size()) {
    return emitOpError() << "operand types " << actualArgTypes
                         << " do not match expected argument types "
                         << executeBlock->getArgumentTypes() << " of "
                         << getCalleeAttr() << "'s execute block.";
  }

  for (auto [expectedArg, actualArgType] :
       llvm::zip(expectedArgs, actualArgTypes)) {
    if (expectedArg.getType() != actualArgType) {
      return emitOpError() << "operand types " << actualArgTypes
                           << " do not match expected argument types "
                           << executeBlock->getArgumentTypes() << " of "
                           << getCalleeAttr() << "'s execute block.";
    }
  }

  auto outputOp = dyn_cast<MEF::OutputOp>(executeBlock->getTerminator());
  if (!outputOp) {
    return emitOpError() << "execute block of " << getCalleeAttr()
                         << " does not end with an mef.output op.";
  }

  auto expectedResultTypes = outputOp.getOperandTypes();
  auto actualResultTypes = getResultTypes();
  if (actualResultTypes != expectedResultTypes) {
    return emitOpError() << "result types " << actualResultTypes
                         << " do not match expected result types "
                         << expectedResultTypes << " of " << getCalleeAttr()
                         << "'s execute block.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//

bool DebugPrintOp::isThrows() { return true; }

LogicalResult DebugPrintOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  MOGG::MojoValue debugStringParam = MOGG::MojoValue(getDebugString()).quote();
  MOGG::MojoValue labelParam = MOGG::MojoValue(getLabel()).quote();
  MOGG::MojoValue ctxVar = state.getMojoValueFor(getTheContext());

  state.builder.call(function, {debugStringParam, labelParam}, ctxVar);

  return success();
}

//===----------------------------------------------------------------------===//
// DebugPrintIntOp
//===----------------------------------------------------------------------===//

LogicalResult DebugPrintIntOp::emitMojo(MOGG::EmitMojoState &state) {
  auto primDecl = state.registry.lookupPrimitive(getOperationName());
  MOGG::MojoValue function(primitiveMojoName(primDecl));

  MOGG::MojoValue labelParam = MOGG::MojoValue(getALabel()).quote();
  MOGG::MojoValue ctxVar = state.getMojoValueFor(getTheContext());
  MOGG::MojoValue valueVar = state.getMojoValueFor(getValue());

  state.builder.call(function, {labelParam}, {ctxVar, valueVar});

  return success();
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

// Builds an !mgp.opaque type with the given symbol name.
static OpaqueType opaqueType(MLIRContext *ctx, StringRef symbol) {
  return OpaqueType::get(ctx, StringAttr::get(ctx, symbol),
                         /*metadata=*/{});
}

// Extends args with the weights registry and device context implicit args.
static void appendInitExecImplicitArgs(SmallVectorImpl<Type> &args,
                                       MLIRContext *ctx,
                                       ArrayRef<DeviceRef> devices) {
  args.push_back(opaqueType(ctx, "weights_registry"));

  for (const DeviceRef &device : devices) {
    args.push_back(DeviceContextType::get(
        ctx, DeviceRefAttr::get(ctx, device.label, device.id)));
  }
}

/// Returns the implicit signature of the 'init' entry point.
static FunctionType implicitInitSignature(Builder &builder,
                                          ArrayRef<DeviceRef> devices) {
  SmallVector<Type> inputs;
  appendInitExecImplicitArgs(inputs, builder.getContext(), devices);

  SmallVector<Type> outputs = {builder.getType<MEF::ChainType>(),
                               builder.getType<MGP::ContextType>()};
  appendInitExecImplicitArgs(outputs, builder.getContext(), devices);

  return builder.getFunctionType(inputs, outputs);
}

/// Returns the implicit signature of the 'execute' entry point, before
/// any explicit inputs and results have been included.
static FunctionType implicitExecuteSignature(Builder &builder,
                                             ArrayRef<DeviceRef> devices) {
  SmallVector<Type> inputs = {builder.getType<MEF::ChainType>(),
                              builder.getType<MGP::ContextType>()};

  appendInitExecImplicitArgs(inputs, builder.getContext(), devices);

  return builder.getFunctionType(
      inputs, /*results=*/{builder.getType<MEF::ChainType>()});
}

/// Returns true if requiredTypes is a prefix of actualTypes, in which case
/// the prefix is removed from actualTypes.
static bool checkAndRemovePrefix(ArrayRef<Type> requiredTypes,
                                 SmallVectorImpl<Type> &actualTypes) {
  auto [mismatchRequired, mismatchActual] = std::mismatch(
      requiredTypes.begin(), requiredTypes.end(), actualTypes.begin());
  if (mismatchRequired != requiredTypes.end())
    return false;
  actualTypes.erase(actualTypes.begin(),
                    actualTypes.begin() + requiredTypes.size());
  return true;
}

/// Parse an optional symbol visibility keyword, mirroring
/// `function_interface_impl::parseFunctionOp`.
static ParseResult parseSymbolVisibility(OpAsmParser &p,
                                         StringAttr &visibility) {
  StringRef keyword;
  if (succeeded(
          p.parseOptionalKeyword(&keyword, {"public", "private", "nested"})))
    visibility = p.getBuilder().getStringAttr(keyword);
  return success();
}

/// Prints the symbol visibility keyword, if present.
static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr visibility) {
  if (visibility)
    p << visibility.getValue() << ' ';
}

/// Parse the frameworkName attribute as a keyword.
static ParseResult parseClassName(OpAsmParser &p, StringAttr &frameworkName) {
  StringRef keyword;
  if (p.parseKeyword(&keyword))
    return failure();
  frameworkName = StringAttr::get(p.getContext(), keyword);
  return success();
}

/// Prints the frameworkName attribute as a keyword.
static void printClassName(OpAsmPrinter &p, const ModelOp &op,
                           StringAttr frameworkName) {
  p << frameworkName.getValue();
}

/// Parse the init region.
static ParseResult parseInitRegion(OpAsmParser &p, Region &initRegion) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;
  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseRegion(initRegion, args, /*enableNameShadowing=*/true))
    return failure();

  return success();
}

/// Print the init region.
static void printInitRegion(OpAsmPrinter &p, const ModelOp &op,
                            Region &initRegion) {
  Block &initBlock = initRegion.front();
  p << "(";
  llvm::interleaveComma(initBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(initRegion, /*printEntryBlockArgs=*/false);
}

/// Parses the 'execute' body, and extracts from it the implicit signature.
static ParseResult parseExecuteRegion(OpAsmParser &p,
                                      TypeAttr &inputOutputTypes,
                                      Region &executeRegion,
                                      ArrayAttr &deviceAttrs) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;
  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseArrowTypeList(outputTypes) ||
      p.parseRegion(executeRegion, args, /*enableNameShadowing=*/true))
    return failure();

  // Strip the implicit prefix; the rest becomes inputOutputTypes.
  SmallVector<DeviceRef> devices;
  for (auto attr : deviceAttrs) {
    if (auto deviceAttr = dyn_cast<DeviceRefAttr>(attr))
      devices.emplace_back(
          DeviceRef(deviceAttr.getLabel().str(), deviceAttr.getId()));
  }
  FunctionType implicitExecSig =
      implicitExecuteSignature(p.getBuilder(), devices);
  SmallVector<Type> inputTypes(executeRegion.getArgumentTypes());
  if (!checkAndRemovePrefix(implicitExecSig.getInputs(), inputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region arguments to begin with types "
           << implicitExecSig.getInputs() << ".";
  if (!checkAndRemovePrefix(implicitExecSig.getResults(), outputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region outputs to begin with types "
           << implicitExecSig.getResults() << ".";

  inputOutputTypes =
      TypeAttr::get(p.getBuilder().getFunctionType(inputTypes, outputTypes));
  return success();
}

/// Prints the 'execute' body. The inputOutputTypes are ignored.
static void printExecuteRegion(OpAsmPrinter &p, const ModelOp &op,
                               TypeAttr inputOutputTypes, Region &executeRegion,
                               ArrayAttr devices) {
  Block &executeBlock = executeRegion.front();
  p << "(";
  llvm::interleaveComma(executeBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printArrowTypeList(executeBlock.getTerminator()->getOperandTypes());
  p << " ";
  p.printRegion(executeRegion, /*printEntryBlockArgs=*/false);
}

void ModelOp::build(OpBuilder &builder, OperationState &state, StringAttr name,
                    StringRef frameworkName, ArrayRef<DeviceRef> devices,
                    ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
                    ArrayRef<Location> inputArgLocs, bool isSubgraph,
                    bool isDeviceGraph, ArrayRef<NamedAttribute> extraAttrs) {
  MLIRContext *ctx = state.getContext();

  SmallVector<Type> finalInputTypes(inputTypes.begin(), inputTypes.end());
  SmallVector<Location> finalInputLocs(inputArgLocs.begin(),
                                       inputArgLocs.end());
  if (isSubgraph) {
    state.addAttribute(getIsSubgraphAttrName(state.name), UnitAttr::get(ctx));
    finalInputTypes.push_back(IndexType::get(ctx));
    finalInputLocs.push_back(state.location);
  }

  DictionaryAttr extraAttrsDict =
      extraAttrs.empty() ? nullptr : DictionaryAttr::get(ctx, extraAttrs);
  auto deviceAttrs = llvm::map_to_vector(devices, [ctx](const DeviceRef &ref) {
    return dyn_cast<Attribute>(DeviceRefAttr::get(ctx, ref.label, ref.id));
  });
  build(builder, state, ArrayAttr::get(ctx, deviceAttrs),
        StringAttr::get(ctx, frameworkName),
        TypeAttr::get(builder.getFunctionType(finalInputTypes, outputTypes)),
        name, /*sym_visibility=*/nullptr,
        isSubgraph ? builder.getUnitAttr() : nullptr,
        isDeviceGraph ? builder.getUnitAttr() : nullptr, extraAttrsDict);

  Region *initRegion = state.regions[0].get();
  initRegion->emplaceBlock();

  for (auto type : implicitInitSignature(builder, devices).getInputs())
    initRegion->addArgument(type, state.location);

  // The 'execute' entry point has both implicit and explicit arguments.
  Region *executeRegion = state.regions[1].get();
  executeRegion->emplaceBlock();
  for (auto type : implicitExecuteSignature(builder, devices).getInputs())
    executeRegion->addArgument(type, state.location);
  for (auto [type, loc] : llvm::zip(finalInputTypes, finalInputLocs))
    executeRegion->addArgument(type, loc);
}

/// Verifies the init and execute regions of the model.
LogicalResult ModelOp::verifyRegions() {
  using BlockArgTypeRange = mlir::ValueTypeRange<mlir::Block::BlockArgListType>;

  Builder builder(getContext());

  SmallVector<DeviceRef> devices;
  for (auto attr : getDevices()) {
    if (auto deviceAttr = dyn_cast<DeviceRefAttr>(attr))
      devices.emplace_back(
          DeviceRef(deviceAttr.getLabel().str(), deviceAttr.getId()));
  }

  // The type of the init region is
  //  () -> (!mef.chain, !mgp.context)
  Block *initBlock = getInitBlock();
  BlockArgTypeRange initArgTypes = initBlock->getArgumentTypes();
  if (auto output = dyn_cast<MEF::OutputOp>(initBlock->getTerminator())) {
    FunctionType implicitInitSig = implicitInitSignature(builder, devices);

    // One arg per device plus the weights registry.
    size_t numDeviceArgs = initArgTypes.size() - 1;
    if (numDeviceArgs != getDevices().size()) {
      return emitOpError() << "Expected the number of arguments to the 'init' "
                              "body to match the number of devices, but got "
                           << numDeviceArgs << " vs " << getDevices().size()
                           << ".";
    }
    for (auto t : initArgTypes) {
      if (!mlir::isa<OpaqueType, DeviceContextType>(t)) {
        return emitOpError() << "init block arguments must be !mgp.opaque or "
                                "!mgp.device_context, but got "
                             << t;
      }
    }

    int contextCreateOpCount = 0;
    initBlock->walk([&](ContextCreateOp) { contextCreateOpCount++; });
    if (contextCreateOpCount != 1) {
      return emitOpError()
             << "expected exactly one mgp.context_create op in the "
                "'init' block, but found "
             << contextCreateOpCount;
    }

    if (failed(checkOperandTypes(output, implicitInitSig.getResults())))
      return failure();
  }

  // The type of the execute region is:
  //   (!mef.chain, !mgp.context, weights_registry, dev_1, ..., dev_n, arg_1,
  //   ... arg_n) ->
  //   (!mef.chain, res_1, ... res_m)
  // where the ModelOp's inputOutputTypes are:
  //   (arg_1, ... arg_n) -> (res_1, ... res_m)
  Block *execBlock = getExecuteBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(execBlock->getTerminator())) {
    FunctionType implicitExecSig = implicitExecuteSignature(builder, devices);
    ArrayRef<Type> modelArgTypes = getInputOutputTypes().getInputs();
    SmallVector<Type> expectedArgTypes(implicitExecSig.getInputs());
    expectedArgTypes.append(modelArgTypes.begin(), modelArgTypes.end());
    if (execBlock->getArgumentTypes() != expectedArgTypes) {
      return emitOpError() << "expected the arguments of the "
                              "'execute' body to be "
                           << expectedArgTypes << ", but got "
                           << execBlock->getArgumentTypes() << ".";
    }

    ArrayRef<Type> modelResultTypes = getInputOutputTypes().getResults();
    SmallVector<Type> expectedResultTypes(implicitExecSig.getResults());
    expectedResultTypes.append(modelResultTypes.begin(),
                               modelResultTypes.end());

    if (failed(checkOperandTypes(output, expectedResultTypes)))
      return failure();
  }

  if (getIsSubgraph()) {
    auto argTypes = execBlock->getArgumentTypes();
    if (argTypes.empty() ||
        !isa<IndexType>(*(argTypes.begin() + argTypes.size() - 1))) {
      return emitOpError() << "expected the last argument of a subgraph to be "
                              "an index offset";
    }
  }

  return success();
}

LogicalResult ModelOp::verify() { return success(); }

ContextCreateOp ModelOp::getInitContext() {
  auto range = getInitBlock()->getOps<ContextCreateOp>();
  ASSERT_STREAM(llvm::hasSingleElement(range),
                "should contain one context create op");
  return *std::begin(range);
}

TypedValue<ContextType> ModelOp::getExecuteContext() {
  auto argument = getExecuteBlock()->getArgument(1);
  return cast<TypedValue<ContextType>>(argument);
}

//===----------------------------------------------------------------------===//
// MGPDialect op support
//===----------------------------------------------------------------------===//

void MGPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MGPDialect/Dialect/MGP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

using PointerType = KGEN::PointerType;
#define GET_OP_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGP.cpp.inc"
