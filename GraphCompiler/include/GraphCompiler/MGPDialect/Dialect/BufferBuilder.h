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
// Support for working with device-dependent buffer and tensor values and
// types.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_BUFFERBUILDER_H
#define GENERICML_MGPDIALECT_BUFFERBUILDER_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// Generic parsing/printing helpers
//===----------------------------------------------------------------------===//

/// Parses one off:
///   ':' resultRawType
///   bufferRawType ':' resultRawType
/// In the first form the bufferRawType is derived from the resultRawType.
template <typename BufferType, typename TensorType>
ParseResult genericParseMayBeDynamicBuffer(AsmParser &parser,
                                           Type &bufferRawType,
                                           Type &resultRawType) {
  mlir::SMLoc loc;
  if (parser.getCurrentLocation(&loc))
    return failure();
  OptionalParseResult parseResult = parser.parseOptionalType(bufferRawType);
  bool hasBufferTypeAnnotation = parseResult.has_value();
  if (hasBufferTypeAnnotation) {
    if (failed(parseResult.value()))
      return failure();
    if (!isa<BufferType>(bufferRawType))
      return parser.emitError(loc)
             << "expecting !" << BufferType::getMnemonic() << " type";
  }

  if (parser.parseColon() || parser.parseType(resultRawType))
    return failure();
  auto resultTensorType = dyn_cast<TensorType>(resultRawType);
  if (resultTensorType == nullptr)
    return parser.emitError(loc)
           << "expecting !" << TensorType::getMnemonic() << " type";
  if (!hasBufferTypeAnnotation) {
    bufferRawType = BufferType::get(resultTensorType);
  }
  return success();
}

/// Prints one off:
///   ':' resultType
///   bufferType ':' resultType
/// The second form is used only if the bufferType does not match the result
/// type.
template <typename BufferType, typename TensorType>
void genericPrintMayBeDynamicBuffer(AsmPrinter &printer, const Operation *op,
                                    BufferType bufferType,
                                    TensorType resultType) {
  if (BufferType::get(resultType) != bufferType) {
    printer.printType(bufferType);
    printer.getStream() << ' ';
  }
  printer.getStream() << ": ";
  printer.printType(resultType);
}

//===----------------------------------------------------------------------===//
// Generic verification helpers
//===----------------------------------------------------------------------===//

/// Checks the tensor spec, buffer type and result tensor types are consistent.
template <typename BufferType, typename TensorType>
LogicalResult genericTensorCreateVerify(Operation *op, BufferType bufferType,
                                        TensorSpecType tensorSpecType,
                                        TensorType tensorType) {
  if (tensorSpecType != tensorType.getTensorSpec())
    return op->emitOpError()
           << "expected spec to match result tensor spec" << " ("
           << tensorSpecType << " vs " << tensorType.getTensorSpec() << ")";

  BufferType expectedBufferType = BufferType::get(tensorType);
  bool bufferIsStatic = bufferType.getOptSize().has_value();
  bool expectedBufferIsStatic = expectedBufferType.getOptSize().has_value();
  if (bufferIsStatic != expectedBufferIsStatic) {
    // Buffer and tensor spec may disagree on staticness when lowering
    // value-semantic reshapes. Here %z reuses %x's buffer with a known size:
    //   %z = reshape %x, %y : (tensor<?x?xf32>, tensor<2xui32>) ->
    //   tensor<8x16xf32>
    // Conversely, in:
    //   %z = reshape %x, %y : (tensor<8x4xf32>, tensor<2xui32>) ->
    //   tensor<?x?xf32>
    // the buffer for %z will now have an unknown size.
  } else if (bufferType != expectedBufferType) {
    return op->emitOpError()
           << "expected buffer type to match type implied by tensor spec ("
           << bufferType << " vs " << expectedBufferType << ").";
  }
  return success();
}

/// Checks the buffer type and tensor types are consistent.
template <typename BufferType, typename TensorType>
LogicalResult genericTensorExtractVerify(Operation *op, TensorType tensorType,
                                         BufferType resultBufferType,
                                         TensorSpecType resultTensorSpecType) {
  if (tensorType.getTensorSpec() != resultTensorSpecType)
    return op->emitOpError()
           << "expected the result spec to match tensor's spec" << " ("
           << resultTensorSpecType << " vs " << tensorType.getTensorSpec()
           << ")";
  BufferType expectedBufferType = BufferType::get(tensorType);
  if (expectedBufferType != resultBufferType)
    return op->emitOpError()
           << "expected the result buffer to match the type "
              "implied by the tensor type ("
           << resultBufferType << " vs " << expectedBufferType << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// MGPTypeBuilder
//===----------------------------------------------------------------------===//

/// An MGPTypeBuilder provides methods for building device-specific tensor
/// types. It is only needed during lowering when lowered types must be
/// provided ahead of lowering itself. Otherwise the MGPBufferBuilder
/// constructs types as it goes.
class MGPTypeBuilder {
public:
  DeviceRefAttr device;

  MGPTypeBuilder(DeviceRefAttr device) : device(device) {}

  /// Returns the device-specific tensor type with the given tensor spec.
  /// `mut` marks the tensor as mutable (the lowered form of `!mo.buffer`);
  /// `stable` marks the tensor as having a stable address across dispatches.
  DeviceTensorType tensorType(TensorSpecType tensorSpecType, bool mut = false,
                              bool stable = false) {
    return DeviceTensorType::get(device, tensorSpecType, mut, stable);
  }
};

//===----------------------------------------------------------------------===//
// MGPBufferBuilder
//===----------------------------------------------------------------------===//

/// An MGPBufferBuilder provides methods for building device-dependent buffer,
/// tensor, chain and kernel ops. An instance of this class is created for
/// each device in the model and accumulates device-specific lowering state.
///
/// All values representing buffers and tensors must be in the 'native' form
/// understood by this device. However, as a convenience, chains may be in
/// either 'native' form, or be host !mef.chain chains.
class MGPBufferBuilder {
public:
  DeviceRefAttr device;

  MGPBufferBuilder(DeviceRefAttr device) : device(device) {}

  MGPBufferBuilder(const MGPBufferBuilder &) = delete;
  MGPBufferBuilder &operator=(const MGPBufferBuilder &) = delete;

  /// Set up the state for emitting ops into a ModelOp. Returns the chain
  /// indicating when the runtime has been established.
  TypedValue<MEF::ChainType> setupForModel(ModelOp modelOp,
                                           MGPBufferBuilder *hostBufferBuilder,
                                           OpBuilder &builder, Location loc,
                                           TypedValue<MEF::ChainType> inChain);

  /// Recover the state from modelOp for this device-specific buffer builder.
  void recoverFromModel(ModelOp modelOp, MGPBufferBuilder *hostBufferBuilder);

  /// Returns the !mgp.context appropriate for op, or null if none can be
  /// found.
  TypedValue<ContextType> getContextForOp(Operation *op);

  DeviceRefAttr getDevice() { return device; }

  /// Returns the constant buffer with given attribute as its contents.
  Value bufferConstant(OpBuilder &builder, Location loc,
                       TypedValue<MEF::ChainType> inChain,
                       TypedValue<ContextType> context, ElementsAttr value);

  /// Returns an undefined, external constant with the given name.
  /// The name is an SSA value of type !mgp.opaque<"mojo_string">.
  Value bufferConstantExternal(OpBuilder &builder, Location loc,
                               TypedValue<MEF::ChainType> inChain,
                               TypedValue<ContextType> context,
                               TypedValue<MGP::OpaqueType> weightsRegistry,
                               TypedValue<MGP::OpaqueType> name, uint64_t size,
                               uint64_t align, bool hasAlias,
                               bool isPlaceholder);

  /// Returns the buffer held by tensor.
  Value tensorExtractBuffer(OpBuilder &builder, Location loc, Value tensor);

  /// Returns the tensor spec of the given tensor.
  TypedValue<TensorSpecType>
  tensorExtractTensorSpec(OpBuilder &builder, Location loc, Value tensor);

  /// Return the tensor constructed from buffer and tensor spec. If necessary
  /// the buffer will be 'unsized' or 'resized' to match the tensorSpec type.
  /// `mut` marks the result type as mutable (the lowered form of
  /// `!mo.buffer`).
  Value tensorCreate(OpBuilder &builder, Location loc,
                     TypedValue<ContextType> context, Value inChain,
                     Value buffer, TypedValue<TensorSpecType> tensorSpec,
                     bool mut = false);

  /// Returns the buffer of size. The alignment may be ignored for some
  /// devices.
  Value bufferAlloc(OpBuilder &builder, Location loc,
                    TypedValue<ContextType> context, Value inChain, Value size,
                    Value deviceContext, uint64_t alignment);

  /// Returns the chain indicating when srcBuffer on the host device has been
  /// transferred to dstBuffer on this builder's device.
  Value bufferTransferFromHost(OpBuilder &builder, Location loc,
                               TypedValue<ContextType> context, Value inChain,
                               MGPBufferBuilder *srcBuilder, Value srcBuffer,
                               Value dstBuffer, Value deviceContext);

  /// Returns the !mef.chain indicating when srcBuffer on this builder's
  /// device has been transferred to dstBuffer on the host.
  Value bufferTransferToHost(OpBuilder &builder, Location loc,
                             TypedValue<ContextType> context, Value inChain,
                             Value srcBuffer, MGPBufferBuilder *dstBuilder,
                             Value dstBuffer, Value deviceContext);

  /// Returns the chain indicating when srcBuffer on this builder's device has
  /// been transferred to dstBuffer (which resides on a different device).
  /// inChain is a device specific chain on the srcBuffer's device.
  Value bufferTransferDevice(OpBuilder &builder, Location loc,
                             TypedValue<ContextType> context, Value inChain,
                             Value srcBuffer,
                             MGPBufferBuilder *dstBufferBuilder,
                             Value dstBuffer, Value srcDeviceContext,
                             Value dstDeviceContext);

  /// Get the runtime size of the buffer
  Value bufferGetSize(OpBuilder &builder, Location loc, Value inChain,
                      Value inBuffer);

  /// Return the chain indicating the debug string has been printed.
  Value debugPrint(OpBuilder &builder, Location loc,
                   TypedValue<ContextType> context, Value inChain,
                   StringAttr debugString, StringAttr label);

  /// Return the chain indicating tensor, in the form of buffer and tensor
  /// spec, has been printed.
  Value debugTensorPrint(OpBuilder &builder, Location loc,
                         TypedValue<ContextType> context, Value inChain,
                         Value buffer, TypedValue<TensorSpecType> spec,
                         StringAttr label);

  /// Returns the chain indicating the given module image with given function
  /// and constant names has been loaded into the runtime.
  TypedValue<MEF::ChainType>
  moduleLoad(OpBuilder &builder, Location loc, TypedValue<ContextType> context,
             TypedValue<MEF::ChainType> inChain, ElementsAttr image,
             ArrayAttr functionSymbols, ArrayAttr constantSymbols);

  /// Returns the join of all the given chain values, which must be all native
  /// to this builder.
  Value chainCreate(OpBuilder &builder, Location loc, ArrayRef<Value> inChains);

  /// Places buffer into a fresh slot of context. Returns the chain indicating
  /// placement and the slot index for later retrieval. Chains are always
  /// !mef.chain, irrespective of this builder's native chain type. Generally
  /// only appropriate within the 'init' block of an mgp.model op.
  std::pair<TypedValue<MEF::ChainType>, uint64_t>
  cache(OpBuilder &builder, Location loc, TypedValue<ContextType> context,
        TypedValue<MEF::ChainType> inChain, ContextCreateOp contextCreateOp,
        Value buffer);

  /// Retrieves a buffer of the expected type from the given slot of the given
  /// !mgp.context. Note that the input chain is always a !mef.chain,
  /// irrespective of the native chain type for this builder.
  Value getCached(OpBuilder &builder, Location loc,
                  TypedValue<ContextType> context,
                  TypedValue<MEF::ChainType> inChain, uint64_t bufferSlot,
                  Type expectedBufferType);

  /// Returns a chain indicating when the given input buffers have been
  /// copied into the output buffer.
  Value bufferConcat(OpBuilder &builder, Location loc,
                     TypedValue<DeviceContextType> deviceContext, Value inChain,
                     Value outBuffer, ArrayRef<Value> inBuffers);

  Value bufferSlice(OpBuilder &builder, Location loc, Value inBuffer,
                    Value offset, Value size, Value deviceContext);

  /// Returns the bool contents of a single-byte buffer.
  Value bufferToBool(OpBuilder &builder, Location loc,
                     TypedValue<ContextType> context, Value inChain,
                     Value inBuffer);

  /// Returns the chain indicating the given index values have been assigned
  /// into the given buffer.
  Value bufferSetWithIndex(OpBuilder &builder, Location loc,
                           TypedValue<ContextType> context, Value buffer,
                           ArrayRef<Value> values);

  /// Adds a device specific sync. Returns a ready chain once the device has
  /// synchronized.
  Value sync(OpBuilder &builder, Location loc, TypedValue<ContextType> context,
             TypedValue<DeviceContextType> deviceContext, Value chain);

  std::pair<uint64_t, uint64_t> getFunction(StringRef name) {
    auto itr = functionToIndexPair.find(name);
    ASSERT_STREAM(itr != functionToIndexPair.end(),
                  << "No function has been registered for name '" << name
                  << "' in the runtime for device '" << device << "'");
    return itr->second;
  }

  std::pair<uint64_t, uint64_t> getConstant(StringRef name) {
    auto itr = constantToIndexPair.find(name);
    ASSERT_STREAM(itr != constantToIndexPair.end(),
                  << "No constant has been registered for name '" << name
                  << "' in the runtime for device '" << device << "'");
    return itr->second;
  }

  uint64_t getRuntimeSlot() const { return runtimeSlot; }

private:
  /// The builder for the host device. May point to this if this is the host
  /// device buffer builder.
  MGPBufferBuilder *hostBufferBuilder = nullptr;
  /// The slot holding the runtime for this device in the !mgp.context.
  uint64_t runtimeSlot = 0;
  /// Maps the name of each compiled kernel to its (module index, function
  /// index) pair in the runtime.
  llvm::StringMap<std::pair<uint64_t, uint64_t>> functionToIndexPair;
  /// Maps the name of each constant to its (module index, constant index)
  /// pair in the runtime.
  llvm::StringMap<std::pair<uint64_t, uint64_t>> constantToIndexPair;
  /// The next available module index for the runtime associated with
  /// this device.
  uint64_t nextModuleIndex = 0;
};

//===----------------------------------------------------------------------===//
// BufferizedTensor
//===----------------------------------------------------------------------===//

/// A representation of tensor as a pair of a buffer and its tensor spec.
/// The tensor spec is always an TensorSpecType, but the buffer type
/// depends on the device being lowered to.
struct BufferizedTensor {
  /// Written by the producing op, then read by all consumers. Its type
  /// (mgp.cpu.buffer, cuda.buffer, etc) depends on the storage device.
  Value buffer;

  /// The mgp.tensor_spec describing the shape and dtype of tensor
  /// at runtime.
  TypedValue<TensorSpecType> tensorSpec;

  BufferizedTensor() = default;

  BufferizedTensor(Value buffer, TypedValue<TensorSpecType> tensorSpec)
      : buffer(buffer), tensorSpec(tensorSpec) {}
};

//===----------------------------------------------------------------------===//
// BufferBuilderContext
//===----------------------------------------------------------------------===//

/// Collects the type and buffer builders for every device in a model.
///
/// Not thread safe. Should only be used per-model and not shared across
/// models.
class BufferBuilderContext {
public:
  /// The distinguished 'host' device.
  DeviceRefAttr hostDevice;

  BufferBuilderContext() = default;

  /// Register a type builder for the given device.
  void registerTypeBuilder(DeviceRefAttr device) {
    [[maybe_unused]] bool inserted =
        typeBuilderMap.insert({device, MGPTypeBuilder(device)}).second;
    ASSERT_STREAM(inserted,
                  << "a type builder has already been registered for device");
  }

  /// Returns the type builder to use for device.
  MGPTypeBuilder getTypeBuilder(DeviceRefAttr device) const {
    auto itr = typeBuilderMap.find(device);
    ASSERT_STREAM(itr != typeBuilderMap.end(),
                  << "no type builder has been registered for device: "
                  << device);
    return itr->second;
  }

  /// Register the given buffer builder for the given device.
  void registerBufferBuilder(DeviceRefAttr device,
                             std::unique_ptr<MGPBufferBuilder> bufferBuilder) {
    [[maybe_unused]] bool inserted =
        bufferBuilderMap.insert({device, std::move(bufferBuilder)}).second;
    ASSERT_STREAM(inserted,
                  << "a buffer builder has already been registered for device");
  }

  /// Returns the buffer builder to use for device. The builder is owned by
  /// this object.
  MGPBufferBuilder &getBufferBuilder(DeviceRefAttr device) const {
    assert(device);
    auto itr = bufferBuilderMap.find(device);
    ASSERT_STREAM(itr != bufferBuilderMap.end(),
                  << "no buffer builder has been registered for device "
                  << device);
    return *itr->second;
  }

  /// Returns the buffer builder to use for the 'host' device.
  MGPBufferBuilder &getHostBufferBuilder() const {
    return getBufferBuilder(hostDevice);
  }

private:
  llvm::DenseMap<DeviceRefAttr, MGPTypeBuilder> typeBuilderMap;

  llvm::DenseMap<DeviceRefAttr, std::unique_ptr<MGPBufferBuilder>>
      bufferBuilderMap;
};

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_BUFFERBUILDER_H
