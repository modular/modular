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

#include "GraphCompiler/MGPDialect/Dialect/BufferBuilder.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// MGPBufferBuilder
//===----------------------------------------------------------------------===//

TypedValue<MEF::ChainType> MGPBufferBuilder::setupForModel(
    ModelOp modelOp, MGPBufferBuilder *hostBufferBuilder, OpBuilder &builder,
    Location loc, TypedValue<MEF::ChainType> inChain) {
  this->hostBufferBuilder = hostBufferBuilder;

  TypedValue<ContextType> context;
  modelOp.walk([&](ContextCreateOp contextCreateOp) {
    runtimeSlot = contextCreateOp.getNumSlots();
    contextCreateOp.setNumSlots(runtimeSlot + 1);
    context = contextCreateOp.getResult();
    return WalkResult::interrupt();
  });

  if (this->hostBufferBuilder == this) {
    auto runtime = MGP::RuntimeCreateOp::create(builder, loc,
                                                /*inChain=*/inChain,
                                                /*theContext=*/context,
                                                /*aRuntimeSlot=*/runtimeSlot);
    ;
    return runtime.getOutChain();
  }
  return inChain;
}

void MGPBufferBuilder::recoverFromModel(ModelOp modelOp,
                                        MGPBufferBuilder *hostBufferBuilder) {
  this->hostBufferBuilder = hostBufferBuilder;

  // Without an mgp.runtime.create (unit tests), the slot index stays 0.
  modelOp.walk([&](RuntimeCreateOp runtimeCreateOp) {
    runtimeSlot = runtimeCreateOp.getARuntimeSlot();
    return WalkResult::interrupt();
  });

  size_t moduleIndex = 0;
  modelOp.walk([&](ModuleLoadOp moduleLoadOp) {
    for (auto [functionIndex, name] :
         llvm::enumerate(moduleLoadOp.getCFunctionSymbols())) {
      functionToIndexPair[name] = std::make_pair(moduleIndex, functionIndex);
    }
    for (auto [constantIndex, name] :
         llvm::enumerate(moduleLoadOp.getDConstantSymbols())) {
      constantToIndexPair[name] = std::make_pair(moduleIndex, constantIndex);
    }
    ++moduleIndex;
  });
}

TypedValue<ContextType> MGPBufferBuilder::getContextForOp(Operation *op) {
  if (auto hasSlotOp = dyn_cast<MGPHasRuntimeSlotOpInterface>(op))
    return hasSlotOp.getTheContext();
  return getContextForBlock(op->getBlock());
}

Value MGPBufferBuilder::bufferConstant(OpBuilder &builder, Location loc,
                                       TypedValue<MEF::ChainType> inChain,
                                       TypedValue<ContextType> context,
                                       ElementsAttr value) {
  return BufferConstantOp::create(builder, loc, /*aValue=*/value);
}

Value MGPBufferBuilder::bufferConstantExternal(
    OpBuilder &builder, Location loc, TypedValue<MEF::ChainType> inChain,
    TypedValue<ContextType> context,
    TypedValue<MGP::OpaqueType> weightsRegistry,
    TypedValue<MGP::OpaqueType> name, uint64_t size, uint64_t align,
    bool hasAlias, bool isPlaceholder) {
  return BufferConstantExternalOp::create(builder, loc,
                                          /*inChain=*/inChain,
                                          /*weightsRegistry=*/weightsRegistry,
                                          /*name=*/name,
                                          /*aSize=*/size,
                                          /*bAlign=*/align,
                                          /*cDevice=*/device,
                                          /*dHasAlias=*/hasAlias,
                                          /*eIsPlaceholder=*/isPlaceholder);
}

Value MGPBufferBuilder::tensorExtractBuffer(OpBuilder &builder, Location loc,
                                            Value tensor) {
  return builder.createOrFold<TensorExtractBufferOp>(
      loc,
      /*tensor=*/cast<TypedValue<DeviceTensorType>>(tensor));
}

TypedValue<TensorSpecType>
MGPBufferBuilder::tensorExtractTensorSpec(OpBuilder &builder, Location loc,
                                          Value tensor) {
  return dyn_cast<TypedValue<TensorSpecType>>(
      builder.createOrFold<TensorExtractTensorSpecOp>(
          loc,
          /*tensor=*/cast<TypedValue<DeviceTensorType>>(tensor)));
}

Value MGPBufferBuilder::tensorCreate(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inChain, Value buffer,
                                     TypedValue<TensorSpecType> tensorSpec,
                                     bool mut) {
  auto deviceBuffer = cast<TypedValue<DeviceBufferType>>(buffer);
  return TensorCreateOp::create(builder, loc,
                                /*inChain=*/inChain,
                                /*buffer=*/deviceBuffer,
                                /*tensorSpec=*/tensorSpec,
                                /*mut=*/mut);
}

Value MGPBufferBuilder::bufferAlloc(OpBuilder &builder, Location loc,
                                    TypedValue<ContextType> context,
                                    Value inChain, Value size,
                                    Value deviceContext, uint64_t alignment) {
  return BufferAllocOp::create(builder, loc, DeviceBufferType::get(device),
                               /*inChain=*/inChain,
                               /*runtimeSize=*/size,
                               /*deviceContext=*/deviceContext,
                               /*bAlign=*/alignment,
                               /*cDevice=*/device);
}

Value MGPBufferBuilder::bufferTransferFromHost(OpBuilder &builder, Location loc,
                                               TypedValue<ContextType> context,
                                               Value inChain,
                                               MGPBufferBuilder *srcBuilder,
                                               Value srcBuffer, Value dstBuffer,
                                               Value deviceContext) {
  Value copyChain =
      BufferHostToDeviceOp::create(
          builder, loc,
          /*inChain=*/inChain,
          /*hostBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
          /*otherBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
          /*deviceContext*/
          cast<TypedValue<DeviceContextType>>(deviceContext),
          /*cHostDevice=*/hostBufferBuilder->device,
          /*dOtherDevice=*/device)
          .getResult();

  Value syncChain =
      MGP::SyncOp::create(builder, loc, copyChain, context, deviceContext,
                          /*bDevice=*/device);

  MGP::ExtendLifetimeOp::create(builder, loc, syncChain, ValueRange{srcBuffer});
  // Return copyChain, not syncChain: ops on the same deviceContext are
  // stream-ordered after the h2d copy, and waiting on syncChain would block
  // enqueueing ahead. A d2d copy using this dst as its src runs on another
  // stream, so it does not get that ordering for free.
  return copyChain;
}

Value MGPBufferBuilder::bufferTransferToHost(OpBuilder &builder, Location loc,
                                             TypedValue<ContextType> context,
                                             Value inChain, Value srcBuffer,
                                             MGPBufferBuilder *dstBuilder,
                                             Value dstBuffer,
                                             Value deviceContext) {
  Value copyChain =
      BufferDeviceToHostOp::create(
          builder, loc,
          /*inChain=*/inChain,
          /*otherBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
          /*hostBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
          /*deviceContext*/
          cast<TypedValue<DeviceContextType>>(deviceContext),
          /*cOtherDevice=*/device,
          /*dHostDevice=*/hostBufferBuilder->device)
          .getResult();
  Value syncChain =
      MGP::SyncOp::create(builder, loc, copyChain, context, deviceContext,
                          /*bDevice*/ device);

  MGP::ExtendLifetimeOp::create(builder, loc, syncChain, ValueRange{dstBuffer});

  // The host cannot read dstBuffer until the d2h copy completes.
  return syncChain;
}

Value MGPBufferBuilder::bufferTransferDevice(
    OpBuilder &builder, Location loc, TypedValue<ContextType> context,
    Value inChain, Value srcBuffer, MGPBufferBuilder *dstBufferBuilder,
    Value dstBuffer, Value srcDeviceContext, Value dstDeviceContext) {
  DeviceRefAttr srcDevice = device;
  DeviceRefAttr dstDevice = dstBufferBuilder->device;

  // Different (virtual) devices use different streams, so the copy (on the
  // destination stream) is synchronized with the source stream via
  // mgp.device_wait on both sides.
  bool needsCrossStreamSync = srcDevice != dstDevice;

  Value copyInChain = inChain;
  if (needsCrossStreamSync) {
    // Source data must be ready before the copy reads it.
    copyInChain =
        MGP::DeviceWaitOp::create(builder, loc, /*inChain=*/inChain, context,
                                  /*waitingDeviceContext=*/dstDeviceContext,
                                  /*signalingDeviceContext=*/srcDeviceContext,
                                  /*bDevice=*/dstDevice)
            .getResult();
  }

  Value copyChain =
      BufferDeviceToDeviceOp::create(
          builder, loc,
          /*inChain=*/copyInChain,
          /*srcBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
          /*dstBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
          /*deviceContext*/
          cast<TypedValue<DeviceContextType>>(srcDeviceContext),
          /*deviceContext*/
          cast<TypedValue<DeviceContextType>>(dstDeviceContext),
          /*cSrcDevice=*/srcDevice,
          /*dDstDevice=*/dstDevice)
          .getResult();

  if (!needsCrossStreamSync)
    return copyChain;

  // The source must not be overwritten or freed until the copy completes.
  return MGP::DeviceWaitOp::create(builder, loc, /*inChain=*/copyChain, context,
                                   /*waitingDeviceContext=*/srcDeviceContext,
                                   /*signalingDeviceContext=*/dstDeviceContext,
                                   /*bDevice=*/srcDevice)
      .getResult();
}

Value MGPBufferBuilder::bufferGetSize(OpBuilder &builder, Location loc,
                                      Value inChain, Value inBuffer) {
  return builder.createOrFold<BufferGetSizeOp>(
      loc,
      /*inChain=*/cast<TypedValue<MEF::ChainType>>(inChain),
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer));
}

Value MGPBufferBuilder::debugPrint(OpBuilder &builder, Location loc,
                                   TypedValue<ContextType> context,
                                   Value inChain, StringAttr debugString,
                                   StringAttr label) {
  return DebugPrintOp::create(builder, loc,
                              /*inChain=*/inChain,
                              /*theContext=*/context,
                              /*aDebugString=*/debugString,
                              /*bLabel=*/label);
}

Value MGPBufferBuilder::debugTensorPrint(OpBuilder &builder, Location loc,
                                         TypedValue<ContextType> context,
                                         Value inChain, Value buffer,
                                         TypedValue<TensorSpecType> spec,
                                         StringAttr label) {
  return DebugTensorPrintOp::create(
      builder, loc,
      /*inChain=*/inChain,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*spec=*/spec,
      /*bLabel=*/label);
}

/// Returns arrayAttr, which should be an array of StringAttrs, encoded as
/// a MultiLineStringAttr. Also adds each string to the given map.
static MultiLineStringAttr
toMultiline(OpBuilder &builder, uint64_t moduleIndex, ArrayAttr arrayAttr,
            llvm::StringMap<std::pair<uint64_t, uint64_t>> &nameToIndexPair) {
  SmallVector<StringAttr> stringAttrs;
  if (arrayAttr) {
    uint64_t nameIndex = 0;
    for (StringRef symbol : arrayAttr.getAsValueRange<mlir::StringAttr>()) {
      stringAttrs.emplace_back(builder.getStringAttr(symbol));
      nameToIndexPair.insert({symbol, std::make_pair(moduleIndex, nameIndex)});
      ++nameIndex;
    }
  }
  return MultiLineStringAttr::get(builder.getContext(), stringAttrs);
}

TypedValue<MEF::ChainType> MGPBufferBuilder::moduleLoad(
    OpBuilder &builder, Location loc, TypedValue<ContextType> context,
    TypedValue<MEF::ChainType> inChain, ElementsAttr image,
    ArrayAttr functionSymbols, ArrayAttr constantSymbols) {
  auto functionSymbolsAttr = toMultiline(builder, nextModuleIndex,
                                         functionSymbols, functionToIndexPair);
  auto constantSymbolsAttr = toMultiline(builder, nextModuleIndex,
                                         constantSymbols, constantToIndexPair);
  auto load = ModuleLoadOp::create(builder, loc,
                                   /*inChain=*/
                                   cast<TypedValue<MEF::ChainType>>(inChain),
                                   /*theContext=*/context,
                                   /*aRuntimeSlot=*/runtimeSlot,
                                   /*bImage=*/image,
                                   /*cFunctionSymbols=*/functionSymbolsAttr,
                                   /*dConstantSymbols=*/constantSymbolsAttr,
                                   /*eDevice=*/device,
                                   /*fModuleIndex=*/nextModuleIndex);
  ++nextModuleIndex;
  return load;
}

Value MGPBufferBuilder::chainCreate(OpBuilder &builder, Location loc,
                                    ArrayRef<Value> inChains) {
  return GRT::ChainCreateOp::create(builder, loc, inChains);
}

std::pair<TypedValue<MEF::ChainType>, uint64_t>
MGPBufferBuilder::cache(OpBuilder &initBuilder, Location loc,
                        TypedValue<ContextType> context,
                        TypedValue<MEF::ChainType> inChain,
                        ContextCreateOp contextCreateOp, Value buffer) {
  uint64_t bufferSlot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(bufferSlot + 1);
  auto cacheOp = BufferCacheOp::create(
      initBuilder, loc, /*inChain=*/inChain, /*theContext=*/contextCreateOp,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*bBufferSlot=*/bufferSlot);
  return {cacheOp.getOutChain(), bufferSlot};
}

Value MGPBufferBuilder::getCached(OpBuilder &builder, Location loc,
                                  TypedValue<ContextType> context,
                                  TypedValue<MEF::ChainType> inChain,
                                  uint64_t bufferSlot,
                                  Type expectedBufferType) {
  auto parentOp = builder.getBlock()->getParentOp();
  if (!isa<ModelOp>(parentOp))
    parentOp = parentOp->getParentOfType<MGP::ModelOp>();

  auto modelOp = cast<ModelOp>(parentOp);
  Value bufferSlotIndex =
      mlir::index::ConstantOp::create(builder, loc, bufferSlot);
  if (modelOp.getIsSubgraph()) {
    bufferSlotIndex = mlir::index::AddOp::create(builder, loc, bufferSlotIndex,
                                                 modelOp.getSubgraphIndex());
  }

  return BufferGetCachedOp::create(builder, loc, expectedBufferType,
                                   /*inChain=*/inChain, /*theContext=*/context,
                                   /*bBufferSlot=*/bufferSlotIndex)
      .getOutBuffer();
}

Value MGPBufferBuilder::bufferConcat(
    OpBuilder &builder, Location loc,
    TypedValue<DeviceContextType> deviceContext, Value inChain, Value outBuffer,
    ArrayRef<Value> inBuffers) {
  return BufferConcatOp::create(
      builder, loc, /*inChain=*/inChain,
      /*outBuffer=*/cast<TypedValue<DeviceBufferType>>(outBuffer),
      /*inBuffers=*/inBuffers, deviceContext,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::bufferSlice(OpBuilder &builder, Location loc,
                                    Value inBuffer, Value offset, Value size,
                                    Value deviceContext) {
  return BufferSliceOp::create(
      builder, loc,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*offset=*/offset,
      /*size=*/size,
      /*deviceContext=*/cast<TypedValue<DeviceContextType>>(deviceContext));
}

Value MGPBufferBuilder::bufferToBool(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inChain, Value inBuffer) {
  return BufferToBoolOp::create(
      builder, loc, /*inChain=*/inChain,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*bDevice=*/device);
}

Value MGPBufferBuilder::bufferSetWithIndex(OpBuilder &builder, Location loc,
                                           TypedValue<ContextType> context,
                                           Value buffer,
                                           ArrayRef<Value> values) {
  return BufferSetWithIndexOp::create(
      builder, loc,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*values=*/values,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::sync(OpBuilder &builder, Location loc,
                             TypedValue<ContextType> context,
                             TypedValue<DeviceContextType> deviceContext,
                             Value chain) {
  return MGP::SyncOp::create(builder, loc, chain, context, deviceContext,
                             /*bDevice*/ device)
      .getResult();
}
