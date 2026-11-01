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

#include "max/c/tensor.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "GraphCompiler/Support/ModelTensorSpec.h"
#include "Support/DeviceSpecs.h"
#include "Support/ML/TensorSpec.h"
#include "max/c/common.h"
#include "max/c/device.h"
#include "max/max_license/API/c/EngineContext.h"
#include "max/max_license/API/c/detail/helpers.h"
#include "max/max_license/API/c/detail/types.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/Twine.h"

using namespace M;
using M::Detail::mapDType;

MODULAR_API_EXPORT M_TensorSpec *
M_newTensorSpec(const int64_t *shape, int64_t rankSize, M_Dtype dtype,
                const char *tensorName, const M_Device *device) {
  if (!shape || !tensorName || !device)
    return nullptr;
  TensorSpec spec(llvm::ArrayRef(shape, rankSize), mapDType(dtype));

  return new M_TensorSpec{ModelTensorSpec(tensorName, spec), device->spec->ref};
}

MODULAR_API_EXPORT M_DeviceType
M_getDeviceTypeFromSpec(const M_TensorSpec *spec) {
  // Any non-CPU label is treated as an accelerator. EngineContext::create
  // guarantees only known labels (cpu/gpu) reach this far.
  return spec->device.label == kCPULabel ? M_HOST : M_ACCELERATOR;
}

MODULAR_API_EXPORT int M_getDeviceIdFromSpec(const M_TensorSpec *spec) {
  return spec->device.id;
}

MODULAR_API_EXPORT M_Device *M_getTensorDevice(const M_AsyncTensor *tensor) {
  if (!tensor || !tensor->storage)
    return nullptr;
  const M_TensorStorage &storage = *tensor->storage;
  return new M_Device{storage.device, M::Driver::DeviceContextRef::copy(
                                          &storage.buffer->context())};
}

MODULAR_API_EXPORT M_AsyncTensor *M_copyTensorToDevice(M_AsyncTensor *tensor,
                                                       M_Device *device,
                                                       M_Status *status) {
  if (!tensor || !tensor->storage) {
    M::Detail::setError(status, "Invalid tensor.");
    return nullptr;
  }

  if (!device) {
    M::Detail::setError(status, "Invalid device.");
    return nullptr;
  }

  const TensorSpec &spec = tensor->storage->spec;
  ErrorOr<M::Driver::DeviceBufferRef> dstBufferOr =
      M::GraphCompiler::Driver::allocateDeviceBuffer(device->context, spec);
  if (dstBufferOr.isError()) {
    M::Detail::setError(status, dstBufferOr.takeError());
    return nullptr;
  }
  M::Driver::DeviceBufferRef dstBuffer = dstBufferOr.takeValue();

  if (ErrorOrSuccess copyResult = tensor->storage->buffer->copyTo(*dstBuffer)) {
    M::Detail::setError(status, copyResult.takeError());
    return nullptr;
  }

  M::Detail::setOK(status);
  return new M_AsyncTensor{std::make_shared<M_TensorStorage>(
      M_TensorStorage{std::move(dstBuffer), spec, device->spec})};
}

MODULAR_API_EXPORT int M_isDynamicRanked(const M_TensorSpec *spec) {
  if (!spec)
    return 0;

  const auto &underlyingSpec = spec->spec.getSpec();
  return !underlyingSpec.hasRank();
}

MODULAR_API_EXPORT int64_t M_getDimAt(const M_TensorSpec *spec, size_t index) {
  if (!spec)
    return 0;

  const auto &underlyingSpec = spec->spec.getSpec();

  if (!underlyingSpec.hasRank() || index >= underlyingSpec.getRank())
    return 0;

  int64_t dim = underlyingSpec[index];
  // Convert dimension from kDynamic which is a large negative number to a value
  // which is more commonly used to represent dynamic dimensions by other
  // libraries, ie -1. Though we have abstracted this with a function for future
  // flexibility.
  if (mlir::ShapedType::isDynamic(dim))
    return kDynamicDimensionValue;

  return underlyingSpec[index];
}

MODULAR_API_EXPORT int64_t M_getRank(const M_TensorSpec *spec) {
  if (!spec)
    return 0;

  const auto &underlyingSpec = spec->spec.getSpec();

  if (!underlyingSpec.hasRank())
    return kDynamicRankValue;

  return underlyingSpec.getRank();
}

MODULAR_API_EXPORT M_Dtype M_getDtype(const M_TensorSpec *spec) {
  if (!spec)
    return M_UNKNOWN;

  return mapDType(spec->spec.getSpec().getEltType());
}

MODULAR_API_EXPORT const char *M_getName(M_TensorSpec *spec) {
  if (!spec)
    return nullptr;

  // This is guaranteed to be null terminated because the `M_TensorSpec` name
  // is stored as a `std::string`.
  return spec->spec.getName().data();
}

MODULAR_API_EXPORT M_AsyncTensorMap *
M_newAsyncTensorMap(const M_RuntimeContext *context) {
  return new M_AsyncTensorMap{{}, {}, &context->context};
}

MODULAR_API_EXPORT void M_borrowTensorInto(M_AsyncTensorMap *tensors,
                                           void *input,
                                           const M_TensorSpec *tensorSpec,
                                           M_Status *status) {
  if (!tensors) {
    M::Detail::setError(status, "Invalid tensor array.");
    return;
  }

  if (!input) {
    M::Detail::setError(status, "Invalid input tensor.");
    return;
  }

  if (!tensorSpec) {
    M::Detail::setError(status, "Invalid tensor spec.");
    return;
  }

  const TensorSpec &spec = tensorSpec->spec.getSpec();
  auto name = tensorSpec->spec.getName();
  if (tensors->indices.contains(name)) {
    M::Detail::setError(status, "Tensor already present with name");
    return;
  }

  auto deviceOr = tensors->context->resolveDevice(tensorSpec->device);
  if (deviceOr.isError()) {
    M::Detail::setError(status, llvm::Twine("Could not resolve device ") +
                                    tensorSpec->device.toString());
    return;
  }
  const M::Driver::DeviceContextRef &device = deviceOr->context;

  M::Driver::DeviceBufferRef buffer;
  if (device->device().api() == M::Driver::Device::cpuAPI ||
      device->ownsDeviceAddress(input)) {
    // Borrow the input data in place with a no-op destructor (ownership stays
    // with the caller). This covers host inputs and device inputs whose
    // pointer already lives on the target device. Borrowing the device
    // pointer (rather than copying) is what makes in-place mutation of a
    // borrowed buffer visible to the caller; see DRIV-220.
    buffer = M::GraphCompiler::Driver::borrowBuffer(device, input,
                                                    spec.getSizeInBytes());
  } else {
    // The source is host memory (or its residency could not be determined):
    // allocate memory on the device and stage the data in via a copy.
    auto hostOr = tensors->context->resolveDevice(M::DeviceRef(M::kCPULabel));
    if (hostOr.isError()) {
      M::Detail::setError(status, "Could not create host device");
      return;
    }
    auto bufferOr =
        M::GraphCompiler::Driver::allocateDeviceBuffer(device, spec);
    if (bufferOr.isError()) {
      M::Detail::setError(status, "Could not allocate memory on device");
      return;
    }
    buffer = bufferOr.takeValue();

    M::Driver::DeviceBufferRef hostBuffer =
        M::GraphCompiler::Driver::borrowBuffer(hostOr->context, input,
                                               spec.getSizeInBytes());
    if (ErrorOrSuccess copyResult = hostBuffer->copyTo(*buffer)) {
      M::Detail::setError(status, copyResult.takeError());
      return;
    }
  }

  tensors->indices[name] = tensors->tensors.size();
  tensors->tensors.push_back(std::make_shared<M_TensorStorage>(
      M_TensorStorage{std::move(buffer), spec, deviceOr->spec}));
  M::Detail::setOK(status);
}

MODULAR_API_EXPORT M_AsyncTensor *
M_getTensorByNameFrom(M_AsyncTensorMap *tensorMap, const char *tensorName,
                      M_Status *status) {
  if (!tensorMap) {
    M::Detail::setError(status, "Invalid 'M_AsyncTensorMap'*.");
    return nullptr;
  }

  if (!tensorName) {
    M::Detail::setError(status, "Invalid tensor name.");
    return nullptr;
  }

  auto entry = tensorMap->indices.find(tensorName);
  if (entry == tensorMap->indices.end()) {
    M::Detail::setError(status, llvm::Twine("No tensor with name ") +
                                    tensorName + " in tensor map.");
    return nullptr;
  }
  auto storage = tensorMap->tensors[entry->second];
  return new M_AsyncTensor{storage};
}

MODULAR_API_EXPORT size_t M_getTensorNumElements(const M_AsyncTensor *t) {
  // 0 is also a legitimate element count, but it is the only sentinel a
  // size_t return affords, and it matches how the accessors either side of
  // this one report an invalid tensor.
  if (!t || !t->storage)
    return 0;
  return t->storage->spec.getNumElements();
}

MODULAR_API_EXPORT M_Dtype M_getTensorType(const M_AsyncTensor *t) {
  if (!t || !t->storage)
    return M_UNKNOWN;
  return mapDType(t->storage->spec.getEltType());
}

MODULAR_API_EXPORT const void *M_getTensorData(const M_AsyncTensor *tensor) {
  if (!tensor || !tensor->storage)
    return nullptr;
  return tensor->storage->buffer->ptr();
}

MODULAR_API_EXPORT M_TensorSpec *M_getTensorSpec(const M_AsyncTensor *tensor) {
  if (!tensor || !tensor->storage)
    return nullptr;

  return new M_TensorSpec{
      ModelTensorSpec(/*tensorName=*/"", tensor->storage->spec),
      tensor->storage->device->ref};
}

MODULAR_API_EXPORT void M_freeTensor(M_AsyncTensor *tensor) { delete tensor; }

MODULAR_API_EXPORT void M_freeTensorNameArray(M_TensorNameArray *names) {
  delete names;
}

MODULAR_API_EXPORT void M_freeTensorSpec(M_TensorSpec *spec) { delete spec; }

MODULAR_API_EXPORT void M_freeAsyncTensorMap(M_AsyncTensorMap *tensorMap) {
  delete tensorMap;
}
