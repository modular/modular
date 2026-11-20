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

#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "GraphCompiler/Driver/Device/DebugAllocPoison.h"
#include <cstring>

using M::AnyAsyncValueRef;
using M::ErrorOr;
using M::Driver::DeviceBufferRef;
using M::Driver::DeviceContextRef;
using M::MLRT::Tensor;
using M::MLRT::TensorBufferRef;

ErrorOr<DeviceBufferRef> M::GraphCompiler::Driver::allocateDeviceBuffer(
    const DeviceContextRef &device, const M::TensorSpec &spec, bool hostAlloc) {
  ErrorOr<DeviceBufferRef> bufOrErr =
      hostAlloc ? device->createHostBuffer(spec.getSizeInBytes())
                : device->enqueueCreateBuffer(spec.getSizeInBytes());
  if (bufOrErr.isError())
    return bufOrErr.takeError();
  DeviceBufferRef buf = bufOrErr.takeValue();

  // MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison. Host and device share
  // the per-type pattern (see DebugAllocPoison.h) so the Mojo stdlib
  // uninit-read check has one value to look for.
  if (device->isDebugAllocatorEnabled("uninitialized-poison")) {
    M::GraphCompiler::Driver::AllocationPoisonDecision decision =
        M::GraphCompiler::Driver::decidePoison(spec, hostAlloc);
    if (const auto &fill = decision.fill) {
      if (hostAlloc)
        M::GraphCompiler::Driver::fillHostMemory(buf->ptr(),
                                                 spec.getSizeInBytes(), *fill);
      else if (M::ErrorOrSuccess err = device->enqueueSetMemory(
                   *buf, fill->bits, fill->patternBytes))
        return err.takeError();
    }
    // Synchronize even unfilled allocations so `sync-only` matches the
    // filling modes' synchronization.
    if (!hostAlloc) {
      if (M::ErrorOrSuccess err = device->synchronize())
        return err.takeError();
    }
  }

  return buf;
}

M::Driver::DeviceBufferRef
M::GraphCompiler::Driver::borrowBuffer(const DeviceContextRef &device,
                                       void *ptr, size_t byteSize) {
  return device->createBufferOwning(ptr, byteSize, /*owning=*/false);
}

// Construct an on-device buffer to hold a provided tensor.
ErrorOr<DeviceBufferRef>
M::GraphCompiler::Driver::constructDeviceMemoryFromTensor(
    M::ContextRef context, DeviceContextRef device, Tensor tensor) {

  TensorBufferRef bufferRef = tensor.releaseBufferRef();
  AnyAsyncValueRef storageRef = bufferRef.getMemStorageHandle();

  // The graph may return a value that won't outlive it; copy it.
  if (!storageRef) {
    // Should be impossible on a GPU.
    if (device->device().api() != M::Driver::Device::cpuAPI)
      return Error("Execution returned a null result on a GPU.");

    ErrorOr<DeviceBufferRef> bufferOr =
        allocateDeviceBuffer(device, tensor.getSpec());
    if (bufferOr.isError())
      return bufferOr.takeError();
    DeviceBufferRef buffer = bufferOr.takeValue();

    void *tensorBuffer = bufferRef.getBuffer();
    std::memcpy(buffer->ptr(), tensorBuffer, tensor.getSpec().getSizeInBytes());

    return buffer;
  }

  AsyncRT::await(storageRef);

  auto destructor = [context = std::move(context),
                     storageRef = storageRef.copy()]() mutable {
    storageRef.reset();

    // The graph frees CPU memory through the AsyncRT Runtime, so cpuDevice must
    // outlive the output (automatic under `mojo run`, not `mojo build`).
    // TODO: The graph, not the driver, should own this.
    context.reset();
  };

  void *outputData = bufferRef.getBuffer();

  DeviceBufferRef buffer = device->createBufferOwning(
      outputData, tensor.getSpec().getSizeInBytes(), /*owning=*/true);
  buffer->setDestructorFn(std::move(destructor));
  return buffer;
}

AnyAsyncValueRef M::GraphCompiler::Driver::prepareInput(
    CPUDevice &cpuDevice, DeviceBufferRef inputBuffer, TensorSpec inputSpec) {
  // Using the refcounted buffer as the storage handle keeps it alive exactly
  // as long as the AsyncValueRef needs it.
  void *ptr = inputBuffer->ptr();
  size_t byteCount = inputBuffer->byteSize();
  AnyAsyncValueRef storageRef = AnyAsyncValueRef::createReady<DeviceBufferRef>(
      cpuDevice, std::move(inputBuffer));
  // FIXME: Get alignment of device memory.
  Tensor tensor(TensorBufferRef::create(ptr, byteCount, std::move(storageRef),
                                        llvm::MaybeAlign(std::nullopt)),
                std::move(inputSpec));

  return AnyAsyncValueRef::createReady<Tensor>(cpuDevice, std::move(tensor));
}
