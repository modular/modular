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

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEVICEEXECUTION_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEVICEEXECUTION_H

#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Context.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"

namespace M::GraphCompiler::Driver {

/// Allocates a device buffer for `spec` against `device`, applying
/// `MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison` poisoning.
MODULAR_VISIBILITY_EXPORT
ErrorOr<M::Driver::DeviceBufferRef>
allocateDeviceBuffer(const M::Driver::DeviceContextRef &device,
                     const TensorSpec &spec, bool hostAlloc = false);

/// Wraps `byteSize` bytes at `ptr` in a non-owning buffer; the caller must keep
/// the memory alive as long as the buffer. Prefer this over
/// `createBufferOwning(..., /*owning=*/true)` with a no-op `setDestructorFn`,
/// which reads as the opposite of what it means.
MODULAR_VISIBILITY_EXPORT M::Driver::DeviceBufferRef
borrowBuffer(const M::Driver::DeviceContextRef &device, void *ptr,
             size_t byteSize);

// Construct a device buffer holding `tensor`, allocated against the MLRT
// `device` context.
MODULAR_VISIBILITY_EXPORT
ErrorOr<M::Driver::DeviceBufferRef>
constructDeviceMemoryFromTensor(M::ContextRef context,
                                M::Driver::DeviceContextRef device,
                                M::MLRT::Tensor tensor);

/// Transforms a DeviceTensor representation into an AnyAsyncValueRef executable
/// returned by an ExecutableModel.
MODULAR_VISIBILITY_EXPORT AnyAsyncValueRef
prepareInput(CPUDevice &cpuDevice, M::Driver::DeviceBufferRef inputBuffer,
             TensorSpec inputSpec);

} // namespace M::GraphCompiler::Driver

#endif
