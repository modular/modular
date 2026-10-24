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
// This file contains runtime bindings for non-DeviceContext components.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "Support/RCRef.h"
#include "Support/SymbolExport.h"
#include "Support/max_license/Tracy/Tracy.h"

using namespace M;

//===----------------------------------------------------------------------===//
// AsyncValue
//===----------------------------------------------------------------------===//

// Reference counting for a bare `AsyncValue *` handle, backing the limited Mojo
// `AnyAsyncValueRef` wrapper. A null handle is the empty/non-tracked reference,
// so both are no-ops on null.
MODULAR_EXPORT void AsyncRT_AsyncValue_retain(AsyncRT::AsyncValue *value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_retain", TRACY_COLOR_GRAY);
  if (value)
    RCRef<AsyncRT::AsyncValue>::lowLevelAddRef(value);
}

MODULAR_EXPORT void AsyncRT_AsyncValue_release(AsyncRT::AsyncValue *value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_release", TRACY_COLOR_GRAY);
  if (value)
    RCRef<AsyncRT::AsyncValue>::lowLevelDropRef(value);
}

// Retains the backing storage handle of a packed buffer/tensor async value and
// hands the owning `AsyncValue*` (refcount +1) to a Mojo `AnyAsyncValueRef`.
// Used by unpack so an unpacked value (or a derivative) keeps its backing
// memory alive when it is re-packed as an output. Non-tracked storage yields
// null.
MODULAR_EXPORT AsyncRT::AsyncValue *
AsyncRT_AsyncValue_retainBufferStorage(AsyncRT::AsyncValue *value) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_retainBufferStorage",
                       TRACY_COLOR_GRAY);
  if (!value)
    return nullptr;
  AsyncRT::AnyAsyncValueRef handle;
  if (value->isType<MLRT::TensorBufferRef>())
    handle = value->get<MLRT::TensorBufferRef>().getMemStorageHandle();
  else if (value->isType<MLRT::Tensor>())
    handle = value->get<MLRT::Tensor>().getBufferRef().getMemStorageHandle();
  return handle.releasePointer();
}

// Moves the AsyncValue out of a runtime-owned slot, resets the slot to empty,
// and hands the owning AsyncValue* to a Mojo AnyAsyncValueRef. This lets the
// MGP-ABI kernel entry point take ownership of its argument slots and rely on
// Mojo destructors instead of an explicit async-delete pass.
MODULAR_EXPORT AsyncRT::AsyncValue *
AsyncRT_AsyncValue_takeHandle(AsyncRT::AnyAsyncValueRef *slot) {
  TRACY_ZONE_SCOPED_NC("AsyncRT_AsyncValue_takeHandle", TRACY_COLOR_GRAY);
  if (!slot)
    return nullptr;
  AsyncRT::AnyAsyncValueRef taken = std::move(*slot);
  return taken.releasePointer();
}

//===----------------------------------------------------------------------===//
// Device affinity
//===----------------------------------------------------------------------===//

// Compute the worker task ID for a GPU device. Called from the MAX accelerator
// library via external_call (max.runtime.asyncrt.task_id_for_device). Obtains
// the thread pool size internally from the current AsyncRT cpuDevice.
MODULAR_EXPORT int32_t MLRT_TaskIdForDevice(int32_t deviceId) {
  auto *rt = AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull();
  size_t numWorkers = rt->getWorkQueue()->getParallelismLevel();
  return MLRT::taskIdForDevice(deviceId, numWorkers);
}
