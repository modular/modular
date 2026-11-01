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
// Shared device-to-worker affinity logic used by both the graph compiler
// (MGPPrimitives) and the Mojo cpuDevice (via CompilerRT).
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICEAFFINITY_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICEAFFINITY_H

#include "AsyncRT/Runtime/Algorithms.h"

#include <cstddef>

namespace M::MLRT {

/// Compute the worker task ID for a given GPU device.
///
/// Resolution order:
///   1. Explicit mapping via MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS config/env.
///   2. NUMA-inferred GPU-to-CPU core mapping (Linux x86 only, and only when
///      the thread pool size matches the physical core count).
///   3. Round-robin fallback across workers 1..N-1 (skipping worker 0).
///
/// Returns kDefaultTaskId (-1) when numWorkers <= 1.
int taskIdForDevice(int deviceId, size_t numWorkers);

/// Compute the worker task ID which explicitly avoids assigning to the
/// per-device-affinitized threads. This is for offloading tasks that should
/// not block per-device threads, which could accidentally pick up task when
/// given a generic assignment of kDefaultTaskId.
int taskIdForNonDevice(int taskId, size_t numWorkers);

/// Like parallelForEachNChainWithTaskIds, but automatically pins each element
/// to the worker thread with affinity for device `elementIdx`, using
/// taskIdForDevice to compute the thread assignment from the cpuDevice's work
/// queue parallelism level.
template <typename... CaptureTys, typename ElementFn>
static inline AsyncRT::AsyncValueRef<AsyncRT::Chain>
launchDeviceCollective(AsyncRT::CPUDevice &cpuDevice, size_t totalCount,
                       ElementFn &&elementFn, CaptureTys &&...captures) {
  AsyncRT::WorkQueue *workQueue = cpuDevice.getWorkQueue();
  auto taskIdFn = [workQueue](size_t deviceIdx) {
    return taskIdForDevice(static_cast<int>(deviceIdx),
                           workQueue->getParallelismLevel());
  };
  return AsyncRT::parallelForEachNChainWithTaskIds(
      cpuDevice, totalCount, taskIdFn, std::forward<ElementFn>(elementFn),
      std::forward<CaptureTys>(captures)...);
}

/// Like parallelForEachNChainWithTaskIds, but routes each element to a worker
/// thread that is NOT a per-device-affinitized thread. This avoids contention
/// with GPU command dispatch workers while still allowing parallel execution
/// across the remaining pool of worker threads.
template <typename... CaptureTys, typename ElementFn>
static inline AsyncRT::AsyncValueRef<AsyncRT::Chain>
launchNonDeviceCollective(AsyncRT::CPUDevice &cpuDevice, size_t totalCount,
                          ElementFn &&elementFn, size_t start = 0) {
  AsyncRT::WorkQueue *workQueue = cpuDevice.getWorkQueue();
  size_t numWorkers = workQueue->getParallelismLevel();

  auto taskIdFn = [numWorkers](size_t i) -> int {
    return taskIdForNonDevice(static_cast<int>(i), numWorkers);
  };

  return AsyncRT::parallelForEachNChainWithTaskIds(
      cpuDevice, totalCount, std::move(taskIdFn),
      std::forward<ElementFn>(elementFn));
}

} // namespace M::MLRT

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICEAFFINITY_H
