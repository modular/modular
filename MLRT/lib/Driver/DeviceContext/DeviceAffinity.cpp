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

#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Configuration.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <map>

using namespace llvm;

namespace M::MLRT {

/// Infer a mapping from each GPU device index to a NUMA-local CPU core.
/// Returns an empty map on any failure (unsupported platform, missing
/// topology, device enumeration errors, etc.).
static std::map<int, int> inferGpuToCpuCoreMapping() {
#if MODULAR_HAL_DEVICE_CONTEXT
  // Under --config=hal-device-context the legacy C++ DeviceContext is not
  // linked into Mojo programs, so device enumeration is unavailable here.
  // Returning an empty mapping makes taskIdForDevice fall back to round-robin
  // placement (tier 3), the same behavior as any other inference failure.
  return {};
#else
  const auto &numaTopologyOrError = NUMATopology::get();
  if (numaTopologyOrError.isError())
    return {};
  const NUMATopology &numaTopology = *numaTopologyOrError;

  int numDevices =
      M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);

  std::map<int, int> gpuToCpuCoreMapping;
  std::map<int, size_t> nextCpuCoreIndexPerNuma;

  for (int i = 0; i < numDevices; i++) {
    auto deviceOrError =
        M::Driver::Device::create(M::Driver::Device::gpuAPI, i);
    if (deviceOrError.isError())
      break;
    auto &device = *deviceOrError;

    auto pciBusIdOrError = device->getPciBusId();
    if (pciBusIdOrError.isError())
      break;

    int numaNode = numaTopology.getNumaNodeForPciBus(*pciBusIdOrError);
    if (numaNode == -1)
      break;

    std::vector<size_t> cpuIds = numaTopology.getCpuIdsForNumaNode(numaNode);
    if (cpuIds.empty())
      break;

    size_t cpuCoreIndex =
        (1 + nextCpuCoreIndexPerNuma[numaNode]++ % (cpuIds.size() - 1));

    gpuToCpuCoreMapping[i] = cpuIds[cpuCoreIndex];
  }
  return gpuToCpuCoreMapping;
#endif // MODULAR_HAL_DEVICE_CONTEXT
}

/// Cached wrapper around inferGpuToCpuCoreMapping().
static const std::map<int, int> &getGpuToCpuCoreMapping() {
  static std::map<int, int> mapping = inferGpuToCpuCoreMapping();
  return mapping;
}

/// Parse and validate the runtime.device_task_cpu_ids config (env var
/// MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS). Initialized once; aborts on
/// invalid input (matching the original MGPPrimitives.cpp behavior).
static ArrayRef<int> getDeviceTaskCpuIds() {
  static SmallVector<int, 8> deviceTaskCpuIds = []() {
    SmallVector<int, 8> result;
    auto config = Config::open();
    if (config.isError())
      return result;

    SmallVector<StringRef, 8> strs;
    config->getValueAsList("runtime.device_task_cpu_ids", strs);
    for (size_t i = 0; i < strs.size(); i++) {
      int cpuId;
      if (strs[i].getAsInteger(10, cpuId)) {
        llvm::errs() << "Failed to parse runtime.device_task_cpu_ids[" << i
                     << "] as an integer: " << strs[i] << "\n";
        std::abort();
      }
      if (cpuId == 0) {
        llvm::errs()
            << "CPU ID 0 is not allowed in runtime.device_task_cpu_ids\n";
        std::abort();
      } else if (std::find(result.begin(), result.end(), cpuId) !=
                 result.end()) {
        llvm::errs()
            << "Duplicate CPU ID found in runtime.device_task_cpu_ids: "
            << cpuId << "\n";
        std::abort();
      } else {
        result.push_back(cpuId);
      }
    }
    return result;
  }();
  return deviceTaskCpuIds;
}

int taskIdForDevice(int deviceId, size_t numWorkers) {
  if (numWorkers <= 1)
    return kDefaultTaskId;

  // Tier 1: explicit per-device CPU IDs from config/env.
  auto deviceTaskCpuIds = getDeviceTaskCpuIds();
  if (deviceTaskCpuIds.size() > static_cast<size_t>(deviceId)) {
    int taskId = deviceTaskCpuIds[deviceId];
    // A worker ID indexes the worker array, so an out-of-range entry would
    // read past it. This mapping is user-supplied, so an invalid entry is an
    // ill-formed config: report it rather than silently placing the device
    // somewhere else.
    if (taskId >= static_cast<int>(numWorkers)) {
      llvm::errs() << "runtime.device_task_cpu_ids[" << deviceId
                   << "] = " << taskId
                   << " is not a valid worker ID for a pool of " << numWorkers
                   << " threads\n";
      std::abort();
    }
    return taskId;
  }

  // Tier 2: NUMA-inferred GPU-to-CPU core mapping.
  // Only use when numWorkers matches physical core count, indicating the
  // thread pool wasn't limited by cgroups or other means.
  if (numWorkers >= M::getNumPhysicalCores()) {
    const auto &gpuToCpuCoreMapping = getGpuToCpuCoreMapping();
    if (!gpuToCpuCoreMapping.empty()) {
      auto it = gpuToCpuCoreMapping.find(deviceId);
      // The NUMA query reports raw /sys CPU IDs, which should not exceed the
      // worker count but can, because there is no strong mapping from CPU core
      // IDs to worker IDs. Fall through to the round-robin tier when it does.
      // TODO(GEX-4049): drop this guard once that mapping exists.
      if (it != gpuToCpuCoreMapping.end() &&
          it->second < static_cast<int>(numWorkers))
        return it->second;
    }
  }

  // Tier 3: round-robin across workers 1..N-1 (skip worker 0).
  return 1 + (deviceId % static_cast<int>(numWorkers - 1));
}

/// Sorted, deduplicated worker IDs that taskIdForNonDevice must skip: worker
/// 0 plus the tier 1 and tier 2 candidates, which can name any worker.
///
/// Tier 3 candidates are deliberately absent. That tier assigns devices
/// contiguously upward from worker 1, and taskIdForNonDevice allocates
/// downward from the top of the pool, so the two only meet once every other
/// worker is spoken for.
static ArrayRef<int> reservedTaskIds() {
  static SmallVector<int> reserved = [] {
    SmallVector<int> ids;

    // Treat taskID 0 as a device thread as well to avoid assigning to thread 0,
    // which is the MEF executor thread.
    ids.push_back(0);

    llvm::append_range(ids, getDeviceTaskCpuIds());
    for (auto [_, cpuId] : getGpuToCpuCoreMapping())
      ids.push_back(cpuId);

    llvm::sort(ids);
    // Two GPUs can share a NUMA-local core, so duplicates must collapse: each
    // ID may displace the candidate only once.
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
  }();

  return reserved;
}

/// Returns the `n`-th (0-based) worker ID absent from sorted `reserved`,
/// counting down from `numWorkers - 1`.
static int nthNonReservedTaskId(int n, size_t numWorkers,
                                ArrayRef<int> reserved) {
  int candidate = static_cast<int>(numWorkers) - 1 - n;
  // Each in-range reserved ID at or above the candidate displaces it down by
  // one. Walk descending so one pass reaches the fixpoint.
  for (int id : llvm::reverse(reserved)) {
    if (id >= static_cast<int>(numWorkers))
      continue;
    if (id < candidate)
      break;
    --candidate;
  }
  return candidate;
}

int taskIdForNonDevice(int taskId, size_t numWorkers) {
  if (numWorkers == 0)
    return kDefaultTaskId;

  ArrayRef<int> reserved = reservedTaskIds();

  // Only reserved IDs below numWorkers consume a worker slot.
  size_t reservedInRange = std::lower_bound(reserved.begin(), reserved.end(),
                                            static_cast<int>(numWorkers)) -
                           reserved.begin();
  if (reservedInRange >= numWorkers)
    return kDefaultTaskId; // Every worker is reserved for a device.

  // Modulus over the usable count (not numWorkers) is what keeps the result in
  // range: counting down from numWorkers - 1, `usable` steps stay at or above
  // worker 0.
  size_t usable = numWorkers - reservedInRange;
  size_t slot = static_cast<size_t>(taskId) % usable;
  return nthNonReservedTaskId(static_cast<int>(slot), numWorkers, reserved);
}

} // namespace M::MLRT
