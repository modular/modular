//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/RuntimeManager.h"
#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "Support/Threading/HWInfo.h"

#include "llvm/Support/ErrorHandling.h"

#include <mutex>
#include <string>

namespace M::MLRT {

CPUDeviceRef getOrCreateCPUDevice(CPUDeviceSource source,
                                  const CPUDeviceOptions &options,
                                  bool allowUsingExistingOptions) {
  std::lock_guard<std::mutex> lock(getGlobalCPUDeviceMutex());
  CPUDevice *existingCPUDevice = getGlobalCPUDevicePointer();
  if (existingCPUDevice) {
    if (getStoredGlobalCPUDeviceCreationOptions() != options &&
        !allowUsingExistingOptions)
      llvm::report_fatal_error(
          "MLRT::getOrCreateCPUDevice called requesting different options to "
          "those used to create the existing CPUDevice.");
    return CPUDeviceRef::copy(existingCPUDevice);
  }

  assert(CPUDevice::getCurrentCPUDeviceOrNull() == nullptr &&
         "creating a CPUDevice from a thread already associated with an outer "
         "CPUDevice");
  CompactCPUDevicePtr cpuDevicePtr = CompactCPUDevicePtr::reserve();
  std::unique_ptr<Allocator> allocator =
      getAllocator(options.getAllocatorOptions());
  std::unique_ptr<WorkQueue> workQueue;
  switch (options.workQueueType) {
  case CPUDeviceOptions::WorkQueueType::kSingleThread:
    workQueue = createSingleThreadWorkQueue(cpuDevicePtr);
    break;
  case CPUDeviceOptions::WorkQueueType::kThreadPool:
    if (options.numaPartitioned) {
      // If NUMA partitioning is enabled create a partitioned work queue per
      // NUMA node and create a delegate work queue owning them.
      const ErrorOr<NUMATopology> &topologyOr = NUMATopology::get();
      if (topologyOr.isError())
        llvm::report_fatal_error(topologyOr.getError());
      const std::vector<int> &numaNodes = topologyOr->getNumaNodes();
      std::vector<std::unique_ptr<WorkQueue>> partitions;
      partitions.reserve(numaNodes.size());
      size_t globalWorkerIdOffset = 0;
      auto busyWait = std::chrono::microseconds(options.threadBusyWaitTime);
      for (size_t i = 0; i < numaNodes.size(); ++i) {
        std::string partitionName =
            options.poolName + " (NUMA " + std::to_string(i) + ")";
        partitions.push_back(createPartitionedThreadPoolWorkQueue(
            cpuDevicePtr, numaNodes[i], busyWait, partitionName,
            globalWorkerIdOffset));
        globalWorkerIdOffset += partitions.back()->getParallelismLevel();
      }
      workQueue = createDelegateThreadPoolWorkQueue(std::move(partitions));
    } else {
      workQueue = createThreadPoolWorkQueue(
          cpuDevicePtr, options.numThreads, options.maxThreads,
          options.mainWillDonate, options.withAffinity,
          std::chrono::microseconds(options.threadBusyWaitTime),
          options.poolName);
    }
    break;
  }
  CPUDeviceRef newCPUDevice = CPUDeviceRef::take(new CPUDevice(
      cpuDevicePtr, std::move(allocator), std::move(workQueue), source,
      options.profileFilename, options.runtimeProfilingTypeMask,
      options.profilerDebuginfo));

  getStoredGlobalCPUDeviceCreationOptions() = options;

  // Initialise the NUMA topology, to be used later when creating allocators and
  // work-queues.
  (void)NUMATopology::get();

  setGlobalCPUDevicePointer(newCPUDevice.getPointer());
  return newCPUDevice.copy();
}

} // namespace M::MLRT
