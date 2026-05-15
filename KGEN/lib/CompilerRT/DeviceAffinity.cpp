//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/DeviceAffinity.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "Support/SymbolExport.h"

using namespace M::MLRT;

/// Compute the worker task ID for a GPU device. Called from Mojo via
/// external_call. Obtains the thread pool size internally from the
/// current AsyncRT cpuDevice.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int32_t
KGEN_CompilerRT_TaskIdForDevice(int32_t deviceId) {
  auto *rt = CPUDevice::getCurrentCPUDeviceOrNull();
  size_t numWorkers = rt->getWorkQueue()->getParallelismLevel();
  return taskIdForDevice(deviceId, numWorkers);
}
