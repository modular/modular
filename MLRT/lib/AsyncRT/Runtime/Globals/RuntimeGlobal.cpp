//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/Globals/RuntimeGlobal.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"

#include <mutex>

namespace M::MLRT {

namespace {

static std::mutex &getGlobalCPUDeviceMutexImpl() {
  static std::mutex m;
  return m;
}

static CPUDevice *&getGlobalCPUDevicePtrImpl() {
  static CPUDevice *ptr = nullptr;
  return ptr;
}

static CPUDeviceOptions &storedGlobalCPUDeviceCreationOptionsImpl() {
  static CPUDeviceOptions opts;
  return opts;
}

} // namespace

std::mutex &getGlobalCPUDeviceMutex() { return getGlobalCPUDeviceMutexImpl(); }

CPUDevice *getGlobalCPUDevicePointer() { return getGlobalCPUDevicePtrImpl(); }

void setGlobalCPUDevicePointer(CPUDevice *ptr) {
  getGlobalCPUDevicePtrImpl() = ptr;
}

void clearGlobalCPUDevicePointerIfEquals(CPUDevice *ptr) {
  std::lock_guard<std::mutex> lock(getGlobalCPUDeviceMutexImpl());
  if (getGlobalCPUDevicePtrImpl() == ptr) {
    getGlobalCPUDevicePtrImpl() = nullptr;
  }
}

CPUDeviceOptions &getStoredGlobalCPUDeviceCreationOptions() {
  return storedGlobalCPUDeviceCreationOptionsImpl();
}

} // namespace M::MLRT
