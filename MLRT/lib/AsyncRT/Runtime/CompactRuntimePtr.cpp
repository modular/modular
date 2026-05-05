//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Runtime/CompactRuntimePtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_os_ostream.h"

#include <mutex>

using namespace M;
using namespace M::MLRT;

Detail::CPUDeviceTable::CPUDeviceTable() {
  freeIndices.resize(kInvalidIndex);
  for (uint8_t i = 0; i < kInvalidIndex; ++i) {
    allCPUDevices[i] = nullptr;
    freeIndices[i] = kInvalidIndex - i - 1;
  }
}

CPUDevice *Detail::CPUDeviceTable::getCPUDevice(uint8_t index) const {
  assert(index != kInvalidIndex && "invalid CPUDevice index");
  assert(allCPUDevices[index] != nullptr &&
         "no CPUDevice has been registered for index");
  // NOTE: We are assuming the mutex lock will force all writes to allCPUDevices
  // to be flushed.
  return allCPUDevices[index];
}

uint8_t Detail::CPUDeviceTable::reserveIndex() {
  std::lock_guard<std::mutex> lock(mu);
  assert(!freeIndices.empty() && "too many CPUDevices are currently active");
  auto index = freeIndices.pop_back_val();
  assert(allCPUDevices[index] == nullptr &&
         "index is still occupied by a CPUDevice");
  return index;
}

void Detail::CPUDeviceTable::setCPUDevice(uint8_t index, CPUDevice *cpuDevice) {
  // NOTE: Take the lock to ensure writes to allCPUDevices are flushed.
  std::lock_guard<std::mutex> lock(mu);
  allCPUDevices[index] = cpuDevice;
}

void Detail::CPUDeviceTable::clearCPUDevice(uint8_t index) {
  std::lock_guard<std::mutex> lock(mu);
  assert(allCPUDevices[index] != nullptr &&
         "no CPUDevice has been registered for index");
  assert(freeIndices.size() < kInvalidIndex && "all indices are already free");
  allCPUDevices[index] = nullptr;
  freeIndices.push_back(index);
}

size_t Detail::CPUDeviceTable::numActiveCPUDevices() const {
  std::lock_guard<std::mutex> lock(mu);
  return kInvalidIndex - freeIndices.size();
}
