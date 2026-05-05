//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Global Runtime pointer and mutex. Implemented in the RuntimeGlobals shared
// library so there is a single definition per process (ODR). M::MLRT
// uses these (e.g. MLRT::getOrCreateCPUDevice) to manage the single global
// cpuDevice.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
#define MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H

#include "Support/SymbolExport.h"

#include <mutex>

namespace M::MLRT {

class CPUDevice;
struct CPUDeviceOptions;

/// Returns the mutex that protects the global cpuDevice pointer.
MODULAR_CXX_EXPORT std::mutex &getGlobalCPUDeviceMutex();

/// Returns the current global cpuDevice pointer, or nullptr if none set.
/// Caller must hold getGlobalCPUDeviceMutex().
MODULAR_CXX_EXPORT CPUDevice *getGlobalCPUDevicePointer();

/// Sets the global cpuDevice pointer. Caller must hold
/// getGlobalCPUDeviceMutex().
MODULAR_CXX_EXPORT void setGlobalCPUDevicePointer(CPUDevice *ptr);

/// If the global CPUDevice pointer equals \p ptr, clears it. Called from
/// CPUDevice::~CPUDevice() when the CPUDevice is destroyed.
MODULAR_CXX_EXPORT void clearGlobalCPUDevicePointerIfEquals(CPUDevice *ptr);

/// Options used when the global cpuDevice was first created (Init path). Caller
/// must hold getGlobalCPUDeviceMutex() when reading or writing.
MODULAR_CXX_EXPORT CPUDeviceOptions &getStoredGlobalCPUDeviceCreationOptions();

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_RUNTIME_GLOBALS_RUNTIMEGLOBAL_H
