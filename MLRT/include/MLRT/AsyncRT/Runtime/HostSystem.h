//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_RUNTIME_HOST_SYSTEM_H
#define MLRT_ASYNCRT_RUNTIME_HOST_SYSTEM_H

#include "MLRT/AsyncRT/Runtime/CPUDevice.h"
#include "Support/SymbolExport.h"

namespace M::MLRT {

/// Returns a reference to the process-wide global AsyncRT CPUDevice, creating
/// it on first use with \p source and \p options. If a global CPUDevice already
/// exists, triggers a fatal error if \p options do not match those used at
/// creation, and returns a copy of the existing reference.
/// \p allowUsingExistingOptions may be set to true to disable the check that
/// the CPUDevice options match and discard the provided options, but the caller
/// should ensure that it is safe to do so.
MODULAR_CXX_EXPORT CPUDeviceRef
getOrCreateCPUDevice(CPUDeviceSource source,
                     const CPUDeviceOptions &options = CPUDeviceOptions(),
                     bool allowUsingExistingOptions = false);

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_RUNTIME_HOST_SYSTEM_H
