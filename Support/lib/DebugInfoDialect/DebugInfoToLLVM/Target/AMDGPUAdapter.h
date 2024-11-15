//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_AMDGPUADAPTER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_AMDGPUADAPTER_H

#include "TargetAdapter.h"

namespace M::DebugInfo {
/// Adapter for AMDGPU backend.
TargetAdapter getAMDGPUAdapter(bool tradeoffPerfForVariableDI);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_AMDGPUADAPTER_H
