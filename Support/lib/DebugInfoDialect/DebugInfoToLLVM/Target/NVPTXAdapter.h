//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H

#include "TargetAdapter.h"

namespace M::DebugInfo {
/// Adapter for NVPTX backend.
TargetAdapter getNVPTXAdapter();
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_NVPTXADAPTER_H
