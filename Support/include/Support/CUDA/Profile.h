//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines functions to control the CUDA profiler.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CUDA_PROFILER_H
#define SUPPORT_CUDA_PROFILER_H

#include "Support/ErrorOr.h"

namespace M::CUDA {
/// Enables profile collection by the active profiling tool for the current
/// context. If profiling is already enabled, then the function has no effect.
ErrorOrSuccess profileStart();

/// Disables profile collection by the active profiling tool for the current
/// context. If profiling is already disabled, then the function has no effect.
ErrorOrSuccess profileStop();
} // namespace M::CUDA

#endif // SUPPORT_CUDA_PROFILER_H
