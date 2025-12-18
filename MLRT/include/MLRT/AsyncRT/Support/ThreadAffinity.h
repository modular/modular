//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_SUPPORT_THREADAFFINITY_H
#define ASYNCRT_SUPPORT_THREADAFFINITY_H

#include "Support/ErrorOr.h"
#include "Support/Threading/HWInfo.h"

#include "llvm/ADT/FunctionExtras.h"

#include <cstddef>
#include <vector>

namespace M::AsyncRT {

/// Determine the number of threads to use (based on the existing suggestion),
/// and return a vector of CPU IDs for every such thread. The CPU ids may be
/// kNoAffinity, indicating no affinity should be set. On error attempt to
/// fallback to defaults, and return error to the caller if the attempt
/// fails. If withAffinity is false, then expected result is a vector
/// containing all entries with kNoAffinity.
M::ErrorOr<std::vector<size_t>> getThreadAffinityCpuIds(bool withAffinity,
                                                        size_t numThreads,
                                                        size_t maxThreads);

/// Execute workFn with affinity to cpuID if it is not kNoAffinity.
/// Gracefully and silently execute workFn directly if errors.
void runWithThreadAffinity(size_t cpuID, llvm::function_ref<void()> workFn);

/// Set the current thread's affinity to cpuID if it is not kNoAffinity.
/// Gracefully and silently continue if errors.
void setThreadAffinity(size_t cpuID);

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_THREADAFFINITY_H
