//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_THREADING_THREADAFFINITY_H
#define SUPPORT_THREADING_THREADAFFINITY_H

#include "Support/ForwardDecls.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace M {
//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

/// Returns true if thread affinity is available on this target.
bool haveThreadAffinity();

/// Attempts to sets the caller's thread affinity to the given CPU id. Returns
/// error if affinity is not supported on this target or the operation fails.
ErrorOrSuccess setThreadAffinity(size_t cpuID);

/// Attempts to runs workFn with caller's thread affinity set to the given CPU
/// id. Returns error if thread affinity is not supported on this target
/// or the operation fails.
ErrorOrSuccess runWithThreadAffinity(size_t cpuID,
                                     llvm::function_ref<void()> workFn);
} // namespace M

#endif // SUPPORT_THREADING_THREADAFFINITY_H
