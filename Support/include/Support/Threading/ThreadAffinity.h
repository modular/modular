//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_THREADING_THREADAFFINITY_H
#define SUPPORT_THREADING_THREADAFFINITY_H

#include "Support/ForwardDecls.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstddef>

namespace M {
//===----------------------------------------------------------------------===//
// Thread affinity
//===----------------------------------------------------------------------===//

/// Returns true if thread affinity is available on this target.
bool haveThreadAffinity();

/// Sets the execution affinity to the specified CPU core and sets the memory
/// policy to the NUMA node that CPU core resides in if possible.
ErrorOrSuccess setThreadAffinity(size_t cpuID);

/// Executes workFn with thread execution affinity to the specified CPU core and
/// sets the memory policy to the NUMA node that CPU core resides in if
/// possible.
ErrorOrSuccess runWithThreadAffinity(size_t cpuID,
                                     llvm::function_ref<void()> workFn);
} // namespace M

#endif // SUPPORT_THREADING_THREADAFFINITY_H
