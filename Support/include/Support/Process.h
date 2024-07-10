//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROCESS_H
#define SUPPORT_PROCESS_H

#include "Support/LLVMForwardDecls.h"
#include <vector>

namespace M {
/// Set the environment variable `name` to `value`. If `overwrite` is false, the
/// variable will not be set if it already exists.
/// TODO: This should be upstreamed to llvm::sys::Process to match the GetEnv
///       method.
LogicalResult setProcessEnv(StringRef name, StringRef value,
                            bool overwrite = true);

/// Get a list of all the environment variables of the current process.
std::vector<StringRef> getEnv();

//===----------------------------------------------------------------------===//
// Memory usage
//===----------------------------------------------------------------------===//

/// Returns the current process' physical memory usage, or 0 if value is
/// not available. Generally determined from the OS's reported resident
/// page value, and may not very reliable.
size_t getProcessPhysicalMemUsage();
} // namespace M

#endif // SUPPORT_PROCESS_H
