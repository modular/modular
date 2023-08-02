//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PROCESS_H
#define SUPPORT_PROCESS_H

#include "Support/LLVMForwardDecls.h"

namespace M {
/// Set the environment variable `name` to `value`. If `overwrite` is false, the
/// variable will not be set if it already exists.
/// TODO: This should be upstreamed to llvm::sys::Process to match the GetEnv
///       method.
LogicalResult setProcessEnv(StringRef name, StringRef value,
                            bool overwrite = true);
} // namespace M

#endif // SUPPORT_PROCESS_H
