//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_LLDB_H
#define KGEN_TOOLS_MOJO_COMMON_LLDB_H

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"

namespace M {

/// Invokes an LLDB process with the provided arguments.
int invokeLLDB(const State &state, ArrayRef<std::string> lldbArgs,
               ArrayRef<std::string> runArgs = {}, bool dryRun = false);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_LLDB_H
