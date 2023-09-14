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
int invokeLLDB(const State &state, llvm::opt::InputArgList &args,
               std::initializer_list<StringRef> extraOptions);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_LLDB_H
