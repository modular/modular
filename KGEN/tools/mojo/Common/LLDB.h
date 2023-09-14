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

/// Returns the path to a suitable `lldb` executable that can be used to launch
/// the REPL, or an error if none exists.
llvm::ErrorOr<std::string> getLLDB(const std::string &executable);

/// Invokes an LLDB process with the provided arguments.
int invokeLLDB(const State &state, llvm::opt::InputArgList &args,
               std::initializer_list<StringRef> extraOptions);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_LLDB_H
