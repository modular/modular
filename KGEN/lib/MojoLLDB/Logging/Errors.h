//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_LOGGING_ERRORS_H
#define KGEN_LIB_MOJOLLDB_LOGGING_ERRORS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "lldb/Core/Debugger.h"

/// Utility to be used for printing bug report messages in a consistent manner.
/// See the MojoLLDB `README.md` for more information on this macro.
///
/// It receives the same parameters as `llvm::formatv` and the first character
/// is expected to be lowercase.
#define EMIT_BUG_REPORT_MESSAGE(format, ...)                                   \
  do {                                                                         \
    lldb_private::Debugger::ReportError(                                       \
        llvm::formatv(                                                         \
            format "\nPlease submit a bug report to "                          \
                   "https://github.com/modular/modular/issues and include "    \
                   "steps for reproduction and all relevant source code.\n",   \
            __VA_ARGS__)                                                       \
            .str());                                                           \
  } while (0)

#endif // KGEN_LIB_MOJOLLDB_LOGGING_ERRORS_H
