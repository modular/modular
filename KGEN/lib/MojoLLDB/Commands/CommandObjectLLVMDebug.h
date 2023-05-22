//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H
#define KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H

#include "lldb/API/LLDB.h"

namespace M::KGEN::Mojo {
/// Register all related `llvm-debug` commands in the given debugger.
void registerLLVMDebugCommands(lldb::SBDebugger debugger);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTLLVMDEBUG_H
