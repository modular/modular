//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H
#define KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H

#include "lldb/API/LLDB.h"

namespace M {
namespace LLCL {
class Runtime;
} // namespace LLCL
} // namespace M

namespace M::KGEN::Mojo {
/// Register all related `mojo` commands in the given debugger.
void registerMojoCommands(lldb::SBDebugger debugger, LLCL::Runtime &runtime);
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_REPL_COMMANDOBJECTMOJO_H
