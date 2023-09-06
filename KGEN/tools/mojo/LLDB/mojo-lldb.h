//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_LLDB_H
#define MOJO_LLDB_H

namespace M {

class SubcommandRegistry;

/// Initializes the `lldb` subcommand and its various options, and registers
/// its callback function with the registry.
void registerLLDBSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_LLDB_H
