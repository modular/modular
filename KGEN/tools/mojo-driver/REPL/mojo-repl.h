//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_REPL_H
#define MOJO_REPL_H

namespace M {

class SubcommandRegistry;

/// Initializes the `repl` subcommand and its various options, and registers
/// its callback function with the registry.
void registerREPLSubCommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_REPL_H
