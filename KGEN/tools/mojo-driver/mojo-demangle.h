//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEMANGLE_H
#define MOJO_DEMANGLE_H

namespace M {

class SubCommandRegistry;

/// Initializes the `demangle` subcommand and its various options, and registers
/// its callback function wih the registry.
void registerDemangleSubCommand(SubCommandRegistry &registry);
} // namespace M

#endif // MOJO_DEMANGLE_H
