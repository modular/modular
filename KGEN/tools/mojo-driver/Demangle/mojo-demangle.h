//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEMANGLE_H
#define MOJO_DEMANGLE_H

namespace M {

class SubcommandRegistry;

/// Initializes the `demangle` subcommand and its various options, and registers
/// its callback function with the registry.
void registerDemangleSubCommand(SubcommandRegistry &registry);
} // namespace M

#endif // MOJO_DEMANGLE_H
