//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEBUG_H
#define MOJO_DEBUG_H

namespace M {

class SubcommandRegistry;

/// Initializes the `debug` subcommand and its various options, and registers
/// its callback function with the registry.
void registerDebugSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_DEBUG_H
