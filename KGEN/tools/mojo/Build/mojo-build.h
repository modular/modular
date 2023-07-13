//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_BUILD_H
#define MOJO_BUILD_H

namespace M {

class SubcommandRegistry;

/// Initializes the `build` subcommand and its various options, and registers
/// its callback function with the registry.
void registerBuildSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // MOJO_BUILD_H
