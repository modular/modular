//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_RUN_H
#define MOJO_RUN_H

namespace M {

class SubcommandRegistry;

/// Initializes the `run` subcommand and its various options, and registers its
/// callback function with the registry.
void registerRunSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // MOJO_RUN_H
