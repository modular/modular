//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_BUILD_PROJECT_H
#define MOJO_BUILD_PROJECT_H

namespace M {

class SubcommandRegistry;

/// Initializes the `build-project` subcommand and its various options, and
/// registers its callback function with the registry.
void registerBuildProjectSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // MOJO_BUILD_PROJECT_H
