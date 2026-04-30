//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_PRECOMPILE_H
#define MOJO_PRECOMPILE_H

namespace M {

class SubcommandRegistry;

/// Initializes the `precompile` subcommand and its various options, and
/// registers its callback function with the registry.
void registerPrecompileSubcommand(SubcommandRegistry &registry);

} // namespace M

#endif // MOJO_PRECOMPILE_H
