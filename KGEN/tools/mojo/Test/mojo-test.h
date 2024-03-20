//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_TEST_H
#define MOJO_TEST_H

namespace M {

class SubcommandRegistry;

/// Initializes the `test` subcommand and its various options, and registers its
/// callback function with the registry.
void registerTestSubcommand(SubcommandRegistry &registry);
} // namespace M

#endif // MOJO_TEST_H
